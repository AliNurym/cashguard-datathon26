"""Локальная VLM (Qwen3-VL, Apache-2.0) для проверки состояний сцены в моменты событий.

Работает полностью локально (transformers, MPS/CUDA/CPU). Ответы кэшируются в
data/vlm_cache/<vid>.json по ключу "<qid>@<t>", чтобы правила можно было перебирать
без повторных прогонов модели.
"""
import json
import re
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .detectors import pick_device
from .labels import ROOT, variant_dir

VLM_ID = "Qwen/Qwen3-VL-4B-Instruct"
VLM_CACHE = variant_dir("vlm_cache")

SCENE = ("This is a frame from an overhead CCTV camera in a bank cash room. "
         "The room has a cashier's desk with a laptop, a black banknote counting machine and a telephone, "
         "and a small grey metal safe standing next to the desk. ")

# Вопросы. Деньги ищем через grounding (координаты): на коротких да/нет вопросах модель
# систематически отвечает «нет», а на просьбу найти объект с рамкой — находит.
MONEY_GROUND = ("Locate the money in the image: banknotes, bills, a stack or bundle of cash, on the desk or "
                "in a hand. Output JSON list: [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"money\"}]. "
                "If there is no money, output [].")
QUESTIONS = {
    "desk": MONEY_GROUND,
    "count": MONEY_GROUND,
    "safe": (SCENE + "This crop shows the small grey metal safe. Is the safe door open? An open safe has its door "
             "swung to the side and the light interior or shelves are visible; a closed safe shows a flat grey "
             "front with a keyhole. Is a person standing right next to the safe with a hand on it? "
             'JSON only: {"safe_open": true/false, "person_at_safe": true/false}'),
    "id": ("Locate any identity document: an ID card or a passport, held in a hand or lying on the desk. "
           "Do not mark banknotes, the telephone or the laptop. Output JSON list: "
           '[{"bbox_2d": [x1, y1, x2, y2], "label": "id document"}]. If there is none, output [].'),
    "laptop": ("This is a close-up of a laptop on a bank cashier desk, seen from a CCTV camera. Is the laptop unlocked "
               "and showing a working application (a spreadsheet, a document, a program window), or is it locked "
               "(lock screen with a wallpaper picture or clock), turned off, black, or closed? "
               'JSON only: {"screen": "unlocked" | "locked" | "off" | "closed"}'),
    "door": ("This crop shows the glass door of a bank cash room, seen from a CCTV camera inside the room. Is the "
             "door open (swung open, a gap in the doorway, the door panel at an angle) or closed (the door panel flush "
             'in its frame)? JSON only: {"door": "open" | "closed"}'),
    "machine": ("Locate the black banknote counting machine on the desk. Output JSON list: "
                '[{"bbox_2d": [x1, y1, x2, y2], "label": "counting machine"}]. If there is none, output [].'),
}
GROUND_Q = {"desk": "cash_on_desk", "count": "cash_visible", "id": "id_visible", "machine": "machine_found"}


def crop_region(frame, box, pad_x=0.3, pad_top=0.3, pad_bottom=0.3, min_side=448):
    """Кроп зоны с полями и апскейл до min_side по короткой стороне (мелкие объекты крупнее)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - pad_x * bw)), int(min(w, x2 + pad_x * bw))
    y1, y2 = int(max(0, y1 - pad_top * bh)), int(min(h, y2 + pad_bottom * bh))
    crop = frame[y1:y2, x1:x2]
    s = min_side / max(1, min(crop.shape[:2]))
    if s > 1:
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    return crop


def region_for(qid, desk, safe, laptop=None, door=None):
    """Какую зону показывать модели для вопроса (None — весь кадр)."""
    if qid == "door":
        return crop_kwargs(door, 0.1, 0.05, 0.05) if door else None
    if qid == "laptop":
        return crop_kwargs(laptop, 0.3, 0.3, 0.3) if laptop else None
    if qid == "safe" and safe:
        return crop_kwargs(safe, 0.8, 0.5, 0.4)
    if qid == "desk" and desk:
        return crop_kwargs(desk, 0.05, 0.15, 0.0)
    if qid in ("count", "id") and desk:
        return crop_kwargs(desk, 0.05, 0.6, 0.0)  # выше стола — руки кассира
    return None


def crop_kwargs(box, px, pt, pb):
    return {"box": box, "pad_x": px, "pad_top": pt, "pad_bottom": pb}


def to_pil(img_bgr, max_side=None):
    if max_side:
        h, w = img_bgr.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)))
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return {}
    s = m.group(0).replace("True", "true").replace("False", "false")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def parse_boxes(text: str) -> list:
    """Рамки из grounding-ответа Qwen-VL: координаты нормированы к 0..1000 по входному изображению."""
    out = []
    for m in re.finditer(r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text):
        out.append([float(v) for v in m.groups()])
    return out


class VLM:
    def __init__(self, model_id=VLM_ID, device=None):
        self.dev = device or pick_device()
        dtype = torch.bfloat16 if self.dev == "cuda" else torch.float16
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(self.dev).eval()

    @torch.no_grad()
    def ask(self, images_bgr, prompt, max_new_tokens=96) -> str:
        content = [{"type": "image", "image": to_pil(im)} for im in images_bgr]
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_dict=True, return_tensors="pt").to(self.dev)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def ask_q(self, images_bgr, qid, desk=None, safe=None, laptop=None, door=None) -> dict:
        """Вопрос qid по кадру(ам); если известны боксы стола/сейфа/ноутбука — показываем кроп зоны."""
        t0 = time.time()
        if (qid == "laptop" and laptop is None) or (qid == "door" and door is None):
            return {"_skip": "no box"}
        reg = region_for(qid, desk, safe, laptop, door)
        if reg:
            images_bgr = [crop_region(im, **reg) for im in images_bgr]
        raw = self.ask(images_bgr, QUESTIONS[qid])
        if qid in GROUND_Q:
            boxes = parse_boxes(raw)
            ans = {GROUND_Q[qid]: bool(boxes), "boxes": boxes}
        else:
            ans = parse_json(raw)
        ans["_raw"] = raw.strip()[:200]
        ans["_sec"] = round(time.time() - t0, 2)
        return ans


class VLMCache:
    """Кэш ответов VLM на диске для одного видео."""

    def __init__(self, vid):
        VLM_CACHE.mkdir(parents=True, exist_ok=True)
        self.path = VLM_CACHE / f"{vid}.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    @staticmethod
    def key(qid, t):
        return f"{qid}@{t:.1f}"

    def get(self, qid, t):
        return self.data.get(self.key(qid, t))

    def put(self, qid, t, ans):
        self.data[self.key(qid, t)] = ans

    def save(self):
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")

    def series(self, qid, field):
        """[(t, bool)] по всем закэшированным моментам для вопроса."""
        out = []
        for k, v in self.data.items():
            if "@" not in k:
                continue  # служебные ключи (_machine_box, _machine_activity)
            q, t = k.split("@")
            if q == qid and field in v:
                out.append((float(t), bool(v[field])))
        return sorted(out)
