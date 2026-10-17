"""Анализ одного видео целиком: детекторы → кэш признаков → сцена → [классификаторы сейфа и денег] →
[VLM: экран ноутбука, дверь] → правила → кадры-доказательства.

    from cashguard.pipeline import Analyzer
    an = Analyzer()                                   # модели грузятся один раз
    res = an.analyze("data/open/videos/Х-15.mov")     # dict по схеме API (ТЗ, раздел 10)

Кэш признаков пишется в data/cache/<job_id>.json, сцена строится build_scene(job_id),
поэтому правила можно перезапустить по кэшу без повторного прогона моделей.
"""
import base64
import importlib.util
import json
import logging
import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from .detectors import CocoDetector, OwlDetector
from .labels import ROOT
from .registry import REGISTRY, describe, refs
from .timeline import build_timeline
from .rules import detect
from .scene import BAG_Q, CACHE, CUP_Q, MONEY_Q, PERSONAL_Q, build_scene, center, inside, iou
from .video import read_frame

log = logging.getLogger("cashguard.pipeline")

# Частоты детекторов. 2.5/1 кадр/с вместо 5/2: на обучающих роликах качество то же
# (14/21, 0 ложных, проверено прореживанием кэша), а обработка ~в 2 раза быстрее — запас по лимиту 5 минут.
FPS = float(os.environ.get("CASHGUARD_FPS", "2.5"))          # RT-DETR
OWL_FPS = float(os.environ.get("CASHGUARD_OWL_FPS", "1.0"))  # OWLv2
MAX_FRAMES = 3     # кадров-доказательств на нарушение (ТЗ: 1–3)
JPEG_QUALITY = 85
OBJ_THR = 0.2      # порог OWL для рамок предметов на кадре, как в scripts/debug_scene.py
# необязательные классификаторы состояний (DINOv2 + логрегрессия, cashguard/states.py)
SAFE_CLF = ROOT / "models" / "safe_clf.joblib"    # «сейф открыт» — правило B
MONEY_CLF = ROOT / "models" / "money_clf.joblib"  # «деньги на столе» — правила A и C
# локальная VLM (Qwen3-VL-4B) в моменты событий: L — экран не заблокирован при уходе, H — дверь не закрыта
VLM_QIDS = ("laptop", "door")
VLM_RULES = "LH"
MAX_VLM_QUESTIONS = 12   # обычно 0–5 вопросов по ~1 с; потолок бережёт лимит ТЗ в 5 минут

# BGR
COL_ROLE = {"cashier": (255, 128, 0), "visitor": (0, 0, 255)}
COL_OBJ = {"money": (0, 200, 0), "bag": (0, 140, 255), "personal": (255, 0, 255), "cup": (255, 255, 0)}


def _load_extract():
    """extract() из scripts/extract_features.py: тот же формат кэша, который читает build_scene."""
    spec = importlib.util.spec_from_file_location(
        "_cashguard_extract_features", ROOT / "scripts" / "extract_features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float):
        return round(x, 3)
    return x


def safe_p_at(sc, t, tol=0.6):
    """p(сейф открыт) от классификатора в ближайший момент к t, если ряд есть."""
    sp = getattr(sc, "safe_prob", None)
    if not sp:
        return None
    ts = np.array([x[0] for x in sp])
    k = int(np.argmin(np.abs(ts - t)))
    return float(sp[k][1]) if abs(ts[k] - t) <= tol else None


def obj_kind(q):
    return ("money" if q in MONEY_Q else "bag" if q in BAG_Q else
            "personal" if q in PERSONAL_Q else "cup" if q in CUP_Q else None)


def money_p_at(money_seen, t, tol=0.3):
    """p(деньги на столе) от классификатора, если правила запрашивали момент рядом с t."""
    if not money_seen:
        return None
    k = min(money_seen, key=lambda x: abs(x - t))
    return money_seen[k] if abs(k - t) <= tol else None


def vlm_answer(cache, qid, field, t, tol=0.6):
    """Ответ VLM (поле field) на вопрос qid в ближайший к t момент, если спрашивали."""
    if cache is None:
        return None
    best = None
    for k, v in cache.data.items():
        q, _, ts = k.partition("@")
        if q != qid or not ts or not isinstance(v, dict) or field not in v:
            continue
        dt = abs(float(ts) - t)
        if dt <= tol and (best is None or dt < best[0]):
            best = (dt, v[field])
    return best[1] if best else None


def draw_evidence(fr, sc, t, money_seen=None, vlm_cache=None):
    """Рамки на кадре: стол, дверь, сейф, люди в помещении (роль), предметы на столе."""
    def rect(b, c, th=2, label=None, below=False):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(fr, (x1, y1), (x2, y2), c, th)
        if label:
            (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = min(fr.shape[0], y2 + tht + 4) if below else max(tht + 4, y1)
            cv2.rectangle(fr, (x1, ty - tht - 4), (x1 + tw + 4, ty), c, -1)
            cv2.putText(fr, label, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                        cv2.LINE_AA)

    if sc.door:
        a = vlm_answer(vlm_cache, "door", "door", t)
        # подпись под рамкой: дверь обычно начинается у верхнего края, там плашка времени
        rect(sc.door, (0, 0, 255) if a == "open" else (190, 190, 190), 2 if a == "open" else 1,
             "door" if a is None else f"door: {a}", below=True)
    laptop = getattr(sc, "laptop", None)
    if laptop:
        a = vlm_answer(vlm_cache, "laptop", "screen", t)
        if a is not None:  # рамку ноутбука показываем, только если о нём спрашивали VLM
            rect(laptop, (0, 0, 255) if a == "unlocked" else (200, 200, 200), 2, f"screen: {a}", below=True)
    if sc.desk:
        # подпись под рамкой: сверху стол перекрыт рамкой кассира
        pm = money_p_at(money_seen, t)
        rect(sc.desk, (255, 255, 255), 1, "desk" if pm is None else f"desk: money {pm:.2f}", below=True)
    if sc.safe:
        p = safe_p_at(sc, t)
        # подпись под рамкой: сверху сейф обычно перекрыт рамкой кассира
        if p is None:
            rect(sc.safe, (200, 200, 0), 1, "safe", below=True)
        elif p >= 0.5:
            rect(sc.safe, (0, 0, 255), 2, f"safe open {p:.2f}", below=True)
        else:
            rect(sc.safe, (200, 200, 0), 1, f"safe closed {1 - p:.2f}", below=True)
    for tr in sc.tracks:
        if tr.start - 0.01 <= t <= tr.end + 0.01:
            b = sc.box_at(tr, t, tol=0.3)
            if b is not None:
                role = sc.roles.get(tr.tid, "visitor")
                rect(b, COL_ROLE.get(role, (0, 0, 255)), 2, f"{role} #{tr.tid}")
    if len(sc.owl_times) and sc.desk:
        k = int(np.argmin(np.abs(sc.owl_times - t)))
        if abs(sc.owl_times[k] - t) <= 0.6:
            drawn = []
            for q, s, *b in sorted(sc.owl[k], key=lambda d: -d[1]):
                kind = obj_kind(q)
                if kind is None or s < OBJ_THR or not inside(center(b), sc.desk, 0.05):
                    continue
                if any(kd == kind and iou(b, ob) > 0.5 for kd, ob in drawn):
                    continue
                drawn.append((kind, b))
                rect(b, COL_OBJ[kind], 2, f"{kind} {s:.2f}")
    stamp = f"{t:.1f} s"
    (tw, th), _ = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(fr, (0, 0), (tw + 12, th + 12), (0, 0, 0), -1)
    cv2.putText(fr, stamp, (6, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return fr


def jpeg_b64(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("не удалось закодировать кадр в JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def evidence_times(v, dur) -> list[float]:
    """Момент нарушения + моменты из Violation.evidence_t, по возрастанию, без повторов, <= 3."""
    hi = max(0.0, dur - 0.05)
    ts = sorted(min(max(0.0, float(x)), hi) for x in [v.t, *v.evidence_t])
    out = []
    for x in ts:
        if not out or x - out[-1] >= 0.4:
            out.append(x)
    if len(out) > MAX_FRAMES:  # оставить момент нарушения и крайние
        main = min(max(0.0, float(v.t)), hi)
        out = sorted({main, out[0], out[-1]})[:MAX_FRAMES]
    return out


def basis(vtype: str) -> str:
    """Почему это может быть нарушением: ссылка на норму с осторожной формулировкой."""
    r = REGISTRY.get(vtype)
    if r is None:
        return ""
    lead = "Возможное нарушение требования" if r.get("direct", True) else "Может не соответствовать требованию"
    return f"{lead}: {r['norm']}."


def _load_clf(name: str, env: str, path: Path):
    """Необязательный классификатор из cashguard/states.py. Нет модели или ошибка — правило без него."""
    if os.environ.get(env, "1") == "0":
        log.info("%s off: %s=0", name, env)
        return None
    if not path.exists():
        log.warning("%s off: %s not found", name, path)
        return None
    try:
        from . import states
        return getattr(states, name)(path=path)
    except Exception:  # noqa: BLE001 — необязательный шаг не должен ронять сервис
        log.exception("%s off: failed to load", name)
        return None


class MoneyProbe:
    """sc.money_fn для правил A и C: зовётся только в моменты событий. Запоминает ответы
    (повторные моменты не пересчитываются, ответы подписываются на кадрах-доказательствах);
    ошибка классификатора не роняет правила — дальше они работают только по OWL."""

    def __init__(self, clf, path, sc):
        self.clf, self.path, self.sc = clf, path, sc
        self.seen: dict[float, float] = {}
        self.sec = 0.0
        self.status = "ok"

    def __call__(self, ts) -> dict:
        keys = [round(float(t), 2) for t in ts]
        need = [t for t in keys if t not in self.seen]
        if need and self.status == "ok":
            t0 = time.perf_counter()
            try:
                self.seen.update(self.clf.at(self.path, self.sc, need))
            except Exception:  # noqa: BLE001
                log.exception("money classifier failed on %s", self.path)
                self.status = "error"
            self.sec += time.perf_counter() - t0
        return {t: self.seen[t] for t in keys if t in self.seen}


class Analyzer:
    def __init__(self, device=None):
        t0 = time.perf_counter()
        self.coco = CocoDetector(device=device)
        self.owl = OwlDetector(device=self.coco.dev)
        self.device = self.coco.dev
        self._extract = _load_extract()
        self.safe_clf = _load_clf("SafeClassifier", "CASHGUARD_SAFE_CLF", SAFE_CLF)
        self.money_clf = _load_clf("MoneyClassifier", "CASHGUARD_MONEY_CLF", MONEY_CLF)
        self.vlm = self._load_vlm()
        self.load_sec = round(time.perf_counter() - t0, 1)
        log.info("models loaded on %s in %.1fs (safe classifier: %s, money classifier: %s, vlm: %s)", self.device,
                 self.load_sec, "on" if self.safe_clf else "off", "on" if self.money_clf else "off",
                 "on" if self.vlm else "off")

    def _load_vlm(self):
        """Qwen3-VL-4B (~9 ГБ). Не загрузилась — правила L и H молчат, остальное работает."""
        if os.environ.get("CASHGUARD_VLM", "1") == "0":
            log.info("vlm off: CASHGUARD_VLM=0")
            return None
        try:
            from .vlm import VLM
            t0 = time.perf_counter()
            vlm = VLM(device=self.device)
            log.info("vlm loaded in %.1fs", time.perf_counter() - t0)
            return vlm
        except Exception:  # noqa: BLE001 — необязательный шаг не должен ронять сервис
            log.exception("vlm off: failed to load")
            return None

    def warmup(self):
        """Первый прогон на MPS/CUDA медленный (компиляция ядер) — делаем его до первой задачи."""
        t0 = time.perf_counter()
        fr = np.zeros((848, 464, 3), dtype=np.uint8)
        self.coco([fr])
        self.owl(fr)
        if self.safe_clf:
            self.safe_clf.emb([fr[:200, :200]])
        if self.money_clf:
            self.money_clf.emb(self.money_clf.tiles(fr[:300, :400]))
        if self.vlm:
            self.vlm.ask([fr[:224, :224]], "Reply with one word: ok", max_new_tokens=2)
        log.info("warmup %.1fs", time.perf_counter() - t0)

    def _safe_series(self, path, sc):
        """sc.safe_prob = [(t, p_open)]; при ошибке — пусто, статус в meta."""
        if self.safe_clf is None:
            return "off"
        if sc.safe is None:
            return "no_safe"
        try:
            sc.safe_prob = self.safe_clf.series(path, sc)
            return "ok"
        except Exception:  # noqa: BLE001
            log.exception("safe classifier failed on %s", path)
            sc.safe_prob = None
            return "error"

    def _vlm_queries(self, path, sc, job_id):
        """Вопросы VLM для правил L/H в моменты событий; кэш data/vlm_cache/<job_id>.json.
        Возвращает (cache | None, статус, число вопросов)."""
        if self.vlm is None:
            return None, "off", 0
        from .vlm import VLMCache
        from .vlm_plan import plan_queries
        cache = VLMCache(job_id)
        qs = [(qid, t) for qid, t in plan_queries(sc) if qid in VLM_QIDS]
        if len(qs) > MAX_VLM_QUESTIONS:
            log.warning("%s: %d vlm questions, keeping first %d", job_id, len(qs), MAX_VLM_QUESTIONS)
            qs = qs[:MAX_VLM_QUESTIONS]
        status, n = "ok", 0
        for qid, t in qs:
            fr = read_frame(path, t)
            if fr is None:
                continue
            try:
                cache.put(qid, t, self.vlm.ask_q([fr], qid, desk=sc.desk, safe=sc.safe,
                                                 laptop=getattr(sc, "laptop", None), door=sc.door))
                n += 1
            except Exception:  # noqa: BLE001 — правила получат то, что успели спросить
                log.exception("%s: vlm failed on %s@%.1f", job_id, qid, t)
                status = "error"
                break
        try:
            cache.save()
        except OSError:
            log.warning("%s: could not save vlm cache", job_id)
        return cache, status, n

    def run(self, video_path, job_id: str | None = None) -> dict:
        """Полный результат: поля API + пояснения для интерфейса (explain, meta)."""
        path = Path(video_path)
        job_id = job_id or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        feats = self._extract(path, self.coco, self.owl, FPS, OWL_FPS)
        if not feats["frames"]:
            raise ValueError("не удалось прочитать кадры видео (формат не поддерживается или файл повреждён)")
        last_t = feats["frames"][-1]["t"]
        if feats["info"]["dur"] < last_t:  # контейнер без числа кадров
            feats["info"]["dur"] = last_t + 1.0 / feats["info"]["fps"]
        t_extract = time.perf_counter() - t0

        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / f"{job_id}.json").write_text(json.dumps(feats), encoding="utf-8")
        sc = build_scene(job_id)
        t1 = time.perf_counter()
        safe_status = self._safe_series(path, sc)
        t_safe = time.perf_counter() - t1
        money = None
        if self.money_clf is not None and sc.desk is not None:
            money = MoneyProbe(self.money_clf, path, sc)
            sc.money_fn = money
        money_status = money.status if money else ("off" if self.money_clf is None else "no_desk")
        t2 = time.perf_counter()
        vlm_cache, vlm_status, vlm_n = self._vlm_queries(path, sc, job_id)
        t_vlm = time.perf_counter() - t2
        t3 = time.perf_counter()
        if vlm_cache is not None:
            vs = detect(sc, vlm_cache=vlm_cache, vlm_rules=VLM_RULES)
        else:
            vs = detect(sc)
        t_money = money.sec if money else 0.0
        t_rules = time.perf_counter() - t3 - t_money
        money_seen = money.seen if money else None

        violations, explain = [], []
        for v in vs:
            d = describe(v.type, v.t, v.details)
            frames = []
            for te in evidence_times(v, sc.dur):
                fr = read_frame(path, te)
                if fr is None and te > 0.3:
                    fr = read_frame(path, te - 0.3)
                if fr is None:
                    continue
                frames.append({"timestamp_sec": round(te, 1),
                               "image_base64": jpeg_b64(draw_evidence(fr, sc, te, money_seen, vlm_cache))})
            violations.append({
                "type": d["type"],
                "timecode_sec": round(float(v.t), 1),
                "description": f"{d['description']} {basis(v.type)}".strip(),
                "evidence_frames": frames,
            })
            explain.append({"code": v.type, "name": d["type"], "observed": d["description"],
                            "norm": d["norm"], "direct": REGISTRY.get(v.type, {}).get("direct", True),
                            "refs": refs(v.type),
                            "score": round(float(v.score), 3), "details": _jsonable(v.details)})
        if money:
            money_status = money.status
        t_total = time.perf_counter() - t0
        meta = {
            "job_id": job_id, "duration_sec": round(float(sc.dur), 1), "device": self.device,
            "fps": FPS, "owl_fps": OWL_FPS, "frames": len(feats["frames"]),
            "scene": {"desk": sc.desk is not None, "door": sc.door is not None, "safe": sc.safe is not None,
                      "people_tracks": len(sc.tracks), "safe_classifier": safe_status,
                      "money_classifier": money_status, "money_points": len(money_seen or {}),
                      "vlm": vlm_status, "vlm_questions": vlm_n},
            "timings": {"extract_sec": round(t_extract, 1), "safe_clf_sec": round(t_safe, 1),
                        "money_clf_sec": round(t_money, 1), "vlm_sec": round(t_vlm, 1), "rules_sec": round(t_rules, 2),
                        "frames_sec": round(t_total - t_extract - t_safe - t_money - t_vlm - t_rules, 2),
                        "total_sec": round(t_total, 1)},
        }
        log.info("%s: %.1fs video, %d violations, %.1fs (extract %.1fs, safe clf %s %.1fs, money clf %s %.1fs/%d pts, "
                 "vlm %s %.1fs/%d q)", job_id, sc.dur, len(violations), t_total, t_extract, safe_status, t_safe,
                 money_status, t_money, len(money_seen or {}), vlm_status, t_vlm, vlm_n)
        try:
            timeline = build_timeline(sc)
        except Exception as e:  # дорожки — пояснение для интерфейса, ошибка не должна ронять задачу
            log.warning("%s: timeline failed: %s", job_id, e)
            timeline = None
        return {"verdict": "violation" if violations else "no_violation", "violations": violations,
                "explain": explain, "meta": meta, "timeline": timeline}

    def analyze(self, video_path, job_id: str | None = None) -> dict:
        return to_api(self.run(video_path, job_id))


def to_api(full: dict) -> dict:
    """Только поля схемы API: verdict и violations[type, timecode_sec, description, evidence_frames]."""
    return {"verdict": full["verdict"], "violations": full["violations"]}


_default: Analyzer | None = None


def analyze(video_path, job_id: str | None = None) -> dict:
    """analyze(video_path) -> {"verdict", "violations"}; модели грузятся при первом вызове."""
    global _default
    if _default is None:
        _default = Analyzer()
    return _default.analyze(video_path, job_id)
