"""Локальные детекторы: RT-DETRv2 (COCO, Apache-2.0) и OWLv2 (open-vocabulary, Apache-2.0)."""
import cv2
import torch
from PIL import Image
from transformers import (AutoImageProcessor, AutoModelForObjectDetection,
                          Owlv2ForObjectDetection, Owlv2Processor)

RTDETR_ID = "PekingU/rtdetr_v2_r50vd"
OWL_ID = "google/owlv2-base-patch16-ensemble"

OWL_QUERIES = [
    "banknotes", "a stack of paper money", "a green cloth bag", "a coffee cup",
    "a handbag", "a coat", "a mobile phone", "a glass door", "an office desk",
    "a banknote counting machine", "a laptop", "a metal safe", "a grey metal cabinet",
    "an open cabinet door",
]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def to_pil(frame_bgr):
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


class CocoDetector:
    def __init__(self, device=None, threshold=0.3):
        self.dev = device or pick_device()
        self.proc = AutoImageProcessor.from_pretrained(RTDETR_ID)
        self.model = AutoModelForObjectDetection.from_pretrained(RTDETR_ID).to(self.dev).eval()
        self.id2label = self.model.config.id2label
        self.thr = threshold

    @torch.no_grad()
    def __call__(self, frames_bgr):
        imgs = [to_pil(f) for f in frames_bgr]
        inp = self.proc(images=imgs, return_tensors="pt").to(self.dev)
        out = self.model(**inp)
        res = self.proc.post_process_object_detection(
            out, threshold=self.thr, target_sizes=[im.size[::-1] for im in imgs])
        batch = []
        for r in res:
            batch.append([[self.id2label[l.item()], round(s.item(), 3), *[round(x, 1) for x in b.tolist()]]
                          for s, l, b in zip(r["scores"], r["labels"], r["boxes"])])
        return batch


class OwlDetector:
    def __init__(self, queries=OWL_QUERIES, device=None, threshold=0.1):
        self.dev = device or pick_device()
        self.proc = Owlv2Processor.from_pretrained(OWL_ID)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_ID).to(self.dev).eval()
        self.queries = list(queries)
        self.thr = threshold

    @torch.no_grad()
    def __call__(self, frame_bgr):
        img = to_pil(frame_bgr)
        inp = self.proc(text=[self.queries], images=img, return_tensors="pt").to(self.dev)
        out = self.model(**inp)
        side = max(img.size)  # OWLv2 паддит до квадрата
        r = self.proc.post_process_grounded_object_detection(
            out, threshold=self.thr, target_sizes=torch.tensor([[side, side]]))[0]
        return [[self.queries[l.item()], round(s.item(), 3), *[round(x, 1) for x in b.tolist()]]
                for s, l, b in zip(r["scores"], r["labels"], r["boxes"])]
