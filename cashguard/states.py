"""Обученные классификаторы состояний сцены (DINOv2 + логистическая регрессия).

SafeClassifier: вероятность «дверца сейфа открыта» по кропу сейфа, ряд [(t, p)] по видео.
Люди на кадре закрашиваются серым (как при обучении), чтобы модель смотрела на дверцу,
а не на человека у сейфа. Обучение и оценка: scripts/train_safe.py, модель models/safe_clf.joblib.
"""
from pathlib import Path

import joblib
import numpy as np

from .embed import Embedder
from .video import iter_frames
from .vlm import crop_region

MODELS = Path(__file__).resolve().parent.parent / "models"


def mask_persons(frame, sc, t):
    k = int(np.argmin(np.abs(sc.times - t)))
    out = frame.copy()
    for _, b in sc.persons[k]:
        x1, y1, x2, y2 = [int(v) for v in b]
        out[max(0, y1):y2, max(0, x1):x2] = 128
    return out


class SafeClassifier:
    def __init__(self, embedder: Embedder | None = None, path=MODELS / "safe_clf.joblib"):
        self.emb = embedder or Embedder()
        self.clf = joblib.load(path)

    def series(self, video_path, sc, fps=2.0):
        """[(t, p_open)] для сцены sc (нужны sc.safe и sc.persons)."""
        if sc.safe is None:
            return []
        ts, crops = [], []
        for t, f in iter_frames(video_path, fps):
            ts.append(round(t, 2))
            crops.append(crop_region(mask_persons(f, sc, t), sc.safe, 0.6, 0.3, 0.2, min_side=0))
        if not crops:
            return []
        p = self.clf.predict_proba(self.emb(crops))[:, 1]
        return list(zip(ts, [round(float(x), 3) for x in p]))


class MoneyClassifier:
    """Вероятность «на столе есть деньги»: макс. по плиткам 4x2 кропа стола (scripts/train_money.py)."""
    GX, GY = 4, 2

    def __init__(self, embedder: Embedder | None = None, path=MODELS / "money_clf.joblib"):
        self.emb = embedder or Embedder(no_crop=True)
        self.clf = joblib.load(path)

    def tiles(self, crop):
        h, w = crop.shape[:2]
        return [crop[j * h // self.GY:(j + 1) * h // self.GY, i * w // self.GX:(i + 1) * w // self.GX]
                for j in range(self.GY) for i in range(self.GX)]

    def at(self, video_path, sc, times):
        """{t: p} в заданные моменты (кадр читается по времени)."""
        from .video import read_frame
        from .vlm import region_for
        out = {}
        if sc.desk is None:
            return out
        for t in times:
            f = read_frame(video_path, t)
            if f is None:
                continue
            crop = crop_region(f, **region_for("desk", sc.desk, sc.safe))
            out[round(float(t), 2)] = float(self.clf.predict_proba(self.emb(self.tiles(crop)))[:, 1].max())
        return out
