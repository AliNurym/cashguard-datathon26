"""Активность в зоне счётной машины: купюры прошли через машину или нет.

По одному кадру VLM это не различает, поэтому смотрим движение пикселей в рамке машины:
когда кассир закладывает пачку и машина её пересчитывает, в зоне лотка много изменений.
"""
import cv2
import numpy as np

from .video import iter_frames


def box_from_norm(b, w, h):
    """Рамка Qwen-VL (0..1000 по входному изображению) -> пиксели кадра."""
    return [b[0] * w / 1000, b[1] * h / 1000, b[2] * w / 1000, b[3] * h / 1000]


def roi_activity(path, box, fps=5.0, pad=0.15):
    """[(t, mean|Δ|)] — средняя абсолютная разница соседних кадров в рамке машины (серое, 0..255)."""
    out, prev = [], None
    for t, f in iter_frames(path, fps):
        h, w = f.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        x1, x2 = int(max(0, x1 - pad * bw)), int(min(w, x2 + pad * bw))
        y1, y2 = int(max(0, y1 - pad * bh)), int(min(h, y2 + pad * bh))
        g = cv2.GaussianBlur(cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        if prev is not None and prev.shape == g.shape:
            out.append((round(t, 2), round(float(np.abs(g - prev).mean()), 2)))
        prev = g
    return out
