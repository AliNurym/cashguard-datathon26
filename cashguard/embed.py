"""Признаки кропов из замороженного DINOv2 (Apache-2.0) для лёгких классификаторов состояний.

Почему не дообучение всей сети: размеченных кадров сотни, а финальное видео снято под
другим углом. Замороженный backbone + логистическая регрессия переобучаются меньше.
"""
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from .detectors import pick_device

DINO_ID = "facebook/dinov2-base"


class Embedder:
    def __init__(self, model_id=DINO_ID, device=None, pooling="grid", no_crop=False):
        """no_crop=True: ресайз в 224x224 без центральной обрезки (мелкие объекты у краёв не теряются)."""
        self.dev = device or pick_device()
        kw = {"do_center_crop": False, "size": {"height": 224, "width": 224}} if no_crop else {}
        self.proc = AutoImageProcessor.from_pretrained(model_id, **kw)
        self.model = AutoModel.from_pretrained(model_id).to(self.dev).eval()
        self.pooling = pooling

    @torch.no_grad()
    def __call__(self, crops_bgr, batch=32) -> np.ndarray:
        """pooling='cls_mean': [N, 2D] — CLS и среднее по патчам.
        pooling='grid': [N, 5D] — CLS и средние по 4 квадрантам (локальные изменения не размываются)."""
        out = []
        for i in range(0, len(crops_bgr), batch):
            imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr[i:i + batch]]
            inp = self.proc(images=imgs, return_tensors="pt").to(self.dev)
            h = self.model(**inp).last_hidden_state
            cls, patches = h[:, 0], h[:, 1:]
            if self.pooling == "grid":
                n = int(patches.shape[1] ** 0.5)
                g = patches.reshape(len(imgs), n, n, -1).permute(0, 3, 1, 2)
                q = torch.nn.functional.adaptive_avg_pool2d(g, 2).flatten(2).permute(0, 2, 1)  # [B,4,D]
                parts = [cls] + [q[:, k] for k in range(4)]
            else:
                parts = [cls, patches.mean(1)]
            f = torch.cat([torch.nn.functional.normalize(p, dim=1) for p in parts], dim=1)
            out.append(f.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2 * self.model.config.hidden_size))


def augment(crop, rng):
    """Аугментации под «другой ракурс»: перспектива, масштаб, отражение, свет."""
    h, w = crop.shape[:2]
    d = rng.uniform(0, 0.15)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = rng.uniform(-d, d, (4, 2)) * [w, h]
    M = cv2.getPerspectiveTransform(src, (src + jit).astype(np.float32))
    out = cv2.warpPerspective(crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    s = rng.uniform(0.8, 1.0)
    ch, cw = int(h * s), int(w * s)
    y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    out = cv2.resize(out[y0:y0 + ch, x0:x0 + cw], (w, h))
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    a, b = rng.uniform(0.6, 1.3), rng.uniform(-25, 25)
    return np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
