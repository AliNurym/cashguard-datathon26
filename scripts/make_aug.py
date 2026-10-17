"""Синтетические «другие ракурсы» для проверки устойчивости (критерий 5).

    python scripts/make_aug.py persp_l   # поворот камеры влево (перспектива)
    python scripts/make_aug.py persp_r
    python scripts/make_aug.py flip      # камера с другой стороны (зеркало)
    python scripts/make_aug.py zoom      # камера ближе (кроп 80% и масштаб)
    python scripts/make_aug.py dark      # хуже свет: яркость/контраст/шум

Результат: data/aug/<aug>/<vid>.mp4 (для дубликатов — одно видео на md5).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cashguard.labels import ROOT, load_videos  # noqa: E402

AUG_DIR = ROOT / "data" / "aug"


def make_transform(aug, w, h):
    if aug in ("persp_l", "persp_r"):
        d = 0.12
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        if aug == "persp_l":
            dst = np.float32([[0, h * d], [w, 0], [w, h], [0, h * (1 - d)]])
        else:
            dst = np.float32([[0, 0], [w, h * d], [w, h * (1 - d)], [0, h]])
        M = cv2.getPerspectiveTransform(src, dst)
        return lambda f: cv2.warpPerspective(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if aug == "flip":
        return lambda f: cv2.flip(f, 1)
    if aug == "zoom":
        cx, cy, s = w / 2, h * 0.55, 0.8
        x0, y0 = int(cx - w * s / 2), int(max(0, cy - h * s / 2))
        return lambda f: cv2.resize(f[y0:y0 + int(h * s), x0:x0 + int(w * s)], (w, h))
    if aug == "dark":
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 6, (h, w, 3)).astype(np.float32)
        return lambda f: np.clip(f.astype(np.float32) * 0.6 + 10 + noise, 0, 255).astype(np.uint8)
    raise SystemExit(f"unknown aug {aug}")


def main():
    aug = sys.argv[1]
    out_dir = AUG_DIR / aug
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in load_videos():
        out = out_dir / f"{v.vid}.mp4"
        if out.exists():
            continue
        cap = cv2.VideoCapture(str(v.path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tf = make_transform(aug, w, h)
        wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        while True:
            ok, f = cap.read()
            if not ok:
                break
            wr.write(tf(f))
        wr.release()
        cap.release()
        print(f"{aug}/{v.vid}", flush=True)


if __name__ == "__main__":
    main()
