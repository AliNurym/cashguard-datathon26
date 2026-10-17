"""Прогон детекторов по видео и сохранение сырых признаков в кэш.

    python scripts/extract_features.py [--fps 5] [--owl-fps 2] [--only X-1,X-2] [--force]

Кэш: data/cache/<vid>.json — детекции RT-DETR (fps), OWLv2 (owl-fps), яркость кадра.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cashguard.detectors import CocoDetector, OwlDetector  # noqa: E402
from cashguard.labels import ROOT, load_videos, variant_dir  # noqa: E402
from cashguard.video import iter_frames, video_info  # noqa: E402

CACHE = variant_dir("cache")


def extract(path, coco, owl, fps, owl_fps, batch=8):
    info = video_info(path)
    frames, owl_rows, buf = [], [], []
    owl_every = max(1, round(fps / owl_fps))

    def flush():
        dets = coco([f for _, f in buf])
        for (t, f), d in zip(buf, dets):
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            frames.append({"t": round(t, 3), "bright": round(float(gray.mean()), 1), "det": d})
        buf.clear()

    for k, (t, f) in enumerate(iter_frames(path, fps)):
        buf.append((t, f))
        if k % owl_every == 0:
            owl_rows.append({"t": round(t, 3), "det": owl(f)})
        if len(buf) == batch:
            flush()
    if buf:
        flush()
    return {"info": info, "fps": fps, "owl_fps": owl_fps, "frames": frames, "owl": owl_rows,
            "owl_queries": owl.queries}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--owl-fps", type=float, default=2.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    vids = [v for v in load_videos() if not only or v.vid in only]
    coco, owl = CocoDetector(), OwlDetector()
    for v in vids:
        out = CACHE / f"{v.vid}.json"
        if out.exists() and not a.force:
            continue
        t0 = time.time()
        feats = extract(v.path, coco, owl, a.fps, a.owl_fps)
        out.write_text(json.dumps(feats), encoding="utf-8")
        print(f"{v.vid}: {feats['info']['dur']:.1f}s video -> {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
