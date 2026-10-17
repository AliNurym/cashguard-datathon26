"""Предразметка «деньги на столе» через grounding локальной VLM (кропы стола раз в секунду).

    python scripts/prelabel_money.py [--fps 1]
Результат: data/money_crops/<vid>_<t>.jpg и data/money_crops/prelabels.json
({"<vid>_<t>": [[x1,y1,x2,y2] в долях 0..1 кропа, ...]}). Позитивы потом проверяются вручную.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cashguard.labels import ROOT, load_videos  # noqa: E402
from cashguard.scene import build_scene  # noqa: E402
from cashguard.video import iter_frames  # noqa: E402
from cashguard.vlm import QUESTIONS, VLM, crop_region, parse_boxes, region_for  # noqa: E402

OUT = ROOT / "data" / "money_crops"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "prelabels.json"
    labels = json.loads(path.read_text()) if path.exists() else {}
    vlm = VLM()
    for v in load_videos():
        sc = build_scene(v.vid)
        if not sc.desk:
            continue
        t0, n = time.time(), 0
        for t, f in iter_frames(v.path, a.fps):
            key = f"{v.vid}_{t:05.1f}"
            if key in labels:
                continue
            crop = crop_region(f, **region_for("desk", sc.desk, sc.safe))
            cv2.imwrite(str(OUT / f"{key}.jpg"), crop)
            raw = vlm.ask([crop], QUESTIONS["desk"], max_new_tokens=120)
            labels[key] = [[c / 1000 for c in b] for b in parse_boxes(raw)]
            n += 1
        path.write_text(json.dumps(labels))
        pos = sum(1 for k, b in labels.items() if k.startswith(v.vid + "_") and b)
        print(f"{v.vid}: {n} кадров, с деньгами {pos}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
