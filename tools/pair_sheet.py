"""Сравнение роликов строками: каждый ролик — ряд кадров с таймкодами (кроп по вертикали).

    python tools/pair_sheet.py OUT.jpg X-28:0:16 X-29:0:17 [--every 1] [--width 150] [--y0 150 --y1 620]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cashguard.video import read_frame, video_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("specs", nargs="+", help="VID:start:end")
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=150)
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--y1", type=int, default=848)
    ap.add_argument("--x0", type=int, default=0)
    ap.add_argument("--x1", type=int, default=464)
    ap.add_argument("--cols", type=int, default=12)
    a = ap.parse_args()
    rows = []
    for spec in a.specs:
        vid, s, e = spec.split(":")
        t, tiles = float(s), []
        while t <= float(e):
            f = read_frame(video_path(vid), t)
            if f is None:
                break
            f = f[a.y0:a.y1, a.x0:a.x1]
            h = int(f.shape[0] * a.width / f.shape[1])
            f = cv2.resize(f, (a.width, h))
            cv2.rectangle(f, (0, 0), (a.width, 16), (0, 0, 0), -1)
            cv2.putText(f, f"{vid} {t:.1f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            tiles.append(f)
            t += a.every
        for i in range(0, len(tiles), a.cols):
            part = tiles[i:i + a.cols]
            part += [np.zeros_like(tiles[0])] * (a.cols - len(part))
            rows.append(np.hstack(part))
        rows.append(np.full((6, rows[-1].shape[1], 3), 255, np.uint8))
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
