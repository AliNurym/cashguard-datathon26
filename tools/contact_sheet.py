"""Раскадровка видео: метаданные + сетка кадров с таймкодами.

Использование:
    python tools/contact_sheet.py VIDEO OUT.jpg [--every 2.0] [--start 0] [--end None] [--cols 6] [--width 320]
    python tools/contact_sheet.py --probe DIR   # метаданные всех видео в папке
"""
import argparse
import os
import sys

import cv2
import numpy as np


def probe(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": n, "w": w, "h": h, "dur": n / fps if fps else 0}


def sheet(path, out, every=2.0, start=0.0, end=None, cols=6, width=320, crop=None):
    info = probe(path)
    end = min(end or info["dur"], info["dur"])
    cap = cv2.VideoCapture(path)
    tiles = []
    t = start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        if crop:
            x0, y0, x1, y1 = crop
            frame = frame[y0:y1, x0:x1]
        h = int(frame.shape[0] * width / frame.shape[1])
        frame = cv2.resize(frame, (width, h))
        label = f"{t:6.1f}s"
        cv2.rectangle(frame, (0, 0), (92, 24), (0, 0, 0), -1)
        cv2.putText(frame, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(frame)
        t += every
    cap.release()
    if not tiles:
        print("no frames", path, file=sys.stderr)
        return
    th, tw = tiles[0].shape[:2]
    while len(tiles) % cols:
        tiles.append(np.zeros((th, tw, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--probe")
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--crop", type=lambda s: tuple(int(v) for v in s.split(",")), help="x0,y0,x1,y1")
    a = ap.parse_args()
    if a.probe:
        for root, _, files in os.walk(a.probe):
            for f in sorted(files):
                if f.lower().endswith((".mov", ".mp4")):
                    p = os.path.join(root, f)
                    i = probe(p)
                    print(f"{os.path.relpath(p, a.probe)}\t{i['w']}x{i['h']}\t{i['fps']:.2f}fps\t{i['dur']:.1f}s")
    else:
        sheet(a.video, a.out, a.every, a.start, a.end, a.cols, a.width, a.crop)
