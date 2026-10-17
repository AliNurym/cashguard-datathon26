"""Отладочная раскадровка сцены: стол, дверь, люди в помещении (+ID трека), предметы на столе.

    python scripts/debug_scene.py X-20 [--every 1] [--start 0] [--end 999] [--cols 8]
Результат: data/debug/<vid>.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cashguard.labels import ROOT  # noqa: E402
from cashguard.scene import (BAG_Q, CUP_Q, MONEY_Q, PERSONAL_Q, build_scene, center,  # noqa: E402
                             inside)
from cashguard.video import read_frame, video_path  # noqa: E402

COL = {"money": (0, 220, 0), "bag": (0, 140, 255), "personal": (255, 0, 255), "cup": (255, 255, 0)}


def draw(fr, sc, t, thr=0.2):
    def rect(b, c, th=2, label=None):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(fr, (x1, y1), (x2, y2), c, th)
        if label:
            cv2.putText(fr, label, (x1 + 2, max(12, y1 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

    if sc.desk:
        rect(sc.desk, (255, 255, 255), 1, "desk")
    if sc.door:
        rect(sc.door, (200, 200, 200), 1, "door")
    i = int(np.argmin(np.abs(sc.times - t)))
    for tr in sc.tracks:
        if tr.start - 0.01 <= t <= tr.end + 0.01:
            j = int(np.argmin(np.abs(np.array(tr.t) - t)))
            if abs(tr.t[j] - t) < 0.3:
                rect(tr.boxes[j], (0, 0, 255), 2, f"#{tr.tid}")
    k = int(np.argmin(np.abs(sc.owl_times - t)))
    for q, s, *b in sc.owl[k]:
        if s < thr or not sc.desk or not inside(center(b), sc.desk, 0.05):
            continue
        kind = ("money" if q in MONEY_Q else "bag" if q in BAG_Q else
                "personal" if q in PERSONAL_Q else "cup" if q in CUP_Q else None)
        if kind:
            rect(b, COL[kind], 1, f"{kind}{s:.2f}")
    cv2.rectangle(fr, (0, 0), (150, 24), (0, 0, 0), -1)
    cv2.putText(fr, f"{t:5.1f}s n={sc.n_people[i]:.0f}", (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vid")
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--width", type=int, default=232)
    a = ap.parse_args()
    sc = build_scene(a.vid)
    path = video_path(a.vid)
    tiles = []
    t = a.start
    while t < min(a.end, sc.dur):
        fr = read_frame(path, t)
        if fr is None:
            break
        fr = draw(fr, sc, t)
        h = int(fr.shape[0] * a.width / fr.shape[1])
        tiles.append(cv2.resize(fr, (a.width, h)))
        t += a.every
    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]
    out = ROOT / "data" / "debug"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{a.vid}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("tracks:", [(tr.tid, round(tr.start, 1), round(tr.end, 1)) for tr in sc.tracks])
    print("desk:", sc.desk, "door:", sc.door)


if __name__ == "__main__":
    main()
