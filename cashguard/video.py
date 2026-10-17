"""Чтение видео с заданной частотой кадров."""
from pathlib import Path

import cv2


def video_info(path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": n, "w": w, "h": h, "dur": n / fps if fps else 0.0}


def iter_frames(path, sample_fps: float = 5.0):
    """Последовательное чтение (быстрее seek'ов). Отдаёт (t_sec, frame_bgr)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / sample_fps))
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            yield i / fps, frame
        i += 1
    cap.release()


def read_frame(path, t: float):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def video_path(name: str) -> Path:
    from .labels import ROOT, VARIANT, video_files
    if VARIANT:
        return ROOT / "data" / "aug" / VARIANT / f"{name}.mp4"
    return video_files()[name]
