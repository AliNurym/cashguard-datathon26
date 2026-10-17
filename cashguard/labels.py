"""Загрузка разметки датасета: нормализация имён, дедупликация по md5, таймкоды."""
import csv
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "open"
VIDEOS = DATA / "videos"
LABELS_XLSX = DATA / "open_labels.xlsx"
TYPES_CSV = ROOT / "labels" / "types.csv"

# Вариант набора: "" — оригинал; иначе синтетический ракурс из data/aug/<variant>/
VARIANT = os.environ.get("CASHGUARD_VARIANT", "")


def variant_dir(base: str) -> Path:
    """data/<base> для оригинала, data/<base>_<variant> для синтетических ракурсов."""
    return ROOT / "data" / (f"{base}_{VARIANT}" if VARIANT else base)


def norm_name(name: str) -> str:
    """'X-1' (латиница) и 'Х-1' (кириллица) -> 'X-1'."""
    return name.strip().replace("Х", "X").replace("х", "x")


def parse_ts(s) -> float | None:
    if s is None:
        return None
    nums = [int(v) for v in re.findall(r"\d+", str(s))]
    if "мин" in str(s):
        return nums[0] * 60 + (nums[1] if len(nums) > 1 else 0)
    return float(nums[0])


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Video:
    vid: str                      # каноническое имя, напр. 'X-4'
    path: Path
    aliases: list = field(default_factory=list)   # все имена из разметки с тем же файлом
    gt_times: list = field(default_factory=list)  # таймкоды нарушений, с
    violation: bool = False


def video_files() -> dict:
    return {norm_name(p.stem): p for p in VIDEOS.glob("*.mov")}


def load_videos() -> list[Video]:
    """Уникальные видео (по md5) с объединённой разметкой."""
    files = video_files()
    ws = openpyxl.load_workbook(LABELS_XLSX).active
    by_hash: dict[str, Video] = {}
    for name, ts, lab in list(ws.iter_rows(values_only=True))[1:]:
        if not name:
            continue
        n = norm_name(name)
        p = files[n]
        h = md5(p)
        v = by_hash.setdefault(h, Video(vid=n, path=p))
        if n not in v.aliases:
            v.aliases.append(n)
        if str(lab).strip() == "нарушение":
            v.violation = True
            t = parse_ts(ts)
            if t is not None and t not in v.gt_times:
                v.gt_times.append(t)
    vids = list(by_hash.values())
    for v in vids:
        v.gt_times.sort()
        if VARIANT:
            v.path = ROOT / "data" / "aug" / VARIANT / f"{v.vid}.mp4"
    return sorted(vids, key=lambda v: int(re.findall(r"\d+", v.vid)[0]))


def load_types() -> dict:
    """{(vid, t): type} — ручная разметка типов (labels/types.csv)."""
    out = {}
    if TYPES_CSV.exists():
        with open(TYPES_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[(norm_name(r["video"]), float(r["t"]))] = r["type"]
    return out
