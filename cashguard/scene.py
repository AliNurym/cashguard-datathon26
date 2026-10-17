"""Понимание сцены по кэшу детекций: стол, дверь, люди в помещении, треки, предметы на столе."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .labels import ROOT, variant_dir

CACHE = variant_dir("cache")

MONEY_Q = ("banknotes", "a stack of paper money")
BAG_Q = ("a green cloth bag",)
PERSONAL_Q = ("a handbag", "a coat")
CUP_Q = ("a coffee cup",)


def load_cache(vid: str) -> dict:
    return json.loads((CACHE / f"{vid}.json").read_text(encoding="utf-8"))


def _median_box(boxes):
    return [float(np.median([b[i] for b in boxes])) for i in range(4)] if boxes else None


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def inside(pt, box, pad=0.0):
    x, y = pt
    w, h = box[2] - box[0], box[3] - box[1]
    return box[0] - pad * w <= x <= box[2] + pad * w and box[1] - pad * h <= y <= box[3] + pad * h


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    tid: int
    t: list = field(default_factory=list)
    boxes: list = field(default_factory=list)
    idx: list = field(default_factory=list)   # индексы кадров

    @property
    def start(self):
        return self.t[0]

    @property
    def end(self):
        return self.t[-1]


@dataclass
class Scene:
    vid: str
    W: int
    H: int
    dur: float
    times: np.ndarray            # времена кадров RT-DETR
    desk: list | None
    door: list | None
    persons: list                # [кадр] -> список боксов людей в помещении
    n_people: np.ndarray         # сглаженное число людей в помещении
    tracks: list                 # list[Track]
    owl_times: np.ndarray
    owl: list                    # [кадр OWL] -> детекции
    bright: np.ndarray
    safe: list | None = None
    roles: dict = field(default_factory=dict)  # tid -> 'cashier' | 'visitor'
    safe_prob: list | None = None  # [(t, p)] «сейф открыт» от SafeClassifier (необязательно)
    money_fn: object = None        # callable(times) -> {t: p} «деньги на столе» от MoneyClassifier (необязательно)
    laptop: list | None = None     # рамка ноутбука кассира
    frames_det: list | None = None # сырые детекции RT-DETR по кадрам (для правил по предметам в руках)

    def box_at(self, tr: Track, t: float, tol=0.3):
        j = int(np.argmin(np.abs(np.array(tr.t) - t)))
        return tr.boxes[j] if abs(tr.t[j] - t) <= tol else None

    def cashier_tracks(self):
        return [tr for tr in self.tracks if self.roles.get(tr.tid) == "cashier"]

    def visitor_tracks(self):
        return [tr for tr in self.tracks if self.roles.get(tr.tid) == "visitor"]

    def obj_on_desk(self, queries, pad=0.05) -> np.ndarray:
        """Макс. скор OWL-объекта из queries с центром на столе, по кадрам OWL."""
        out = np.zeros(len(self.owl))
        if self.desk is None:
            return out
        for i, dets in enumerate(self.owl):
            s = [d[1] for d in dets if d[0] in queries and inside(center(d[2:]), self.desk, pad)]
            out[i] = max(s) if s else 0.0
        return out

    def at(self, series_t, series, t):
        """Значение ряда в момент t (ближайший кадр)."""
        if len(series_t) == 0:
            return 0.0
        return float(series[int(np.argmin(np.abs(series_t - t)))])


def estimate_desk(cache, W, H):
    """Стол кассира: OWL 'an office desk', не касающийся низа кадра (там передний белый стол)."""
    cands = []
    for row in cache["owl"]:
        best = None
        for q, s, *b in row["det"]:
            if q == "an office desk" and s >= 0.25 and b[3] < 0.8 * H and (b[2] - b[0]) > 0.3 * W:
                if best is None or s > best[0]:
                    best = (s, b)
        if best:
            cands.append(best[1])
    if not cands:  # fallback: RT-DETR diningtable
        for fr in cache["frames"]:
            for l, s, *b in fr["det"]:
                if l == "diningtable" and s >= 0.4 and b[3] < 0.8 * H:
                    cands.append(b)
    return _median_box(cands)


def estimate_safe(cache):
    """Сейф: кластер OWL 'a grey metal cabinet' с лучшим средним топ-5 скором.
    Сумма скоров не годится: пока сейф открыт, OWL его теряет, а тумба видна всё время.
    Рамки у левого/правого края кадра (тумбы, обрезанная мебель) штрафуются."""
    W = cache["info"]["w"]
    dets = [(s, b) for row in cache["owl"] for q, s, *b in row["det"]
            if q in ("a grey metal cabinet", "a metal safe") and s >= 0.2]
    clusters: list[list] = []
    for s, b in sorted(dets, key=lambda x: -x[0]):
        for cl in clusters:
            if iou(cl[0][1], b) > 0.5:
                cl.append((s, b))
                break
        else:
            clusters.append([(s, b)])
    if not clusters:
        return None

    def score(cl):
        top = sorted((s for s, _ in cl), reverse=True)[:5]
        b = _median_box([b for _, b in cl])
        edge = b[0] < 0.02 * W or b[2] > 0.98 * W
        return np.mean(top) * (0.6 if edge else 1.0) * min(1.0, len(cl) / 3)

    best = max(clusters, key=score)
    return _median_box([b for _, b in best])


def assign_roles(tracks) -> dict:
    """Кассир — самый долгий трек; треки, не пересекающиеся с ним по времени и
    начинающиеся там, где он закончился, — продолжение кассира. Остальные — посетители."""
    if not tracks:
        return {}
    main = max(tracks, key=lambda tr: tr.end - tr.start)
    roles = {tr.tid: "visitor" for tr in tracks}
    roles[main.tid] = "cashier"
    return roles


def estimate_laptop(cache):
    """Ноутбук кассира: медиана рамок RT-DETR 'laptop' (уверенные), запасной вариант — OWL 'a laptop'."""
    bs = [b for f in cache["frames"] for l, s, *b in f["det"] if l == "laptop" and s >= 0.6]
    if len(bs) < 5:
        bs = [b for row in cache["owl"] for q, s, *b in row["det"] if q == "a laptop" and s >= 0.4]
    return _median_box(bs) if len(bs) >= 3 else None


def estimate_door(cache):
    boxes = [b for row in cache["owl"] for q, s, *b in row["det"] if q == "a glass door" and s >= 0.25]
    return _median_box(boxes)


def room_persons(frame_det, desk, H, min_conf=0.45, margin=0.05):
    """Люди в помещении: низ рамки не выше верхнего края стола (минус отступ).
    Отсекает людей за стеклянной стеной и отражения в стекле."""
    top = desk[1] if desk else 0.4 * H
    out = []
    for l, s, *b in frame_det:
        if l != "person" or s < min_conf:
            continue
        if b[3] < top - margin * H or (b[3] - b[1]) < 0.07 * H:
            continue
        out.append((s, b))
    # подавление дублей: по IoU и по вложенности — вторая рамка той же фигуры (например,
    # вытянутая до стула) почти целиком накрывает первую и давала ложного «вошедшего» (X-43 на GPU)
    out.sort(key=lambda x: -x[0])
    keep = []
    for s, b in out:
        if all(iou(b, k) < 0.5 and contained(b, k) < 0.8 for _, k in keep):
            keep.append((s, b))
    return keep


def contained(a, b):
    """Доля меньшей из рамок, лежащая внутри другой."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small > 0 else 0.0


def smooth_count(counts, k=5):
    """Медианный фильтр по окну k кадров."""
    c = np.asarray(counts, dtype=float)
    if len(c) < k:
        return c
    pad = k // 2
    cp = np.pad(c, pad, mode="edge")
    return np.array([np.median(cp[i:i + k]) for i in range(len(c))])


def build_tracks(times, persons, H, max_lost=1.0, min_len=3):
    """Жадный трекер: сопоставление по IoU и нормированному расстоянию центров.
    Людей в кадре мало (<=4), поэтому этого достаточно."""
    active: list[Track] = []
    done: list[Track] = []
    nid = 0
    for k, (t, ps) in enumerate(zip(times, persons)):
        boxes = [b for _, b in ps]
        pairs = []
        for i, tr in enumerate(active):
            lb = tr.boxes[-1]
            gap = t - tr.end
            for j, b in enumerate(boxes):
                d = np.hypot(*(np.subtract(center(lb), center(b)))) / H
                ov = iou(lb, b)
                # после пропуска кадров — только при заметном перекрытии
                if ov > 0.3 or (gap < 0.5 and (ov > 0.1 or d < 0.08)):
                    pairs.append((ov - d, i, j))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            active[i].t.append(float(t))
            active[i].boxes.append(list(boxes[j]))
            active[i].idx.append(k)
        for j, b in enumerate(boxes):
            if j not in used_d:
                nid += 1
                active.append(Track(nid, [float(t)], [list(b)], [k]))
        still = []
        for tr in active:
            (still if t - tr.end <= max_lost else done).append(tr)
        active = still
    done += active
    return sorted([tr for tr in done if len(tr.t) >= min_len], key=lambda tr: tr.start)


def real_track(tr: Track, desk, door, H) -> bool:
    """Трек живого человека в помещении, а не отражения в стекле / человека за стеклом.
    Отражения и люди за стеклом мельче (рост < 0.2 H) и не опускаются ниже края стола."""
    top = desk[1] if desk else 0.4 * H

    def suspect(b):  # мелкая фигура, не доходящая до края стола
        return b[3] < top + 0.01 * H and (b[3] - b[1]) < 0.2 * H

    ok = sum(not suspect(b) for b in tr.boxes)
    return ok >= 0.5 * len(tr.boxes)


def is_static_late(tr: Track, H) -> bool:
    """Появился посреди ролика и почти не двигается (кандидат в отражение)."""
    if tr.start <= 1.0:
        return False
    c0 = np.array(center(tr.boxes[0]))
    disp = max(np.hypot(*(np.array(center(b)) - c0)) for b in tr.boxes) / H
    max_h = max(b[3] - b[1] for b in tr.boxes) / H
    return disp < 0.1 and max_h < 0.25


def has_twin(tr: Track, others, W, H) -> bool:
    """Отражение в стекле дублирует реального человека: похожий рост и уровень низа рамки,
    рядом по горизонтали, в те же моменты времени (>= 60% кадров трека)."""
    hits = 0
    for k, b in zip(tr.idx, tr.boxes):
        h = b[3] - b[1]
        for o in others:
            if k not in o.idx:
                continue
            ob = o.boxes[o.idx.index(k)]
            oh = ob[3] - ob[1]
            if (abs(ob[3] - b[3]) < 0.05 * H and 0.55 < oh / max(h, 1) < 1.8
                    and abs(center(ob)[0] - center(b)[0]) < 0.4 * W):
                hits += 1
                break
    return hits >= 0.5 * len(tr.boxes)


def build_scene(vid: str) -> Scene:
    c = load_cache(vid)
    W, H, dur = c["info"]["w"], c["info"]["h"], c["info"]["dur"]
    desk = estimate_desk(c, W, H)
    door = estimate_door(c)
    times = np.array([f["t"] for f in c["frames"]])
    cand = [room_persons(f["det"], desk, H) for f in c["frames"]]
    tracks = [tr for tr in build_tracks(times, cand, H) if real_track(tr, desk, door, H)]
    # отражение: неподвижный поздний трек внутри стеклянной панели (низ рамки выше
    # нижней кромки стекла), дублирующий реального человека
    def in_glass(tr):
        return door is not None and np.median([b[3] for b in tr.boxes]) < door[3] - 0.01 * H

    static = [tr for tr in tracks if is_static_late(tr, H) and in_glass(tr)]
    moving = [tr for tr in tracks if tr not in static]
    tracks = moving + [tr for tr in static if not has_twin(tr, moving, W, H)]
    tracks.sort(key=lambda tr: tr.start)
    persons = [[] for _ in times]
    for tr in tracks:
        for k, b in zip(tr.idx, tr.boxes):
            persons[k].append((1.0, b))
    n_people = smooth_count([len(p) for p in persons], k=5)
    owl_times = np.array([r["t"] for r in c["owl"]])
    owl = [r["det"] for r in c["owl"]]
    bright = np.array([f["bright"] for f in c["frames"]])
    return Scene(vid, W, H, dur, times, desk, door, persons, n_people, tracks, owl_times, owl, bright,
                 safe=estimate_safe(c), roles=assign_roles(tracks), laptop=estimate_laptop(c),
                 frames_det=[f["det"] for f in c["frames"]])
