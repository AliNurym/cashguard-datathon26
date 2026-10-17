"""Дорожки событий для интерфейса: на каких фактах сцены построен вывод.

Не входит в ответ API (схема ТЗ); отдаётся только веб-интерфейсу.
"""
import numpy as np

from .rules import P, entries, entry_time, stays_after_entry
from .scene import BAG_Q, MONEY_Q, Scene
from .vlm_plan import cashier_entry, empty_runs


def _segments(times, mask, merge_gap=0.0, min_len=0.0):
    """Отрезки [t0, t1], где mask истинна; соседние ближе merge_gap сливаются."""
    segs, start = [], None
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.5
    for i, m in enumerate(mask):
        if m and start is None:
            start = float(times[i])
        if start is not None and (not m or i == len(mask) - 1):
            end = float(times[i]) if m else float(times[i - 1]) + step
            if segs and start - segs[-1][1] <= merge_gap:
                segs[-1][1] = end
            else:
                segs.append([start, end])
            start = None
    return [[round(a, 1), round(b, 1)] for a, b in segs if b - a >= min_len]


def build_timeline(sc: Scene) -> dict:
    t = sc.times
    # люди в помещении: отрезки постоянного числа
    people, start = [], 0
    n = sc.n_people.astype(int)
    for i in range(1, len(n) + 1):
        if i == len(n) or n[i] != n[start]:
            people.append([round(float(t[start]), 1), round(float(t[i]) if i < len(n) else float(sc.dur), 1),
                           int(n[start])])
            start = i

    safe = None
    if sc.safe_prob:
        st = np.array([x[0] for x in sc.safe_prob])
        sp = np.array([x[1] for x in sc.safe_prob])
        sp = np.array([np.median(sp[max(0, i - 1):i + 2]) for i in range(len(sp))])
        safe = _segments(st, sp >= 0.5, merge_gap=1.0, min_len=0.5)

    desk = []
    if len(sc.owl_times):
        money = sc.obj_on_desk(MONEY_Q) >= P["money_thr"]
        bag = sc.obj_on_desk(BAG_Q) >= P["bag_thr"]
        desk = _segments(sc.owl_times, money | bag, merge_gap=1.5, min_len=0.5)

    events = []
    ce = cashier_entry(sc)
    if ce:
        events.append({"t": round(ce[0], 1), "kind": "in", "label": "кассир входит"})
        events.append({"t": round(ce[1], 1), "kind": "sit", "label": "кассир села"})
    for tc in entries(sc, base=1, hold=1.0):
        if stays_after_entry(sc, tc):
            events.append({"t": round(entry_time(sc, tc), 1), "kind": "in", "label": "входит посетитель"})
    for t0, t1 in empty_runs(sc):
        if t1 - t0 >= 1.0 or t1 >= sc.times[-1] - 1e-6:  # короткие провалы трека не показываем
            events.append({"t": round(t0, 1), "kind": "out", "label": "помещение опустело"})
    events.sort(key=lambda e: e["t"])
    return {"dur": round(float(sc.dur), 1), "people": people, "safe_open": safe, "desk_items": desk,
            "events": events}
