"""Где спрашивать VLM: моменты событий сцены (без загрузки модели — удобно для тестов)."""
import numpy as np

from .scene import Scene


def empty_runs(sc: Scene, min_sec=0.3):
    out, start = [], None
    for i, n in enumerate(sc.n_people):
        if n == 0 and start is None:
            start = sc.times[i]
        if (n != 0 or i == len(sc.n_people) - 1) and start is not None:
            end = sc.times[i] if n == 0 else sc.times[i - 1]
            if end - start >= min_sec and start > 0.5:
                out.append((float(start), float(end)))
            start = None
    return out


def cashier_entry(sc: Scene):
    """Кассир входит в пустое помещение (начало смены). Возвращает (t_появления, t_посадки) или None.
    t_посадки — первый момент, когда кассир сидит >= 1 с (рост рамки < 0.15 H): к этому времени
    дверь уже должна быть закрыта, по нему и спрашиваем VLM."""
    n, t = sc.n_people, sc.times
    if len(n) == 0 or n[0] != 0:
        return None
    nz = np.nonzero(n > 0)[0]
    if len(nz) == 0 or float(t[nz[0]]) <= 0.5:
        return None
    t_in = float(t[nz[0]])
    ct = sc.cashier_tracks()
    if not ct:
        return None
    tr = ct[0]
    seated = [(ti, (b[3] - b[1]) < 0.15 * sc.H) for ti, b in zip(tr.t, tr.boxes) if ti > t_in]
    run0 = None
    for ti, s in seated:
        if s:
            run0 = ti if run0 is None else run0
            if ti - run0 >= 1.0:
                return t_in, float(run0)
        else:
            run0 = None
    return None


def visits(sc: Scene):
    """Интервалы, когда в помещении кроме кассира есть посетитель (n >= 2)."""
    out, start = [], None
    for i, n in enumerate(sc.n_people):
        if n >= 2 and start is None:
            start = sc.times[i]
        if (n < 2 or i == len(sc.n_people) - 1) and start is not None:
            end = sc.times[i]
            if end - start >= 2.0:
                out.append((float(start), float(end)))
            start = None
    return out


def snap(t):
    return round(float(t) * 2) / 2  # шаг 0.5 с — ключи кэша стабильны


def plan_queries(sc: Scene, safe_every=3.0, count_every=2.0, cache=None, refine=False):
    """Список (qid, t). refine=True — дополнительные моменты по уже полученным ответам."""
    q = []
    if not refine:
        q.append(("machine", 0.5))
        for t in np.arange(0.5, sc.dur - 0.2, safe_every):
            q.append(("safe", snap(t)))
        for t0, t1 in empty_runs(sc):
            q.append(("laptop", snap(t0 + 0.5)))
            q.append(("laptop", snap(min(t1, t0 + 1.5))))
            q.append(("desk", snap(t0 + 0.5)))
            q.append(("desk", snap(min(t1, t0 + 1.5))))
            q.append(("safe", snap(t0 + 0.5)))
        ce = cashier_entry(sc)
        if ce is not None:
            for dt in (1.0, 3.0, 5.0):
                if ce[1] + dt < sc.dur - 0.2:
                    q.append(("door", snap(ce[1] + dt)))
        from .rules import entries, entry_time  # локальный импорт: избегаем цикла
        for tc in entries(sc, base=1, hold=1.0):
            te = entry_time(sc, tc)
            q.append(("desk", snap(te)))
            q.append(("desk", snap(te + 1.0)))
        for t0, t1 in visits(sc):
            for t in np.arange(t0, t1, count_every):
                q.append(("count", snap(t)))
                q.append(("id", snap(t)))
    else:
        # сейф: найти первую выборку «открыт и никого рядом» и уточнить начало с шагом 0.5 с
        ser = sorted((float(k.split("@")[1]), v) for k, v in cache.data.items() if k.startswith("safe@"))
        prev_t = None
        for t, v in ser:
            if v.get("safe_open") and not v.get("person_at_safe"):
                if prev_t is not None:
                    for tt in np.arange(prev_t + 0.5, t, 0.5):
                        q.append(("safe", snap(tt)))
                break
            prev_t = t
    seen, out = set(), []
    for item in q:
        if item not in seen and 0 <= item[1] < sc.dur:
            seen.add(item)
            out.append(item)
    return out
