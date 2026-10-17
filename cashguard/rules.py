"""Правила v0: события сцены + состояние стола -> кандидаты нарушений."""
from dataclasses import dataclass, field

import numpy as np

from .scene import BAG_Q, MONEY_Q, PERSONAL_Q, Scene, center, inside

# Пороги (подбираются на обучающих роликах)
P = {
    "empty_min_sec": 1.5,     # помещение пустое не меньше, с
    "money_thr": 0.30,        # OWL: деньги на столе
    "money_clf_thr": 0.65,
    "visit_min_sec": 6.0,     # C: вошедший должен пробыть в помещении не меньше, с    # классификатор плиток: деньги на столе (нормальные ролики <= 0.43 на всех ракурсах)
    "bag_thr": 0.22,          # OWL: зелёная сумка на столе
    "personal_thr": 0.25,     # OWL: дамская сумка / верхняя одежда на столе
    "personal_min_sec": 2.0,
    "crowd_n": 3,             # людей в помещении одновременно
    "crowd_min_sec": 1.0,
    "bag_leave_sec": 6.0,     # инкассатор ушёл раньше, чем через N с после сумки
    "still_speed": 0.035,     # доля высоты кадра в секунду: «стоит/сидит»
    "dep_still_v": 0.02,      # уход: «неподвижно у сейфа» — скорость не выше (доля H/с)
    "dep_still_sec": 1.2,     # уход: длительность неподвижности у сейфа
    "dep_near": 0.15,         # уход: зазор до сейфа по горизонтали (доля W)
    "side_depth": 0.06,       # |низ рамки посетителя - низ рамки кассира| < доля H
    "side_min_sec": 1.5,
    "safe_lookback": 15.0,    # окно перед уходом, в котором ищем подход к сейфу, с
    "safe_thr": 0.75,         # B: средняя вероятность «открыт» перед уходом (консервативно: без FP)
    "safe_before": 1.5,       # B: окно перед уходом от сейфа, с
    "drink_frac": 0.5,        # G: доля кадров окна 2 с с поднятым стаканом (X-46: 37 детекций, прочие ролики <= 4 за весь ролик)
    "stand_h": 0.18,          # рост рамки (доля H): выше — стоит, ниже — сидит
    "safe_near": 0.1,         # зазор по горизонтали между кассиром и сейфом (доля W): «у сейфа»
    "vlm_safe_min_hits": 2,   # B: выборок подряд «сейф открыт, у сейфа никого»
    "vlm_desk_min_hits": 1,   # A/C: из 2 кадров у стола сколько должны показать деньги
    "vlm_hand_min_hits": 2,   # D: кадров с пересчётом руками за визит
    "vlm_money_min_hits": 2,  # I: кадров с деньгами за визит
}


@dataclass
class Violation:
    type: str
    t: float
    score: float
    evidence_t: list = field(default_factory=list)
    details: dict = field(default_factory=dict)


def runs(mask, times, min_sec):
    """Непрерывные участки mask=True длиной >= min_sec: [(t0, t1)]."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if m else i - 1
            if times[end] - times[start] >= min_sec:
                out.append((float(times[start]), float(times[end])))
            start = None
    return out


def money_clf_max(sc: Scene, t0, t1, step=0.5):
    """Макс. вероятность «деньги на столе» от обученного классификатора на [t0, t1] (0, если не подключён)."""
    if sc.money_fn is None:
        return 0.0
    ts = np.arange(max(0.0, t0), min(sc.dur - 0.05, t1) + 1e-6, step)
    p = sc.money_fn(ts)
    return max(p.values()) if p else 0.0


def series_max(sc: Scene, series, t0, t1):
    m = (sc.owl_times >= t0 - 1e-6) & (sc.owl_times <= t1 + 1e-6)
    return float(series[m].max()) if m.any() else 0.0


def departure_start(sc: Scene, t_exit: float) -> float:
    """Начало ухода с рабочего места (так размечен датасет): более поздний из моментов
    (а) встала со стула (рамка выросла выше stand_h), (б) закончила неподвижно стоять у сейфа.
    Медленная ходьба за стеклом к двери в кадре выглядит как «стоит», поэтому неподвижность
    считается только рядом с сейфом. Параметры подобраны по 6 уходам (средняя ошибка ~1.9 с)."""
    best = None
    for tr in sc.tracks:
        if abs(tr.end - t_exit) < 1.5 and (best is None or len(tr.t) > len(best.t)):
            best = tr
    if best is None or len(best.t) < 4:
        return max(0.0, t_exit - 2.5)
    t, vs = track_speed(sc, best)
    stand = np.array([(b[3] - b[1]) >= P["stand_h"] * sc.H for b in best.boxes])
    cands = [float(t[i - 1]) for i in range(1, len(t)) if not stand[i - 1] and stand[i] and t[i] < t_exit]
    if sc.safe:
        near = np.array([max(sc.safe[0] - b[2], b[0] - sc.safe[2], 0) <= P["dep_near"] * sc.W for b in best.boxes])
        cands += [b for a, b in runs((vs <= P["dep_still_v"]) & stand & near, t, P["dep_still_sec"]) if b < t_exit - 0.3]
    return max(cands) if cands else max(0.0, t_exit - 2.5)


def track_speed(sc: Scene, tr):
    """Скорость (доли H в секунду): движение центра или смена позы, сглаженная по 3 кадрам."""
    t = np.array(tr.t)
    c = np.array([center(b) for b in tr.boxes])
    top = np.array([b[1] for b in tr.boxes])
    v = np.zeros(len(t))
    for i in range(1, len(t)):
        dt = max(1e-3, t[i] - t[i - 1])
        v[i] = max(np.hypot(*(c[i] - c[i - 1])), abs(top[i] - top[i - 1])) / sc.H / dt
    return t, np.convolve(v, np.ones(3) / 3, mode="same")


def arrival_time(sc: Scene, tr) -> float:
    """Момент, когда вошедший дошёл и остановился (первая неподвижная фаза >= 0.6 с)."""
    t, vs = track_speed(sc, tr)
    still = [r for r in runs(vs < P["still_speed"], t, 0.6) if r[0] > tr.start + 0.5]
    return float(still[0][0]) if still else float(tr.start + 2.0)


def rule_unattended(sc: Scene, money, bag):
    """A: все ушли из помещения, на столе остались деньги/сумка."""
    out = []
    empty = runs(sc.n_people == 0, sc.times, P["empty_min_sec"])
    # короткая пустота в самом конце ролика — тоже уход
    tail = runs(sc.n_people == 0, sc.times, 0.3)
    if tail and tail[-1][1] >= sc.times[-1] - 1e-6 and tail[-1] not in empty:
        empty.append(tail[-1])
    for t0, t1 in empty:
        if t0 < 0.5:  # пусто с начала ролика — это не уход
            continue
        m = series_max(sc, money, t0, min(t1, t0 + 3))
        b = series_max(sc, bag, t0, min(t1, t0 + 3))
        mc = money_clf_max(sc, t0, min(t1, t0 + 3))
        if m >= P["money_thr"] or b >= P["bag_thr"] or mc >= P["money_clf_thr"]:
            td = departure_start(sc, t0)
            out.append(Violation("A_unattended_valuables", td, max(m, b), [td, t0 + 1.0],
                                 {"empty_from": t0, "money": m, "bag": b}))
    return out


def entries(sc: Scene, base=1, hold=1.0):
    """Моменты входа: число людей было == base не меньше hold с, затем выросло."""
    out = []
    n, t = sc.n_people, sc.times
    run_start = None
    for i in range(len(n)):
        if n[i] == base:
            if run_start is None:
                run_start = t[i]
        elif n[i] > base and run_start is not None:
            if t[i] - run_start >= hold:
                out.append(float(t[i]))
            run_start = None
        else:
            run_start = None
    return out


def entry_time(sc: Scene, t_count: float) -> float:
    """Уточнение: начало трека, появившегося рядом с моментом роста счётчика."""
    near = [tr.start for tr in sc.tracks if t_count - 1.5 <= tr.start <= t_count + 0.2]
    return min(near) if near else t_count


def visited_safe(sc: Scene, tr, t0, t1) -> bool:
    """Был ли человек у сейфа в интервале: рамка перекрывает >= 30% рамки сейфа."""
    if sc.safe is None:
        return True  # не знаем, где сейф — не обвиняем
    s = sc.safe
    area = (s[2] - s[0]) * (s[3] - s[1])
    for t, b in zip(tr.t, tr.boxes):
        if t0 <= t <= t1:
            ix = max(0, min(b[2], s[2]) - max(b[0], s[0]))
            iy = max(0, min(b[3], s[3]) - max(b[1], s[1]))
            if ix * iy >= 0.3 * area:
                return True
    return False


def rule_left_without_safe(sc: Scene):
    """A: кассир покинула помещение, не подойдя перед уходом к сейфу (не убрала/не заперла ценности)."""
    out = []
    empty = runs(sc.n_people == 0, sc.times, 0.3)
    for t0, t1 in empty:
        if t0 < 0.5:
            continue
        last = [tr for tr in sc.tracks if abs(tr.end - t0) < 1.5]
        if not last:
            continue
        tr = max(last, key=lambda x: len(x.t))
        if tr.start > t0 - 3.0:
            continue  # слишком короткий трек — не о чем судить
        if not visited_safe(sc, tr, t0 - P["safe_lookback"], t0):
            td = departure_start(sc, t0)
            out.append(Violation("A_unattended_valuables", td, 0.6, [td, t0 + 0.5],
                                 {"empty_from": t0, "kind": "no_safe_visit"}))
    return out


def stays_after_entry(sc: Scene, t_count: float) -> bool:
    """Вошедший остаётся в помещении: его трек длится >= visit_min_sec или до конца ролика.
    Короткий трек — это уход прежнего клиента, которого детектор не видел сидящим (X-38 на GPU)."""
    near = [tr for tr in sc.tracks if t_count - 1.5 <= tr.start <= t_count + 0.2]
    if not near:
        return True  # трека не нашли — судим по счётчику, как раньше
    tr = min(near, key=lambda x: x.start)
    return tr.end - tr.start >= P["visit_min_sec"] or tr.end >= sc.dur - 0.5


def rule_cash_next_client(sc: Scene, money):
    """C: входит следующий человек (и остаётся), а на столе открыто лежат деньги."""
    out = []
    for tc in entries(sc, base=1, hold=1.0):
        if not stays_after_entry(sc, tc):
            continue
        te = entry_time(sc, tc)
        m = series_max(sc, money, te - 1.0, te + 1.5)
        mc = money_clf_max(sc, te - 1.0, te + 1.5)
        if m >= P["money_thr"] or mc >= P["money_clf_thr"]:
            out.append(Violation("C_cash_on_desk_next_client", te, max(m, mc), [te, te + 1.5],
                                 {"money_owl": m, "money_clf": mc}))
    return out


def rule_crowd(sc: Scene):
    """E: в кассовом помещении одновременно >= 3 человек."""
    out = []
    for t0, t1 in runs(sc.n_people >= P["crowd_n"], sc.times, P["crowd_min_sec"]):
        # самый поздний из вошедших к этому моменту посетителей
        newest = [tr for tr in sc.visitor_tracks() if tr.start <= t0 + 0.5 and tr.end >= t0]
        ta = arrival_time(sc, max(newest, key=lambda tr: tr.start)) if newest else t0
        ta = min(max(ta, t0), t1)
        out.append(Violation("E_unauthorized_person", ta, 1.0, [ta], {"from": t0, "until": t1}))
    return out


def rule_behind_cashier(sc: Scene):
    """E: посетитель стоит рядом с кассиром на её стороне стола (та же глубина сцены)."""
    out = []
    cash = sc.cashier_tracks()
    if not cash:
        return out
    ct = cash[0]
    for vt in sc.visitor_tracks():
        hits = []
        for t, vb in zip(vt.t, vt.boxes):
            cb = sc.box_at(ct, t)
            if cb is None:
                continue
            same_depth = abs(vb[3] - cb[3]) < P["side_depth"] * sc.H
            h_overlap = min(vb[2], cb[2]) - max(vb[0], cb[0]) > -0.15 * sc.W
            standing = (vb[3] - vb[1]) > 0.2 * sc.H
            hits.append(t if (same_depth and h_overlap and standing) else None)
        mask = np.array([h is not None for h in hits])
        ts = np.array(vt.t)
        for t0, t1 in runs(mask, ts, P["side_min_sec"]):
            out.append(Violation("E_unauthorized_person", t0, 0.8, [t0 + 0.5], {"until": t1, "kind": "cashier_side"}))
    return out


def rule_bag_no_check(sc: Scene, bag):
    """F: сумка появилась на столе, принёсший человек ушёл слишком быстро."""
    out = []
    on = runs(bag >= P["bag_thr"], sc.owl_times, 0.4)
    for t0, _ in on:
        if series_max(sc, bag, t0 - 3.0, t0 - 0.1) >= P["bag_thr"]:
            continue  # сумка уже лежала — это не момент передачи
        # трек, который заканчивается вскоре после появления сумки (не кассир: тот остаётся)
        leavers = [tr for tr in sc.tracks if t0 - 2 <= tr.end <= t0 + P["bag_leave_sec"]
                   and tr.end < sc.dur - 1.0 and tr.start > 0.5]
        if not leavers:
            continue
        t_left = max(tr.end for tr in leavers)
        after = sc.n_people[(sc.times > t_left + 1.0) & (sc.times < t_left + 3.0)]
        if len(after) and np.median(after) >= 1:  # кассир остался с сумкой
            out.append(Violation("F_bag_no_check", t0, series_max(sc, bag, t0, t0 + 1), [t0, t0 + 1.0],
                                 {"leaver_end": t_left}))
    return out


def rule_personal(sc: Scene, personal):
    """G: личные вещи (сумка, верхняя одежда) на рабочем столе."""
    out = []
    for t0, t1 in runs(personal >= P["personal_thr"], sc.owl_times, P["personal_min_sec"]):
        out.append(Violation("G_personal_items", t0, series_max(sc, personal, t0, t1), [t0 + 0.5],
                             {"until": t1}))
    return out


# ---------- v1: правила на ответах локальной VLM (кэш data/vlm_cache) ----------

def _ans(cache, qid, t, field):
    a = cache.get(qid, t)
    return None if a is None or field not in a else bool(a[field])


def rule_vlm_safe(sc: Scene, cache):
    """B: дверца сейфа открыта, у сейфа никого — >= 2 выборки подряд (или в пустом помещении)."""
    from .vlm_plan import empty_runs, snap
    out = []
    ser = sorted((float(k.split("@")[1]), v) for k, v in cache.data.items() if k.startswith("safe@"))
    run = []
    for t, v in ser + [(1e9, {})]:
        hit = bool(v.get("safe_open")) and not bool(v.get("person_at_safe"))
        if hit:
            run.append(t)
            continue
        if len(run) >= P["vlm_safe_min_hits"]:
            out.append(Violation("B_safe_open", run[0], 0.7, run[:2], {"until": run[-1], "hits": len(run)}))
        run = []
    # сейф открыт в момент, когда все ушли
    for t0, _ in empty_runs(sc):
        if _ans(cache, "safe", snap(t0 + 0.5), "safe_open"):
            td = departure_start(sc, t0)
            out.append(Violation("B_safe_open", td, 0.8, [td, snap(t0 + 0.5)], {"empty_from": t0}))
    return out


def rule_vlm_unattended(sc: Scene, cache):
    """A: помещение опустело, VLM видит на столе деньги или сумку."""
    from .vlm_plan import empty_runs, snap
    out = []
    for t0, t1 in empty_runs(sc):
        ts = [snap(t0 + 0.5), snap(min(t1, t0 + 1.5))]
        hits = [t for t in ts if _ans(cache, "desk", t, "cash_on_desk") or _ans(cache, "desk", t, "bag_on_desk")]
        if len(hits) >= P["vlm_desk_min_hits"]:
            td = departure_start(sc, t0)
            out.append(Violation("A_unattended_valuables", td, 0.75, [td, hits[0]], {"empty_from": t0, "vlm": True}))
    return out


def rule_vlm_cash_next(sc: Scene, cache):
    """C: в момент входа следующего человека VLM видит деньги на столе."""
    from .vlm_plan import snap
    out = []
    for tc in entries(sc, base=1, hold=1.0):
        if not stays_after_entry(sc, tc):
            continue
        te = entry_time(sc, tc)
        ts = [snap(te), snap(te + 1.0)]
        hits = [t for t in ts if _ans(cache, "desk", t, "cash_on_desk")]
        if len(hits) >= P["vlm_desk_min_hits"]:
            out.append(Violation("C_cash_on_desk_next_client", te, 0.75, [te, hits[0]], {"vlm": True}))
    return out


def rule_vlm_no_machine(sc: Scene, cache):
    """D: за визит клиента кассир пересчитывает деньги руками и ни разу не использует счётную машину."""
    from .vlm_plan import visits
    out = []
    for t0, t1 in visits(sc):
        ser = sorted((float(k.split("@")[1]), v) for k, v in cache.data.items()
                     if k.startswith("count@") and t0 - 0.3 <= float(k.split("@")[1]) <= t1 + 0.3)
        hand = [t for t, v in ser if v.get("cash_in_hands") and not v.get("hand_at_counting_machine")]
        machine = [t for t, v in ser if v.get("hand_at_counting_machine")]
        if len(hand) >= P["vlm_hand_min_hits"] and not machine:
            out.append(Violation("D_no_authenticity_check", hand[0], 0.7, hand[:2],
                                 {"visit": [t0, t1], "hand": len(hand), "n": len(ser)}))
    return out


def rule_unlocked_workstation(sc: Scene, cache):
    """L: кассир покинула помещение, оставив компьютер разблокированным (VLM по кропу экрана:
    оба кадра после ухода — «unlocked»). В норме перед уходом экран блокируется."""
    from .vlm_plan import empty_runs, snap
    out = []
    for t0, t1 in empty_runs(sc):
        ts = [snap(t0 + 0.5), snap(min(t1, t0 + 1.5))]
        ans = [a for a in ((cache.get("laptop", t) or {}).get("screen") for t in ts) if a]
        if ans and all(a == "unlocked" for a in ans):  # кадр за концом ролика ответа не даёт
            td = departure_start(sc, t0)
            out.append(Violation("L_workstation_unlocked", td, 0.8, [td, ts[0]], {"empty_from": t0, "screen": ans}))
    return out


def rule_door_left_open(sc: Scene, cache):
    """H: кассир вошла в пустое кассовое помещение и не закрыла за собой дверь.
    VLM по кропу двери через 1, 3 и 5 с после того, как кассир села: «открыта» хотя бы в 2 из 3.
    Таймкод — середина между появлением за стеклом и посадкой (момент прохода через дверь)."""
    from .vlm_plan import cashier_entry, snap
    ce = cashier_entry(sc)
    if ce is None:
        return []
    t_in, t_sit = ce
    pairs = [(snap(t_sit + dt), (cache.get("door", snap(t_sit + dt)) or {}).get("door")) for dt in (1.0, 3.0, 5.0)]
    pairs = [(t, a) for t, a in pairs if a]
    opened = [t for t, a in pairs if a == "open"]
    if len(pairs) >= 2 and len(opened) >= 2:
        tc = (t_in + t_sit) / 2
        # доказательство — кадр, где VLM видит дверь открытой
        return [Violation("H_door_open", tc, 0.7, [tc, opened[0]], {"door": [a for _, a in pairs], "sit": t_sit})]
    return []


def rule_vlm_no_id(sc: Scene, cache):
    """I: за визит деньги появились в кадре раньше, чем документ клиента (или документа не видно)."""
    from .vlm_plan import visits
    out = []
    for t0, t1 in visits(sc):
        def times(prefix, field):
            return sorted(float(k.split("@")[1]) for k, v in cache.data.items()
                          if k.startswith(prefix) and t0 - 0.3 <= float(k.split("@")[1]) <= t1 + 0.3 and v.get(field))
        money, ids = times("count@", "cash_visible"), times("id@", "id_visible")
        if len(money) >= P["vlm_money_min_hits"] and (not ids or ids[0] > money[0]):
            out.append(Violation("I_no_client_id", money[0], 0.65, money[:2],
                                 {"visit": [t0, t1], "money": money[:4], "id": ids[:4]}))
    return out


def rule_safe_open(sc: Scene):
    """B: кассир стояла у сейфа и отошла (села за стол или вышла), оставив дверцу открытой.

    Почему не «открыт N секунд»: в нормальных роликах кассир работает с открытым сейфом и стоя
    (X-25, 10 с), и прямо со стула (X-8, X-14). Нарушение — уход от открытого сейфа (X-20).
    Классификатор надёжен, пока человек у сейфа, поэтому смотрим состояние в момент ухода."""
    if not sc.safe_prob or sc.safe is None:
        return []
    ct = sc.cashier_tracks()
    if not ct:
        return []
    tr = ct[0]
    t = np.array([x[0] for x in sc.safe_prob])
    p = np.array([x[1] for x in sc.safe_prob])
    ps = np.array([np.median(p[max(0, i - 1):i + 2]) for i in range(len(p))])
    tt_arr = np.array(tr.t)
    standing = np.array([(b[3] - b[1]) >= P["stand_h"] * sc.H for b in tr.boxes])
    near = np.array([max(sc.safe[0] - b[2], b[0] - sc.safe[2], 0) <= P["safe_near"] * sc.W for b in tr.boxes])
    out = []
    # моменты ухода от сейфа: стояла рядом -> села, или трек кассира закончился (вышла)
    events = []
    for i in range(1, len(tt_arr)):
        if standing[i - 1] and not standing[i]:
            later = (tt_arr > tt_arr[i]) & (tt_arr <= tt_arr[i] + 1.5)
            if later.any() and not standing[later].any():
                events.append(float(tt_arr[i]))
    if tr.end < sc.dur - 1.0:
        events.append(float(tr.end))
    for te in events:
        was_near = near[(tt_arr >= te - 2.0) & (tt_arr <= te)].any() and standing[(tt_arr >= te - 2.0) & (tt_arr <= te)].any()
        if not was_near:
            continue
        m = (t >= te - P["safe_before"]) & (t <= te)
        if m.any() and ps[m].mean() >= P["safe_thr"]:
            out.append(Violation("B_safe_open", te, float(ps[m].mean()), [te, min(sc.dur - 0.1, te + 2.0)],
                                 {"p_before": float(ps[m].mean())}))
    return out


def rule_drinking(sc: Scene, frames_det):
    """G: кассир пьёт на рабочем месте — стакан/бутылка поднят выше края стола у человека
    (не меньше половины кадров окна 2 с). Стакан, просто стоящий на столе, нарушением не считается (X-25)."""
    top = sc.desk[1] if sc.desk else 0.47 * sc.H
    hits = []
    for k, dets in enumerate(frames_det):
        for l, s, *b in dets:
            if l not in ("cup", "bottle", "wine glass") or s < 0.35:
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if cy < top - 0.01 * sc.H and any(pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3] for _, pb in sc.persons[k]):
                hits.append(float(sc.times[k]))
                break
    # порог — доля кадров окна 2 с, чтобы правило не зависело от частоты кадров детектора
    fps = (len(sc.times) - 1) / max(1e-6, sc.times[-1] - sc.times[0]) if len(sc.times) > 1 else 5.0
    need = max(3, int(round(P["drink_frac"] * 2.0 * fps)))
    for i, t0 in enumerate(hits):
        if sum(1 for t in hits[i:] if t - t0 <= 2.0) >= need:
            return [Violation("G_personal_items", t0, 0.7, [t0 + 0.5], {"kind": "drinking", "hits": len(hits)})]
    return []


def dedup(vs, gap=3.0):
    """Убрать повторы одного типа, ближе gap секунд."""
    vs = sorted(vs, key=lambda v: (v.type, v.t))
    out = []
    for v in vs:
        if out and out[-1].type == v.type and v.t - out[-1].t < gap:
            continue
        out.append(v)
    return sorted(out, key=lambda v: v.t)


VLM_RULES = {"B": rule_vlm_safe, "A": rule_vlm_unattended, "C": rule_vlm_cash_next, "D": rule_vlm_no_machine,
             "I": rule_vlm_no_id, "L": rule_unlocked_workstation, "H": rule_door_left_open}


def detect(sc: Scene, vlm_cache=None, vlm_rules="ABCD", use_v0=True) -> list[Violation]:
    money = sc.obj_on_desk(MONEY_Q)
    bag = sc.obj_on_desk(BAG_Q)
    personal = sc.obj_on_desk(PERSONAL_Q)
    vs = []
    if vlm_cache is not None:
        for k in vlm_rules:
            vs += VLM_RULES[k](sc, vlm_cache)
    if not use_v0:
        return dedup(vs)
    vs += rule_unattended(sc, money, bag)
    vs += rule_safe_open(sc)
    if sc.frames_det is not None:
        vs += rule_drinking(sc, sc.frames_det)
    # rule_left_without_safe отключено: в 2D рамка сидящего кассира перекрывает сейф,
    # «подход к сейфу» так не отличить (FP на X-27). Нужна поза/глубина или VLM.
    vs += rule_cash_next_client(sc, money)
    vs += rule_crowd(sc)
    vs += rule_behind_cashier(sc)
    vs += rule_bag_no_check(sc, bag)
    vs += rule_personal(sc, personal)
    return dedup(vs)
