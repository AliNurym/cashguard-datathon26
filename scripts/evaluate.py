"""Оценка предсказаний на обучающем датасете.

Использование:
    python scripts/evaluate.py predictions.json [--tol 2]

predictions.json: {"X-1": {"verdict": "violation", "violations": [{"type": "...", "timecode_sec": 22.0}]}, ...}
Ключи могут быть любым алиасом видео (X-4 или X-47): дубликаты по md5 сливаются.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cashguard.labels import load_types, load_videos, norm_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--tol", type=float, default=2.0, help="допуск таймкода, с")
    a = ap.parse_args()

    raw = json.loads(Path(a.pred).read_text(encoding="utf-8"))
    pred = {norm_name(k): v for k, v in raw.items()}
    # ответ API содержит русское название типа — переводим обратно в код реестра
    from cashguard.registry import REGISTRY
    name2code = {r["name"]: code for code, r in REGISTRY.items()}
    for p in pred.values():
        for x in p.get("violations") or []:
            x["type"] = name2code.get(x.get("type"), x.get("type"))
    videos = load_videos()
    types = load_types()

    tp = fp = tn = fn = 0
    errs, type_hits, type_total = [], 0, 0
    rows = []
    for v in videos:
        p = next((pred[al] for al in v.aliases if al in pred), None)
        if p is None:
            rows.append((v.vid, "GT+" if v.violation else "GT-", "MISSING", "", ""))
            fn += v.violation
            fp += 0
            continue
        pv = p.get("verdict") == "violation"
        ptimes = [x["timecode_sec"] for x in p.get("violations", [])]
        if v.violation and pv:
            tp += 1
        elif v.violation:
            fn += 1
        elif pv:
            fp += 1
        else:
            tn += 1
        detail = []
        for gt in v.gt_times:
            if ptimes:
                j = min(range(len(ptimes)), key=lambda i: abs(ptimes[i] - gt))
                e = ptimes[j] - gt
                errs.append(abs(e))
                gtype = types.get((v.vid, gt))
                ptype = p["violations"][j].get("type", "")
                if gtype and not gtype.startswith("U_"):
                    type_total += 1
                    type_hits += gtype == ptype
                detail.append(f"gt={gt:g} pred={ptimes[j]:.1f} ({e:+.1f}) {gtype or ''} -> {ptype}")
            else:
                detail.append(f"gt={gt:g} pred=—")
        status = {(True, True): "TP", (True, False): "FN", (False, True): "FP", (False, False): "TN"}[(v.violation, pv)]
        extra = "; ".join(detail) if detail else ", ".join(f"{x.get('type')}@{x['timecode_sec']:.1f}" for x in p.get("violations", []))
        rows.append((v.vid, "GT+" if v.violation else "GT-", status, len(ptimes), extra))

    for r in rows:
        print(f"{r[0]:6} {r[1]:4} {r[2]:7} n={r[3]!s:2}  {r[4]}")
    n = tp + fp + tn + fn
    print("\n=== Вердикт (уникальные видео) ===")
    print(f"acc={(tp + tn) / n:.3f}  recall={tp / max(1, tp + fn):.3f}  precision={tp / max(1, tp + fp):.3f}  TP={tp} FN={fn} FP={fp} TN={tn}")
    if errs:
        within = sum(e <= a.tol for e in errs)
        print(f"=== Таймкод ({len(errs)} событий) === MAE={sum(errs) / len(errs):.2f}s  within±{a.tol:g}s: {within}/{len(errs)}")
    if type_total:
        print(f"=== Тип (по уверенной ручной разметке) === {type_hits}/{type_total}")


if __name__ == "__main__":
    main()
