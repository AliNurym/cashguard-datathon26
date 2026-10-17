"""Прогон правил по кэшу признаков -> predictions JSON для evaluate.py.

    python scripts/predict.py runs/pred_v0.json [--only X-1,X-2] [--verbose]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cashguard.labels import load_videos  # noqa: E402
from cashguard.registry import describe  # noqa: E402
from cashguard.rules import detect  # noqa: E402
from cashguard.scene import CACHE, build_scene  # noqa: E402
from cashguard.vlm import VLMCache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--vlm", default="", help="правила на VLM, напр. ABCD; пусто — без VLM")
    ap.add_argument("--no-v0", action="store_true", help="только VLM-правила")
    ap.add_argument("--safe-oof", default="", help="LOVO|LOSO: out-of-fold вероятности сейфа (data/safe_ds/oof_*)")
    ap.add_argument("--money-folds", action="store_true", help="классификатор денег: модель фолда LOSO для ракурса ролика")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    safe_oof = {}
    if a.safe_oof:
        from cashguard.labels import ROOT, VARIANT
        safe_oof = json.loads((ROOT / "data" / "safe_ds" / f"oof_{a.safe_oof}" / f"oof_{VARIANT or 'orig'}.json").read_text())
    money_models, setups = {}, {}
    if a.money_folds:
        from cashguard.embed import Embedder
        from cashguard.labels import ROOT as R
        from cashguard.states import MoneyClassifier
        setups = json.loads((R / "labels" / "setups.json").read_text())
        emb = Embedder(no_crop=True)
        money_models = {g: MoneyClassifier(emb, R / "models" / "money_folds" / f"LOSO_{g}.joblib")
                        for g in set(setups.values())}
    preds = {}
    for v in load_videos():
        if only and v.vid not in only:
            continue
        if not (CACHE / f"{v.vid}.json").exists():
            continue
        sc = build_scene(v.vid)
        if a.safe_oof:
            sc.safe_prob = safe_oof.get(v.vid)
        if a.money_folds:
            mc = money_models[setups[v.vid]]
            sc.money_fn = (lambda ts, mc=mc, path=v.path, sc=sc: mc.at(path, sc, ts))
        cache = VLMCache(v.vid) if a.vlm else None
        vs = detect(sc, vlm_cache=cache, vlm_rules=a.vlm, use_v0=not a.no_v0)
        preds[v.vid] = {
            "verdict": "violation" if vs else "no_violation",
            "violations": [{"type": x.type, "timecode_sec": round(x.t, 1), "score": round(x.score, 3),
                            "evidence_t": [round(e, 1) for e in x.evidence_t],
                            **{k: v for k, v in describe(x.type, x.t, x.details).items() if k != "type"},
                            "type_name": describe(x.type, x.t, x.details)["type"],
                            "details": x.details} for x in vs],
        }
        if a.verbose:
            gt = f"GT {v.gt_times}" if v.violation else "GT -"
            print(f"{v.vid:6} {gt:14} -> " + ", ".join(f"{x.type[:1]}@{x.t:.1f}({x.score:.2f})" for x in vs))
    Path(a.out).write_text(json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
