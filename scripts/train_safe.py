"""Классификатор «сейф открыт/закрыт»: DINOv2 (заморожен) + логистическая регрессия.

    python scripts/train_safe.py            # собрать признаки, оценить LOVO/LOSO, сохранить модель
Данные: кропы сейфа 2 раза в секунду с оригиналов и синтетических ракурсов (persp_l, flip),
разметка labels/safe_state.csv (всё вне интервалов open/ignore — закрыт).
Модель: models/safe_clf.joblib.
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DS = ROOT / "data" / ("safe_ds" if os.environ.get("SAFE_MASK", "1") == "1" else "safe_ds_nomask")
VARIANTS = ["", "persp_l", "flip"]


def load_intervals():
    iv = {}
    with open(ROOT / "labels" / "safe_state.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            iv.setdefault(r["video"], []).append((float(r["t_from"]), float(r["t_to"]), r["state"]))
    return iv


def label_at(iv, vid, t):
    for a, b, s in iv.get(vid, []):
        if a - 0.25 <= t <= b + 0.25:
            return s
    return "closed"


MASK_PERSONS = os.environ.get("SAFE_MASK", "1") == "1"


def build_variant(variant):
    """Запускается в отдельном процессе с CASHGUARD_VARIANT (пути кэша фиксируются при импорте)."""
    from cashguard.embed import Embedder, augment
    from cashguard.labels import load_videos
    from cashguard.scene import build_scene
    from cashguard.video import iter_frames
    from cashguard.states import mask_persons
    from cashguard.vlm import crop_region
    iv = load_intervals()
    emb, rng = Embedder(), np.random.default_rng(0)
    X, y, vids, ts, aug_flag = [], [], [], [], []
    for v in load_videos():
        sc = build_scene(v.vid)
        if not sc.safe:
            continue
        crops, labs, tt = [], [], []
        for t, f in iter_frames(v.path, 2.0):
            lab = label_at(iv, v.vid, t)
            if lab == "ignore":
                continue
            if MASK_PERSONS:
                f = mask_persons(f, sc, t)
            crops.append(crop_region(f, sc.safe, 0.6, 0.3, 0.2, min_side=0))
            labs.append(1 if lab == "open" else 0)
            tt.append(t)
        # аугментации одинаково для обоих классов (иначе «искажённый кадр = открыт»): позитивы x4,
        # негативы — каждый 4-й кадр x1; только оригинал — синтетика и так «другой ракурс»
        extra = []
        if not variant:
            for i, (c, l, t) in enumerate(zip(crops, labs, tt)):
                for _ in range(4 if l else (1 if i % 4 == 0 else 0)):
                    extra.append((augment(c, rng), l, t))
        allc = crops + [e[0] for e in extra]
        E = emb(allc)
        X.append(E)
        y += labs + [e[1] for e in extra]
        ts += tt + [e[2] for e in extra]
        vids += [v.vid] * len(allc)
        aug_flag += [0] * len(crops) + [1] * len(extra)
    DS.mkdir(parents=True, exist_ok=True)
    np.savez(DS / f"{variant or 'orig'}.npz", X=np.concatenate(X), y=np.array(y), vid=np.array(vids),
             t=np.array(ts), aug=np.array(aug_flag))


def evaluate(data, setups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def fit(mask):
        X = np.concatenate([d["X"][m] for d, m in zip(data.values(), mask)])
        y = np.concatenate([d["y"][m] for d, m in zip(data.values(), mask)])
        return LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000).fit(X, y)

    for mode in ("LOVO", "LOSO"):
        groups = sorted(set(data["orig"]["vid"])) if mode == "LOVO" else sorted(set(setups.values()))
        res = {k: ([], []) for k in data}
        for g in groups:
            held = {v for v in set(data["orig"]["vid"]) if (v == g if mode == "LOVO" else setups[v] == g)}
            if not held:
                continue
            clf = fit([~np.isin(d["vid"], list(held)) for d in data.values()])
            for k, d in data.items():
                m = np.isin(d["vid"], list(held)) & (d["aug"] == 0)
                if m.any():
                    res[k][0].extend(d["y"][m])
                    res[k][1].extend(clf.predict_proba(d["X"][m])[:, 1])
        for k, (yt, p) in res.items():
            yt, p = np.array(yt), np.array(p)
            pr = p >= 0.5
            tp, fp, fn = int((pr & (yt == 1)).sum()), int((pr & (yt == 0)).sum()), int((~pr & (yt == 1)).sum())
            auc = roc_auc_score(yt, p) if len(set(yt)) > 1 else float("nan")
            print(f"{mode:4} {k:8} AUC={auc:.3f}  кадры: TP={tp} FN={fn} FP={fp} (из {int((yt == 0).sum())} закрытых)")
    return fit([np.ones(len(d["y"]), bool) for d in data.values()])


def main():
    if os.environ.get("BUILD_ONLY") is not None:
        build_variant(os.environ.get("CASHGUARD_VARIANT", ""))
        return
    for var in VARIANTS:
        if not (DS / f"{var or 'orig'}.npz").exists():
            env = dict(os.environ, CASHGUARD_VARIANT=var, BUILD_ONLY="1")
            subprocess.run([sys.executable, __file__], env=env, check=True)
    data = {var or "orig": dict(np.load(DS / f"{var or 'orig'}.npz")) for var in VARIANTS}
    setups = json.loads((ROOT / "labels" / "setups.json").read_text())
    for k, d in data.items():
        print(f"{k}: {len(d['y'])} кропов, открыт {int(d['y'].sum())}")
    clf = evaluate(data, setups)
    import joblib
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(clf, ROOT / "models" / "safe_clf.joblib")
    print("saved models/safe_clf.joblib")


if __name__ == "__main__":
    main()


def export_oof(mode="LOSO"):
    """Out-of-fold вероятности «открыт» по каждому кадру (модель не видела ракурс/ролик) —
    для честной оценки правила B на уровне роликов. data/safe_ds/oof_<variant>.json"""
    from sklearn.linear_model import LogisticRegression
    data = {var or "orig": dict(np.load(DS / f"{var or 'orig'}.npz")) for var in VARIANTS}
    setups = json.loads((ROOT / "labels" / "setups.json").read_text())
    vids_all = sorted(set(data["orig"]["vid"]))
    groups = vids_all if mode == "LOVO" else sorted(set(setups.values()))
    out = {k: {} for k in data}
    for g in groups:
        held = {v for v in vids_all if (v == g if mode == "LOVO" else setups[v] == g)}
        X = np.concatenate([d["X"][~np.isin(d["vid"], list(held))] for d in data.values()])
        y = np.concatenate([d["y"][~np.isin(d["vid"], list(held))] for d in data.values()])
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000).fit(X, y)
        for k, d in data.items():
            for v in held:
                m = (d["vid"] == v) & (d["aug"] == 0)
                if m.any():
                    p = clf.predict_proba(d["X"][m])[:, 1]
                    out[k][v] = sorted(zip(d["t"][m].round(2).tolist(), p.round(3).tolist()))
    for k, o in out.items():
        (DS / f"oof_{k}.json").write_text(json.dumps(o))
    print("oof saved", {k: len(o) for k, o in out.items()})


if __name__ == "__main__" and os.environ.get("EXPORT_OOF"):
    export_oof(os.environ.get("EXPORT_OOF"))
