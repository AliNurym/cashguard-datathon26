"""Классификатор «деньги на столе» по плиткам кропа стола: DINOv2 (заморожен) + логрегрессия.

    python scripts/train_money.py
Разметка: предразметка VLM-grounding (data/money_crops/prelabels.json) минус ручные исключения
(labels/money_exclude.txt — кадры с ошибочной предразметкой). Плитка 4x2: положительная, если
в неё попадает центр рамки денег. Обучение на оригиналах + аугментации; модель models/money_clf.joblib.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CROPS = ROOT / "data" / "money_crops"
DS = ROOT / "data" / "money_ds"
GX, GY = 4, 2


def tiles(crop):
    h, w = crop.shape[:2]
    out = []
    for j in range(GY):
        for i in range(GX):
            out.append(crop[j * h // GY:(j + 1) * h // GY, i * w // GX:(i + 1) * w // GX])
    return out


def tile_labels(boxes):
    lab = np.zeros(GX * GY, int)
    for b in boxes:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        i, j = min(GX - 1, int(cx * GX)), min(GY - 1, int(cy * GY))
        lab[j * GX + i] = 1
    return lab


def build():
    from cashguard.embed import Embedder, augment
    pre = json.loads((CROPS / "prelabels.json").read_text())
    excl_path = ROOT / "labels" / "money_exclude.txt"
    excl = set(excl_path.read_text().split()) if excl_path.exists() else set()
    emb, rng = Embedder(no_crop=True), np.random.default_rng(0)
    X, y, vid, key, aug = [], [], [], [], []
    for k, boxes in sorted(pre.items()):
        if k in excl:
            continue
        img = cv2.imread(str(CROPS / f"{k}.jpg"))
        if img is None:
            continue
        lab = tile_labels(boxes)
        ts = tiles(img)
        # аугментации одинаково для позитивов и негативов: иначе модель выучит «искажённый кадр = деньги»
        # аугментируем каждую плитку отдельно: геометрия внутри плитки, метка плитки не меняется
        n_aug = 2 if lab.any() else 1
        reps = [(ts, 0)] + [([augment(t, rng) for t in ts], 1) for _ in range(n_aug)]
        for tl, a in reps:
            X.append(emb(tl))
            y += lab.tolist()
            vid += [k.rsplit("_", 1)[0]] * len(tl)
            key += [k] * len(tl)
            aug += [a] * len(tl)
    DS.mkdir(parents=True, exist_ok=True)
    np.savez(DS / "tiles.npz", X=np.concatenate(X), y=np.array(y), vid=np.array(vid), key=np.array(key), aug=np.array(aug))


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    if not (DS / "tiles.npz").exists():
        build()
    d = dict(np.load(DS / "tiles.npz"))
    setups = json.loads((ROOT / "labels" / "setups.json").read_text())
    print(f"плиток {len(d['y'])}, с деньгами {int(d['y'].sum())}; кадров {len(set(d['key']))}")
    for mode in ("LOVO", "LOSO"):
        vids = sorted(set(d["vid"]))
        groups = vids if mode == "LOVO" else sorted(set(setups[v] for v in vids))
        yt, pt, frame_res = [], [], []
        for g in groups:
            held = [v for v in vids if (v == g if mode == "LOVO" else setups[v] == g)]
            tr, te = ~np.isin(d["vid"], held), np.isin(d["vid"], held) & (d["aug"] == 0)
            if not te.any() or len(set(d["y"][tr])) < 2:
                continue
            clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000).fit(d["X"][tr], d["y"][tr])
            p = clf.predict_proba(d["X"][te])[:, 1]
            yt += d["y"][te].tolist()
            pt += p.tolist()
            # уровень кадра: «есть деньги» = макс по плиткам
            for k in sorted(set(d["key"][te])):
                m = d["key"][te] == k
                frame_res.append((int(d["y"][te][m].max()), float(p[m].max())))
        yt, pt = np.array(yt), np.array(pt)
        fy, fp_ = np.array([r[0] for r in frame_res]), np.array([r[1] for r in frame_res])
        for thr in (0.5, 0.7, 0.85):
            pr = fp_ >= thr
            print(f"{mode} плитки AUC={roc_auc_score(yt, pt):.3f} | кадры AUC={roc_auc_score(fy, fp_):.3f} "
                  f"thr={thr}: TP={int((pr & (fy == 1)).sum())} FN={int((~pr & (fy == 1)).sum())} "
                  f"FP={int((pr & (fy == 0)).sum())} из {int((fy == 0).sum())} пустых")
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000).fit(d["X"], d["y"])
    import joblib
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(clf, ROOT / "models" / "money_clf.joblib")
    print("saved models/money_clf.joblib")


if __name__ == "__main__":
    main()


def save_folds():
    """Модели по фолдам «оставить ракурс» — для честной оценки правил (models/money_folds/)."""
    from sklearn.linear_model import LogisticRegression
    import joblib
    d = dict(np.load(DS / "tiles.npz"))
    setups = json.loads((ROOT / "labels" / "setups.json").read_text())
    out = ROOT / "models" / "money_folds"
    out.mkdir(parents=True, exist_ok=True)
    for g in sorted(set(setups.values())):
        held = [v for v, s in setups.items() if s == g]
        tr = ~np.isin(d["vid"], held)
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000).fit(d["X"][tr], d["y"][tr])
        joblib.dump(clf, out / f"LOSO_{g}.joblib")
    print("folds saved")
