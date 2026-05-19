"""K-fold isotonic calibration on the gold distribution.

Addresses §5.4 limitation 4: the current isotonic calibration (silver-fit,
gold-eval) showed average ECE deterioration on the gold test set. This script
fits isotonic regression on the gold distribution itself via 5-fold CV
(unbiased estimate of calibration quality at gold-distribution).

For each (cat, attr) ∈ {pasta, chocolate, cheeses} × production attrs:
  - "before"     = ECE of raw XGB confidence on full gold test set.
  - "after_kfold" = mean ECE across 5 folds, where each fold's isotonic
                    mapping is fitted on the other 4 folds.

Output: datasets/processed/ece_kfold_gold_table.parquet.

Entry: python -m src.diagnostics.ml.ece_kfold_gold
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from src.common import MODELS_DIR, PROCESSED_DIR, setup_logging
from src.diagnostics.ml.ece_table import compute_ece_from_conf, load_model

logger = logging.getLogger(__name__)

CATS = ["pasta", "chocolate", "cheeses"]
N_BINS = 10
N_SPLITS = 5
SEED = 42


def run_attr(
    cat: str,
    attr: str,
    silver: pd.DataFrame,
    embeddings: np.ndarray,
    gold_long: pd.DataFrame,
) -> dict | None:
    clf, le = load_model(cat, attr)
    if clf is None:
        return None

    g = gold_long[(gold_long["category"] == cat) & (gold_long["attr"] == attr)].copy()
    g = g[~g["gold_is_null"].astype(bool)]
    if len(g) == 0:
        return None

    g["code"] = g["code"].astype(str)
    silver = silver.copy()
    silver["code"] = silver["code"].astype(str)
    code_to_idx = {c: i for i, c in enumerate(silver["code"].values)}
    g["row_idx"] = g["code"].map(code_to_idx)
    g = g.dropna(subset=["row_idx"]).copy()
    if len(g) < 5:
        return None
    g["row_idx"] = g["row_idx"].astype(int)

    valid_classes = set(map(str, le.classes_))
    g["gold_value"] = g["gold_value"].astype(str)
    g = g[g["gold_value"].isin(valid_classes)].copy()
    if len(g) < N_SPLITS * 5:
        return {
            "category": cat,
            "attr": attr,
            "n_test": int(len(g)),
            "ece_before": None,
            "ece_after_kfold": None,
            "ece_reduction_pp": None,
            "note": f"n_test<{N_SPLITS*5}_skip",
        }

    test_idx = g["row_idx"].values
    X_test = embeddings[test_idx]
    y_test_labels = g["gold_value"].values

    proba_test = clf.predict_proba(X_test)
    pred_idx = proba_test.argmax(axis=1)
    pred_labels = np.array([str(c) for c in le.classes_])[pred_idx]
    conf_raw = proba_test.max(axis=1)
    correct = (pred_labels == y_test_labels).astype(float)

    ece_before = compute_ece_from_conf(conf_raw, correct)

    # K-fold isotonic
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    ece_after_folds = []
    for fold_idx, (cal_idx, eval_idx) in enumerate(kf.split(conf_raw)):
        cal_conf, cal_correct = conf_raw[cal_idx], correct[cal_idx]
        if len(np.unique(cal_correct)) < 2:
            # cannot fit isotonic — fall back to identity on this fold
            ece_after_folds.append(
                compute_ece_from_conf(conf_raw[eval_idx], correct[eval_idx])
            )
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(cal_conf, cal_correct)
        conf_eval_iso = iso.predict(conf_raw[eval_idx])
        ece_after_folds.append(
            compute_ece_from_conf(conf_eval_iso, correct[eval_idx])
        )

    ece_after = float(np.mean(ece_after_folds))
    return {
        "category": cat,
        "attr": attr,
        "n_test": int(len(y_test_labels)),
        "ece_before": round(ece_before, 4),
        "ece_after_kfold": round(ece_after, 4),
        "ece_reduction_pp": round((ece_before - ece_after) * 100, 2),
        "note": "",
    }


def main():
    setup_logging()
    gold = pd.read_parquet(Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet")
    gold["code"] = gold["code"].astype(str)

    results = []
    for cat in CATS:
        silver_path = Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        emb_path = Path(PROCESSED_DIR) / f"{cat}_stratified_embeddings.npy"
        if not silver_path.exists() or not emb_path.exists():
            continue
        silver = pd.read_parquet(silver_path).reset_index(drop=True)
        embeddings = np.load(emb_path)
        if len(silver) != embeddings.shape[0]:
            continue

        model_glob = sorted(Path(MODELS_DIR).glob(f"{cat}_stratified_*_xgb_hybrid.pkl"))
        attrs = [
            p.name[len(f"{cat}_stratified_"):-len("_xgb_hybrid.pkl")] for p in model_glob
        ]

        for attr in attrs:
            try:
                r = run_attr(cat, attr, silver, embeddings, gold)
                if r is not None:
                    results.append(r)
                    logger.info(
                        "  %s/%s: n=%d before=%s after_kfold=%s",
                        cat, attr, r["n_test"], r["ece_before"], r["ece_after_kfold"],
                    )
            except Exception:
                logger.exception("[%s/%s] failed", cat, attr)

    if not results:
        return
    df = pd.DataFrame(results)
    out = Path(PROCESSED_DIR) / "ece_kfold_gold_table.parquet"
    df.to_parquet(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(df))

    ok = df.dropna(subset=["ece_after_kfold"]).copy()
    print("\n=== ECE K-FOLD ON GOLD ===")
    print(df.to_string(index=False))
    if len(ok) > 0:
        print(f"\nn attrs with k-fold ECE: {len(ok)}")
        print(f"mean ECE before     : {ok['ece_before'].mean():.4f}")
        print(f"mean ECE after_kfold: {ok['ece_after_kfold'].mean():.4f}")
        print(f"mean reduction (pp) : {ok['ece_reduction_pp'].mean():.2f}")
        print(f"attrs improved (>0p.p.): "
              f"{int((ok['ece_reduction_pp'] > 0).sum())}/{len(ok)}")


if __name__ == "__main__":
    main()
