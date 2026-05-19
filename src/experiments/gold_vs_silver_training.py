"""Gold-only and hybrid cascade training experiment.

For each (cat, attr) where v2 gold has labels:
  1. Split v2 gold codes 80/20 (random, seed=42).
  2. Silver-only baseline: predict on 20% test using existing silver-trained XGB.
  3. Gold-only: train new XGB on 80% gold embeddings, predict on 20% test.
  4. Hybrid: train XGB on full silver + 80% gold (gold sample_weight=5), predict on 20% test.

Accuracy computed only on non-null gold cells.

Output: per (cat, attr, mode) accuracy + n_test, plus per-cat summary.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.common import MODELS_DIR, PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)

OFF_CATS = ["pasta", "chocolate", "cheeses"]
RANDOM_STATE = 42
TEST_FRACTION = 0.2


def train_xgb_and_score(X_tr, y_tr, X_te, y_te,
                         sample_weight: Optional[np.ndarray] = None) -> float:
    """Fit XGB on (X_tr, y_tr) with optional sample weights, score on (X_te, y_te)."""
    train_classes = sorted(set(y_tr))
    if len(train_classes) < 2:
        # Degenerate; predict majority class
        if len(y_te) == 0:
            return float("nan")
        majority = train_classes[0] if train_classes else y_te[0]
        return float((y_te == majority).mean())
    remap = {c: i for i, c in enumerate(train_classes)}
    y_tr_r = np.array([remap[c] for c in y_tr])
    y_te_r = np.array([remap.get(c, -1) for c in y_te])  # unseen test classes = wrong

    n_classes = len(train_classes)
    common_kwargs = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        tree_method="hist", verbosity=0,
    )
    if n_classes == 2:
        pos = int((y_tr_r == 1).sum())
        neg = int((y_tr_r == 0).sum())
        spw = max(neg / max(pos, 1), 0.5)
        clf = xgb.XGBClassifier(scale_pos_weight=spw, **common_kwargs)
    else:
        clf = xgb.XGBClassifier(objective="multi:softmax",
                                num_class=n_classes, **common_kwargs)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    clf.fit(X_tr, y_tr_r, **fit_kwargs)
    pred = clf.predict(X_te)
    valid = y_te_r >= 0
    if not valid.any():
        return float("nan")
    return float(accuracy_score(y_te_r[valid], pred[valid]))


def _load_silver_xgb(category: str, attr: str):
    """Return (model, le) for silver-trained model, or (None, None)."""
    base = os.path.join(MODELS_DIR, f"{category}_stratified_{attr}")
    mp, lp = base + "_xgb.pkl", base + "_le.pkl"
    if not (os.path.exists(mp) and os.path.exists(lp)):
        return None, None
    with open(mp, "rb") as f:
        model = pickle.load(f)
    with open(lp, "rb") as f:
        le = pickle.load(f)
    return model, le


def _silver_predict(model, le, X_te) -> np.ndarray | None:
    """Predict labels using silver-trained model. None if degenerate."""
    if model is None or le is None:
        return None
    proba = model.predict_proba(X_te)
    top = proba.argmax(axis=1)
    return le.inverse_transform(top)


def run_one_attr(cat: str, attr: str, gold_long: pd.DataFrame,
                 silver: pd.DataFrame, emb: np.ndarray,
                 code_to_idx: dict) -> list[dict]:
    """For one (cat, attr): build 20% test split, run 3 modes."""
    cat_gold = gold_long[(gold_long["category"] == cat)
                         & (gold_long["attr"] == attr)
                         & ~gold_long["gold_is_null"]].copy()
    cat_gold["code"] = cat_gold["code"].astype(str)
    cat_gold = cat_gold[cat_gold["code"].isin(code_to_idx)]
    if len(cat_gold) < 20:
        logger.info("[%s/%s] only %d non-null gold cells, skipping", cat, attr, len(cat_gold))
        return []
    codes = cat_gold["code"].tolist()
    train_codes, test_codes = train_test_split(
        codes, test_size=TEST_FRACTION, random_state=RANDOM_STATE)
    train_codes_set = set(train_codes)
    test_codes_set = set(test_codes)

    train_gold = cat_gold[cat_gold["code"].isin(train_codes_set)].copy()
    test_gold = cat_gold[cat_gold["code"].isin(test_codes_set)].copy()

    train_idx = np.array([code_to_idx[c] for c in train_gold["code"]])
    test_idx = np.array([code_to_idx[c] for c in test_gold["code"]])

    X_tr_gold = emb[train_idx]
    y_tr_gold = train_gold["gold_value"].astype(str).values
    X_te = emb[test_idx]
    y_te = test_gold["gold_value"].astype(str).values

    rows = []

    # (1) silver-only baseline: predict using existing silver model
    sm, sle = _load_silver_xgb(cat, attr)
    silver_pred = _silver_predict(sm, sle, X_te)
    if silver_pred is not None:
        silver_acc = float(accuracy_score(y_te, silver_pred.astype(str)))
    else:
        silver_acc = float("nan")
    rows.append({"category": cat, "attr": attr, "mode": "silver_only",
                 "accuracy": silver_acc, "n_train": "—", "n_test": len(test_codes)})

    # (2) gold-only: train on 80% gold, predict on 20%
    gold_acc = train_xgb_and_score(X_tr_gold, y_tr_gold, X_te, y_te)
    rows.append({"category": cat, "attr": attr, "mode": "gold_only",
                 "accuracy": gold_acc, "n_train": len(train_codes), "n_test": len(test_codes)})

    # (3) hybrid: silver (all rows for this cat) + 80% gold, gold weighted 5x
    # Silver labels for attr — must come from silver wide parquet.
    if attr not in silver.columns:
        rows.append({"category": cat, "attr": attr, "mode": "hybrid",
                     "accuracy": float("nan"), "n_train": "—", "n_test": len(test_codes)})
        return rows
    silver_for_attr = silver[silver[attr].notna()].copy()
    silver_for_attr["code"] = silver_for_attr["code"].astype(str)
    # Exclude codes that are in test set (avoid leakage)
    silver_for_attr = silver_for_attr[~silver_for_attr["code"].isin(test_codes_set)]
    silver_idx = np.array([code_to_idx[c] for c in silver_for_attr["code"]
                           if c in code_to_idx])
    # Re-align silver_for_attr to those that have embeddings
    silver_for_attr = silver_for_attr[silver_for_attr["code"].isin(code_to_idx)].iloc[:len(silver_idx)]
    silver_y = silver_for_attr[attr].astype(str).values
    X_silver = emb[silver_idx]
    # Combine — but DROP rows from silver where the same code is in train_gold
    # (avoid double-counting and silver-overrules-gold conflicts)
    silver_codes_arr = silver_for_attr["code"].values
    keep = ~np.isin(silver_codes_arr, list(train_codes_set))
    X_silver = X_silver[keep]
    silver_y = silver_y[keep]

    X_hybrid = np.vstack([X_silver, X_tr_gold])
    y_hybrid = np.concatenate([silver_y, y_tr_gold])
    w_hybrid = np.concatenate([np.ones(len(silver_y)),
                               5.0 * np.ones(len(y_tr_gold))])
    hybrid_acc = train_xgb_and_score(X_hybrid, y_hybrid, X_te, y_te,
                                     sample_weight=w_hybrid)
    rows.append({"category": cat, "attr": attr, "mode": "hybrid",
                 "accuracy": hybrid_acc,
                 "n_train": f"{len(silver_y)} silver + {len(y_tr_gold)} gold(w=5)",
                 "n_test": len(test_codes)})
    return rows


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--out",
                   default=str(Path(PROCESSED_DIR) / "gold_vs_silver_training.parquet"))
    args = p.parse_args()

    gold = pd.read_parquet(Path(PROCESSED_DIR) / "consensus_gold_v2_off_grounded.parquet")
    gold["code"] = gold["code"].astype(str)

    all_rows = []
    for cat in OFF_CATS:
        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet")
        silver["code"] = silver["code"].astype(str)
        emb = np.load(Path(PROCESSED_DIR) / f"{cat}_stratified_embeddings.npy")
        # Build code → index map (silver row order = embedding row order)
        code_to_idx = {c: i for i, c in enumerate(silver["code"].tolist())}

        attrs = sorted(gold[gold["category"] == cat]["attr"].unique())
        for attr in attrs:
            rows = run_one_attr(cat, attr, gold, silver, emb, code_to_idx)
            for r in rows:
                logger.info("[%s/%s/%s] acc=%.3f n_test=%s",
                            cat, attr, r["mode"], r["accuracy"], r["n_test"])
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df["n_train"] = df["n_train"].astype(str)
    df.to_parquet(args.out, index=False)

    # Pivot summary
    print("\n=== Summary: accuracy by mode (per cat, attr) ===")
    pivot = df.pivot_table(index=["category", "attr"], columns="mode",
                           values="accuracy")
    if {"silver_only", "gold_only", "hybrid"}.issubset(pivot.columns):
        pivot["gold_vs_silver_pp"] = (pivot["gold_only"] - pivot["silver_only"]) * 100
        pivot["hybrid_vs_silver_pp"] = (pivot["hybrid"] - pivot["silver_only"]) * 100
    print(pivot.to_string())
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
