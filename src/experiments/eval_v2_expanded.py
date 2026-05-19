"""Honest 80/20 eval: silver_only vs hybrid_v1 (717 gold) vs hybrid_v2 (2120 gold).

For each (cat, attr):
  - Split EXPANDED gold codes 80/20 (random seed=42)
  - silver_only: existing pre-trained silver model (no retrain), predict on 20% test
  - hybrid_v1:   train XGB on full silver + 80% of ORIGINAL 717-only gold (gold_weight=5)
  - hybrid_v2:   train XGB on full silver + 80% of EXPANDED gold (gold_weight=5)
  - Eval all 3 on the same 20% held-out

Accuracy computed only on non-null gold cells.

Output: datasets/processed/expanded_eval_80_20.parquet
        columns: category, attr, mode, accuracy, n_test, n_train_gold
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.common import MODELS_DIR, PROCESSED_DIR, setup_logging
from src.experiments.gold_vs_silver_training import train_xgb_and_score

logger = logging.getLogger(__name__)

OFF_CATS = ["pasta", "chocolate", "cheeses"]
RANDOM_STATE = 42
TEST_FRACTION = 0.2
GOLD_WEIGHT = 5.0

EXPANDED_GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet"
ORIG_GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_off_grounded.parquet"
OUT_PATH = Path(PROCESSED_DIR) / "expanded_eval_80_20.parquet"


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


def _silver_predict(model, le, X_te) -> Optional[np.ndarray]:
    """Predict labels using silver-trained model. None if degenerate."""
    if model is None or le is None:
        return None
    proba = model.predict_proba(X_te)
    top = proba.argmax(axis=1)
    return le.inverse_transform(top)


def _build_hybrid_train(
    cat: str,
    attr: str,
    silver: pd.DataFrame,
    emb: np.ndarray,
    code_to_idx: dict,
    gold_long: pd.DataFrame,
    train_codes_set: set,
    test_codes_set: set,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, weights) for hybrid training: silver + gold subset."""
    if attr not in silver.columns:
        return np.array([]), np.array([]), np.array([])

    # Silver rows with non-null attr labels, excluding test codes
    silver_for_attr = silver[silver[attr].notna()].copy()
    silver_for_attr["code"] = silver_for_attr["code"].astype(str)
    silver_for_attr = silver_for_attr[~silver_for_attr["code"].isin(test_codes_set)]
    # Keep only those that have embeddings
    silver_for_attr = silver_for_attr[silver_for_attr["code"].isin(code_to_idx)]
    # Exclude codes that are in train_codes_set (already in gold train — avoid conflict)
    silver_for_attr = silver_for_attr[~silver_for_attr["code"].isin(train_codes_set)]

    silver_idx = np.array([code_to_idx[c] for c in silver_for_attr["code"]])
    X_silver = emb[silver_idx]
    y_silver = silver_for_attr[attr].astype(str).values
    w_silver = np.ones(len(y_silver))

    # Gold train portion for this attr
    train_gold = gold_long[
        (gold_long["category"] == cat)
        & (gold_long["attr"] == attr)
        & ~gold_long["gold_is_null"]
        & gold_long["code"].isin(train_codes_set)
    ].copy()
    train_gold = train_gold[train_gold["code"].isin(code_to_idx)]

    if len(train_gold) == 0:
        return X_silver, y_silver, w_silver

    train_idx = np.array([code_to_idx[c] for c in train_gold["code"]])
    X_gold = emb[train_idx]
    y_gold = train_gold["gold_value"].astype(str).values
    w_gold = GOLD_WEIGHT * np.ones(len(y_gold))

    X_hybrid = np.vstack([X_silver, X_gold])
    y_hybrid = np.concatenate([y_silver, y_gold])
    w_hybrid = np.concatenate([w_silver, w_gold])
    return X_hybrid, y_hybrid, w_hybrid


def run_one_attr(
    cat: str,
    attr: str,
    expanded_gold: pd.DataFrame,
    orig_gold: pd.DataFrame,
    silver: pd.DataFrame,
    emb: np.ndarray,
    code_to_idx: dict,
) -> list[dict]:
    """For one (cat, attr): build shared 20% test from expanded gold, run 3 modes."""

    # Expanded gold for this (cat, attr) — non-null only
    exp_attr = expanded_gold[
        (expanded_gold["category"] == cat)
        & (expanded_gold["attr"] == attr)
        & ~expanded_gold["gold_is_null"]
    ].copy()
    exp_attr["code"] = exp_attr["code"].astype(str)
    exp_attr = exp_attr[exp_attr["code"].isin(code_to_idx)]

    if len(exp_attr) < 20:
        logger.info("[%s/%s] only %d non-null expanded gold cells, skipping",
                    cat, attr, len(exp_attr))
        return []

    # --- CONSISTENT 80/20 split on EXPANDED gold codes ---
    all_codes = exp_attr["code"].tolist()
    train_codes, test_codes = train_test_split(
        all_codes, test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )
    train_codes_set = set(train_codes)
    test_codes_set = set(test_codes)

    # Test set is SHARED across all 3 modes
    test_gold = exp_attr[exp_attr["code"].isin(test_codes_set)].copy()
    test_idx = np.array([code_to_idx[c] for c in test_gold["code"]])
    X_te = emb[test_idx]
    y_te = test_gold["gold_value"].astype(str).values
    n_test = len(test_codes)

    rows = []

    # --- Mode 1: silver_only (no retrain, existing silver model) ---
    sm, sle = _load_silver_xgb(cat, attr)
    silver_pred = _silver_predict(sm, sle, X_te)
    if silver_pred is not None:
        silver_acc = float(accuracy_score(y_te, silver_pred.astype(str)))
    else:
        silver_acc = float("nan")
    rows.append({
        "category": cat, "attr": attr, "mode": "silver_only",
        "accuracy": silver_acc, "n_test": n_test, "n_train_gold": 0,
    })
    logger.info("[%s/%s/silver_only] acc=%.3f n_test=%d", cat, attr, silver_acc, n_test)

    # --- Mode 2: hybrid_v1 (original 717 gold only, same test set) ---
    orig_attr = orig_gold[
        (orig_gold["category"] == cat)
        & (orig_gold["attr"] == attr)
        & ~orig_gold["gold_is_null"]
    ].copy()
    orig_attr["code"] = orig_attr["code"].astype(str)
    orig_attr = orig_attr[orig_attr["code"].isin(code_to_idx)]

    # Split original gold: use codes NOT in test_codes_set for training
    # (the test set was drawn from expanded gold, which is a superset of original)
    orig_train = orig_attr[~orig_attr["code"].isin(test_codes_set)]
    orig_train_codes_set = set(orig_train["code"])

    X_hybrid_v1, y_hybrid_v1, w_hybrid_v1 = _build_hybrid_train(
        cat, attr, silver, emb, code_to_idx, orig_gold,
        orig_train_codes_set, test_codes_set,
    )
    if len(X_hybrid_v1) > 0:
        n_orig_gold_train = int((w_hybrid_v1 == GOLD_WEIGHT).sum())
        acc_v1 = train_xgb_and_score(X_hybrid_v1, y_hybrid_v1, X_te, y_te,
                                     sample_weight=w_hybrid_v1)
    else:
        n_orig_gold_train = 0
        acc_v1 = float("nan")
    rows.append({
        "category": cat, "attr": attr, "mode": "hybrid_v1",
        "accuracy": acc_v1, "n_test": n_test, "n_train_gold": n_orig_gold_train,
    })
    logger.info("[%s/%s/hybrid_v1] acc=%.3f n_test=%d n_gold_train=%d",
                cat, attr, acc_v1, n_test, n_orig_gold_train)

    # --- Mode 3: hybrid_v2 (expanded gold, same test set) ---
    X_hybrid_v2, y_hybrid_v2, w_hybrid_v2 = _build_hybrid_train(
        cat, attr, silver, emb, code_to_idx, expanded_gold,
        train_codes_set, test_codes_set,
    )
    if len(X_hybrid_v2) > 0:
        n_exp_gold_train = int((w_hybrid_v2 == GOLD_WEIGHT).sum())
        acc_v2 = train_xgb_and_score(X_hybrid_v2, y_hybrid_v2, X_te, y_te,
                                     sample_weight=w_hybrid_v2)
    else:
        n_exp_gold_train = 0
        acc_v2 = float("nan")
    rows.append({
        "category": cat, "attr": attr, "mode": "hybrid_v2",
        "accuracy": acc_v2, "n_test": n_test, "n_train_gold": n_exp_gold_train,
    })
    logger.info("[%s/%s/hybrid_v2] acc=%.3f n_test=%d n_gold_train=%d",
                cat, attr, acc_v2, n_test, n_exp_gold_train)

    return rows


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Honest 80/20 eval: silver vs hybrid_v1 vs hybrid_v2")
    ap.add_argument("--expanded-gold", type=Path, default=EXPANDED_GOLD_PATH)
    ap.add_argument("--orig-gold", type=Path, default=ORIG_GOLD_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    expanded_gold = pd.read_parquet(args.expanded_gold)
    expanded_gold["code"] = expanded_gold["code"].astype(str)
    logger.info("Expanded gold: %d rows, %d unique codes",
                len(expanded_gold), expanded_gold["code"].nunique())

    orig_gold = pd.read_parquet(args.orig_gold)
    orig_gold["code"] = orig_gold["code"].astype(str)
    logger.info("Original v2 gold: %d rows, %d unique codes",
                len(orig_gold), orig_gold["code"].nunique())

    all_rows = []
    for cat in OFF_CATS:
        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        )
        silver["code"] = silver["code"].astype(str)
        emb = np.load(Path(PROCESSED_DIR) / f"{cat}_stratified_embeddings.npy")
        code_to_idx = {c: i for i, c in enumerate(silver["code"].tolist())}

        attrs = sorted(expanded_gold[expanded_gold["category"] == cat]["attr"].unique())
        logger.info("[%s] attrs: %s", cat, attrs)
        for attr in attrs:
            rows = run_one_attr(
                cat, attr, expanded_gold, orig_gold, silver, emb, code_to_idx
            )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_parquet(args.out, index=False)
    logger.info("Wrote %d rows to %s", len(df), args.out)

    # Per-cat summary table
    print("\n=== Summary: mean accuracy by mode per category ===")
    pivot = df.pivot_table(
        index="category", columns="mode", values="accuracy", aggfunc="mean"
    )
    if "hybrid_v1" in pivot.columns and "hybrid_v2" in pivot.columns:
        pivot["v2_minus_v1_pp"] = (pivot["hybrid_v2"] - pivot["hybrid_v1"]) * 100
    if "hybrid_v2" in pivot.columns and "silver_only" in pivot.columns:
        pivot["v2_minus_silver_pp"] = (pivot["hybrid_v2"] - pivot["silver_only"]) * 100
    print(pivot.round(4).to_string())

    print("\n=== Grand means ===")
    grand = df.groupby("mode")["accuracy"].mean()
    print(grand.round(4).to_string())

    if "hybrid_v1" in grand.index and "hybrid_v2" in grand.index:
        delta = (grand["hybrid_v2"] - grand["hybrid_v1"]) * 100
        print(f"\nhybrid_v2 − hybrid_v1 = {delta:+.2f} pp")
        verdict = "GAIN" if delta > 1.0 else ("REGRESSION" if delta < -1.0 else "FLAT")
        print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
