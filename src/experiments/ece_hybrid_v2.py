"""ECE for hybrid_v2 cascade — honest 80/20 split.

For each (cat, attr) in the expanded gold (v2):
  1. Split gold codes 80/20 (seed=42, same as eval_v2_expanded.py).
  2. Train hybrid_v2 XGB on full silver + 80% gold (gold_weight=5).
  3. Predict PROBABILITIES on 20% held-out gold.
  4. Compute binned ECE (10 bins) + reliability data.

Output:
  datasets/processed/ece_hybrid_v2.json
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.common import MODELS_DIR, PROCESSED_DIR, setup_logging
from src.pipeline.ml.train import compute_ece

logger = logging.getLogger(__name__)

OFF_CATS = ["pasta", "chocolate", "cheeses"]
RANDOM_STATE = 42
TEST_FRACTION = 0.2
GOLD_WEIGHT = 5.0

EXPANDED_GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet"
OUT_PATH = Path(PROCESSED_DIR) / "ece_hybrid_v2.json"


def train_xgb_and_predict_proba(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit XGB and return (top_proba, top_labels_int, true_labels_int).

    Returns arrays aligned to the test rows where y_te class is known to the model.
    top_proba: max-class confidence per prediction
    top_labels_int: argmax class index
    true_labels_int: ground-truth class index (matching remap)
    """
    train_classes = sorted(set(y_tr))
    if len(train_classes) < 2:
        # Degenerate case
        n = len(y_te)
        proba = np.zeros((n, max(2, len(train_classes))))
        proba[:, 0] = 1.0
        return proba.max(axis=1), proba.argmax(axis=1), np.zeros(n, dtype=int)

    remap = {c: i for i, c in enumerate(train_classes)}
    y_tr_r = np.array([remap[c] for c in y_tr])
    y_te_r = np.array([remap.get(c, -1) for c in y_te])

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
        clf = xgb.XGBClassifier(
            objective="multi:softprob", num_class=n_classes, **common_kwargs
        )

    fit_kwargs: dict = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    clf.fit(X_tr, y_tr_r, **fit_kwargs)

    # Filter test rows to those with known class labels
    valid_mask = y_te_r >= 0
    X_te_valid = X_te[valid_mask]
    y_te_valid = y_te_r[valid_mask]

    if len(X_te_valid) == 0:
        return np.array([]), np.array([]), np.array([])

    proba = clf.predict_proba(X_te_valid)  # shape (n_valid, n_classes)
    top_proba = proba.max(axis=1)
    top_labels = proba.argmax(axis=1)

    return top_proba, top_labels, y_te_valid


def _build_hybrid_train(
    cat: str,
    attr: str,
    silver: pd.DataFrame,
    emb: np.ndarray,
    code_to_idx: dict,
    expanded_gold: pd.DataFrame,
    train_codes_set: set,
    test_codes_set: set,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, weights) for hybrid_v2 training: silver + expanded gold subset."""
    if attr not in silver.columns:
        return np.array([]), np.array([]), np.array([])

    # Silver rows with non-null attr labels, excluding test codes and gold train codes
    silver_for_attr = silver[silver[attr].notna()].copy()
    silver_for_attr["code"] = silver_for_attr["code"].astype(str)
    silver_for_attr = silver_for_attr[~silver_for_attr["code"].isin(test_codes_set)]
    silver_for_attr = silver_for_attr[silver_for_attr["code"].isin(code_to_idx)]
    # Exclude codes that overlap with gold train (avoid conflict)
    silver_for_attr = silver_for_attr[~silver_for_attr["code"].isin(train_codes_set)]

    silver_idx = np.array([code_to_idx[c] for c in silver_for_attr["code"]])
    X_silver = emb[silver_idx]
    y_silver = silver_for_attr[attr].astype(str).values
    w_silver = np.ones(len(y_silver))

    # Gold train portion for this attr
    train_gold = expanded_gold[
        (expanded_gold["category"] == cat)
        & (expanded_gold["attr"] == attr)
        & ~expanded_gold["gold_is_null"]
        & expanded_gold["code"].isin(train_codes_set)
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
    silver: pd.DataFrame,
    emb: np.ndarray,
    code_to_idx: dict,
) -> dict | None:
    """For one (cat, attr): hybrid_v2 ECE on honest 20% hold-out."""

    # Expanded gold for this (cat, attr) — non-null only
    exp_attr = expanded_gold[
        (expanded_gold["category"] == cat)
        & (expanded_gold["attr"] == attr)
        & ~expanded_gold["gold_is_null"]
    ].copy()
    exp_attr["code"] = exp_attr["code"].astype(str)
    exp_attr = exp_attr[exp_attr["code"].isin(code_to_idx)]

    if len(exp_attr) < 20:
        logger.info("[%s/%s] only %d non-null gold cells, skipping", cat, attr, len(exp_attr))
        return None

    # CONSISTENT 80/20 split on expanded gold codes — same seed as eval_v2_expanded.py
    all_codes = exp_attr["code"].tolist()
    train_codes, test_codes = train_test_split(
        all_codes, test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )
    train_codes_set = set(train_codes)
    test_codes_set = set(test_codes)

    # Test set
    test_gold = exp_attr[exp_attr["code"].isin(test_codes_set)].copy()
    test_idx = np.array([code_to_idx[c] for c in test_gold["code"]])
    X_te = emb[test_idx]
    y_te_raw = test_gold["gold_value"].astype(str).values

    # Build hybrid_v2 training data
    X_hybrid, y_hybrid, w_hybrid = _build_hybrid_train(
        cat, attr, silver, emb, code_to_idx,
        expanded_gold, train_codes_set, test_codes_set,
    )

    if len(X_hybrid) == 0:
        logger.info("[%s/%s] no hybrid training data, skipping", cat, attr)
        return None

    top_proba, top_labels, y_te_enc = train_xgb_and_predict_proba(
        X_hybrid, y_hybrid, X_te, y_te_raw, sample_weight=w_hybrid
    )

    if len(top_proba) == 0:
        return None

    # Build proba matrix for compute_ece (it expects 2d matrix with argmax = top label)
    n = len(top_proba)
    n_classes = int(top_labels.max()) + 1 if len(top_labels) > 0 else 2
    n_classes = max(n_classes, int(y_te_enc.max()) + 1)
    proba_matrix = np.zeros((n, n_classes))
    for i in range(n):
        proba_matrix[i, top_labels[i]] = top_proba[i]
        # Fill remaining probability mass evenly (does not affect argmax/confidence)
        remainder = (1.0 - top_proba[i]) / max(n_classes - 1, 1)
        for j in range(n_classes):
            if j != top_labels[i]:
                proba_matrix[i, j] = remainder

    ece_val, _ = compute_ece(y_te_enc, proba_matrix, n_bins=10)

    accuracy = float((top_labels == y_te_enc).mean())
    mean_top_proba = float(top_proba.mean())
    calibration_gap = float(mean_top_proba - accuracy)

    result = {
        "n_test": int(len(y_te_enc)),
        "ece_10bins": round(float(ece_val), 4),
        "mean_top_proba": round(mean_top_proba, 4),
        "accuracy": round(accuracy, 4),
        "calibration_gap": round(calibration_gap, 4),
    }
    logger.info(
        "[%s/%s] n_test=%d ece=%.4f acc=%.4f mean_conf=%.4f gap=%.4f",
        cat, attr, result["n_test"], ece_val, accuracy, mean_top_proba, calibration_gap,
    )
    return result


def main() -> None:
    setup_logging()

    expanded_gold = pd.read_parquet(EXPANDED_GOLD_PATH)
    expanded_gold["code"] = expanded_gold["code"].astype(str)
    logger.info(
        "Expanded gold: %d rows, %d unique codes",
        len(expanded_gold), expanded_gold["code"].nunique(),
    )

    output: dict[str, dict] = {}

    for cat in OFF_CATS:
        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        )
        silver["code"] = silver["code"].astype(str)
        emb = np.load(Path(PROCESSED_DIR) / f"{cat}_stratified_embeddings.npy")
        code_to_idx = {c: i for i, c in enumerate(silver["code"].tolist())}

        attrs = sorted(
            expanded_gold[expanded_gold["category"] == cat]["attr"].unique()
        )
        logger.info("[%s] attrs: %s", cat, attrs)

        for attr in attrs:
            result = run_one_attr(cat, attr, expanded_gold, silver, emb, code_to_idx)
            if result is not None:
                output[f"{cat}_{attr}"] = result

    OUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info("Wrote %d entries to %s", len(output), OUT_PATH)

    # Per-cat mean ECE summary
    print("\n=== Per-cat mean ECE (hybrid_v2, 80/20 honest split) ===")
    cat_eces: dict[str, list[float]] = {c: [] for c in OFF_CATS}
    for key, val in output.items():
        for cat in OFF_CATS:
            if key.startswith(cat + "_"):
                cat_eces[cat].append(val["ece_10bins"])
                break
    for cat in OFF_CATS:
        eces = cat_eces[cat]
        if eces:
            print(f"  {cat}: mean_ece={np.mean(eces):.4f} over {len(eces)} attrs")

    all_eces = [v["ece_10bins"] for v in output.values()]
    print(f"\nGrand mean ECE: {np.mean(all_eces):.4f} over {len(all_eces)} (cat, attr) pairs")


if __name__ == "__main__":
    main()
