"""GroupKFold(brand) × N seeds — honest CV stability for §6.6.

Replaces cv_stability.py (StratifiedKFold, leaks brand). Reuses
CATEGORIES_ATTRS from cv_stability for alignment with historical comparisons.

For each (cat, attr, seed, fold): fit XGB on train fold (cached embeddings),
predict on test fold, accuracy + F1. Aggregate mean ± std across (fold × seed).
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from src.common import PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)


# Per-cat attribute lists. Source of truth: existing cv_stability.CATEGORIES_ATTRS.
# Importing avoids drift; if it imports cleanly, use it. Otherwise replicate here.
try:
    from src.diagnostics.ml.cv_stability import CATEGORIES_ATTRS
except ImportError:
    CATEGORIES_ATTRS = {}


def iter_group_folds(groups: np.ndarray, *, n_splits: int = 5, seed: int = 0
                     ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold doesn't honor random seed natively. We permute groups for seed."""
    rng = np.random.default_rng(seed)
    unique_groups = np.array(sorted(set(groups)))
    perm = rng.permutation(len(unique_groups))
    remap = {g: i for i, g in enumerate(unique_groups[perm])}
    permuted = np.array([remap[g] for g in groups])
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(np.zeros((len(groups), 1)), groups=permuted)


def _fit_and_score(X_tr, y_tr, X_te, y_te) -> tuple[float, float] | None:
    """Fit XGB on one fold, return (accuracy, f1_macro). None if degenerate."""
    train_classes = sorted(np.unique(y_tr).tolist())
    if len(train_classes) < 2:
        return None
    remap = {c: i for i, c in enumerate(train_classes)}
    y_tr_r = np.array([remap[c] for c in y_tr])
    y_te_r = np.array([remap.get(c, -1) for c in y_te])

    n_classes = len(train_classes)
    if n_classes == 2:
        pos = int((y_tr_r == 1).sum())
        neg = int((y_tr_r == 0).sum())
        spw = max(neg / max(pos, 1), 0.5)
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
            scale_pos_weight=spw,
            tree_method="hist", verbosity=0,
        )
    else:
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
            objective="multi:softmax", num_class=n_classes,
            tree_method="hist", verbosity=0,
        )
    clf.fit(X_tr, y_tr_r)
    pred = clf.predict(X_te)
    valid = y_te_r >= 0  # unseen test classes → counted as wrong
    if not valid.any():
        return None
    acc = float(accuracy_score(y_te_r[valid], pred[valid]))
    f1 = float(f1_score(y_te_r[valid], pred[valid], average="macro", zero_division=0))
    return acc, f1


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--categories", nargs="+",
                   default=["pasta_stratified", "chocolate_stratified",
                            "beverages_stratified", "cheeses_stratified",
                            "cereals_stratified", "cosmetics_stratified"])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--out",
                   default=str(Path(PROCESSED_DIR) / "cv_stability_groupkfold.parquet"))
    args = p.parse_args()

    rows = []
    for cat in args.categories:
        attrs = CATEGORIES_ATTRS.get(cat) or CATEGORIES_ATTRS.get(cat.replace("_stratified", ""))
        if not attrs:
            logger.warning("No attrs for %s; skip", cat)
            continue
        silver_path = Path(PROCESSED_DIR) / f"{cat}_silver_standard.parquet"
        emb_path = Path(PROCESSED_DIR) / f"{cat}_embeddings.npy"
        if not silver_path.exists() or not emb_path.exists():
            logger.warning("Missing silver or embeddings for %s; skip", cat)
            continue
        silver = pd.read_parquet(silver_path)
        emb = np.load(emb_path)
        if len(silver) != len(emb):
            logger.warning("[%s] silver rows=%d vs emb rows=%d — skip", cat, len(silver), len(emb))
            continue
        brand_norm = silver["brands"].fillna("UNKNOWN").astype(str) \
                          .str.split(",").str[0].str.strip().str.lower().values
        for attr in attrs:
            if attr not in silver.columns:
                continue
            y = silver[attr].fillna("__NULL__").astype(str).values
            for seed in range(args.n_seeds):
                for fold_i, (tr, te) in enumerate(iter_group_folds(
                        brand_norm, n_splits=args.n_splits, seed=seed)):
                    res = _fit_and_score(emb[tr], y[tr], emb[te], y[te])
                    if res is None:
                        continue
                    acc, f1 = res
                    rows.append({"category": cat, "attr": attr,
                                 "seed": seed, "fold": fold_i,
                                 "n_train": int(len(tr)), "n_test": int(len(te)),
                                 "accuracy": acc, "f1_macro": f1})
            # Periodic save after each attr
            pd.DataFrame(rows).to_parquet(args.out, index=False)
            logger.info("[%s/%s] done (rows so far: %d)", cat, attr, len(rows))

    final = pd.DataFrame(rows)
    final.to_parquet(args.out, index=False)
    summary = final.groupby(["category", "attr"]).agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        n=("accuracy", "size"),
    ).reset_index()
    logger.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
