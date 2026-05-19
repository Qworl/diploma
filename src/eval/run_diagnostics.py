"""
Запускает все ML-диагностики на стратифицированном серебре.

Перезаписывает в datasets/processed/:
- feature_ablation.parquet
- cv_stability_5fold.parquet
- stratified_split_ablation.parquet
- per_brand_fairness.parquet

Категория в выходных файлах нормализована (без суффикса `_stratified`),
чтобы notebooks/00_thesis_main.ipynb видел её как `pasta`/`chocolate`/`beverages`.

Usage:
    python -m src.eval.run_diagnostics
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

from src.common import PROCESSED_DIR, RANDOM_STATE, setup_logging
from src.diagnostics.ml import (
    cv_stability,
    feature_ablation,
    per_brand_fairness,
    stratified_split_ablation,
)

STRAT_CATEGORIES = ["pasta_stratified", "chocolate_stratified", "beverages_stratified",
                    "cheeses_stratified", "cereals_stratified", "cosmetics_stratified"]


def run_feature_ablation():
    print("\n=== feature_ablation (stratified) ===")
    parts = []
    for cat in STRAT_CATEGORIES:
        attrs = feature_ablation.CATEGORIES_ATTRS[cat]
        parts.append(feature_ablation.run_ablation(cat, attrs))
    out = pd.concat(parts, ignore_index=True)
    path = os.path.join(PROCESSED_DIR, "feature_ablation.parquet")
    out.to_parquet(path, index=False)
    print(f"  -> {path} ({len(out)} rows)")


def run_cv_stability(n_splits: int = 5):
    print(f"\n=== cv_stability {n_splits}-fold (stratified) ===")
    parts = []
    for cat in STRAT_CATEGORIES:
        parts.append(cv_stability.run(cat, cv_stability.CATEGORIES_ATTRS[cat],
                                       n_splits=n_splits, seed=RANDOM_STATE))
    out = pd.concat(parts, ignore_index=True)
    path = os.path.join(PROCESSED_DIR, f"cv_stability_{n_splits}fold.parquet")
    out.to_parquet(path, index=False)
    print(f"  -> {path} ({len(out)} rows)")


def run_split_ablation():
    print("\n=== stratified_split_ablation (stratified) ===")
    parts = []
    for cat in STRAT_CATEGORIES:
        parts.append(stratified_split_ablation.run(cat,
                                                    stratified_split_ablation.CATEGORIES_ATTRS[cat]))
    out = pd.concat(parts, ignore_index=True)
    path = os.path.join(PROCESSED_DIR, "stratified_split_ablation.parquet")
    out.to_parquet(path, index=False)
    print(f"  -> {path} ({len(out)} rows)")


def run_brand_fairness():
    print("\n=== per_brand_fairness (stratified) ===")
    parts = []
    for cat in STRAT_CATEGORIES:
        df = per_brand_fairness.per_brand_accuracy(cat)
        if not df.empty:
            parts.append(df)
    if not parts:
        print("  no data — нужны experiment_per_product_*_stratified.parquet")
        return
    out = pd.concat(parts, ignore_index=True)
    path = os.path.join(PROCESSED_DIR, "per_brand_fairness.parquet")
    out.to_parquet(path, index=False)
    print(f"  -> {path} ({len(out)} rows)")


if __name__ == "__main__":
    setup_logging()
    run_feature_ablation()
    run_cv_stability()
    run_split_ablation()
    run_brand_fairness()
    print("\nDone.")
