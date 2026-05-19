"""Regex layer ablation: compare regex+ml-hybrid vs ml-hybrid only.

For each cat in [pasta, chocolate, cheeses]:
1. Take v2 expanded gold (2666 codes), split 80/20 (seed=42, stratified by cat)
2. Predict with:
   - regex + ml-hybrid (current production)
   - ml-hybrid only (include_regex=False)
3. Compute per-(cat, attr) accuracy on held-out 20% non-null gold cells
4. Output: datasets/processed/regex_ablation.parquet
5. Print pivot table: cat × attr × (with_regex, no_regex, delta_pp)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import PROCESSED_DIR, setup_logging
from src.eval.cascade_predict import predict_cascade

logger = logging.getLogger(__name__)

CATEGORIES = ["pasta", "chocolate", "cheeses"]
SEED = 42
TEST_SIZE = 0.2


def _compute_accuracy(preds: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(cat, attr) accuracy on non-null gold cells."""
    gold = gold[~gold["gold_is_null"]].copy()
    gold["code"] = gold["code"].astype(str)
    preds = preds.copy()
    preds["code"] = preds["code"].astype(str)

    m = gold.merge(preds[["code", "attr", "predicted"]], on=["code", "attr"], how="left")
    m["correct"] = (m["predicted"].astype(object) == m["gold_value"].astype(object)).fillna(False)

    return (
        m.groupby(["category", "attr"])
        .agg(n_cells=("correct", "count"), n_correct=("correct", "sum"))
        .assign(accuracy=lambda x: x["n_correct"] / x["n_cells"])
        .reset_index()
    )


def main():
    setup_logging()
    gold = pd.read_parquet(Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet")
    gold["code"] = gold["code"].astype(str)

    all_rows = []

    for cat in CATEGORIES:
        logger.info("Processing category: %s", cat)
        cat_gold = gold[gold["category"] == cat].copy()
        unique_codes = sorted(cat_gold["code"].unique().tolist())

        # 80/20 split by code (seed=42)
        train_codes, test_codes = train_test_split(unique_codes, test_size=TEST_SIZE, random_state=SEED)
        test_codes_set = set(test_codes)
        logger.info("  %s: %d train codes, %d test codes", cat, len(train_codes), len(test_codes))

        test_gold = cat_gold[cat_gold["code"].isin(test_codes_set)].copy()

        # Load product data for those test codes from silver standard
        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        )
        silver["code"] = silver["code"].astype(str)
        products = silver[silver["code"].isin(test_codes_set)].copy()
        if len(products) == 0:
            logger.warning("  No products found for %s test codes in silver standard", cat)
            continue

        logger.info("  Running with_regex (regex + ml-hybrid)...")
        preds_with_regex = predict_cascade(
            products, category=f"{cat}_stratified",
            use_hybrid=True, include_regex=True
        )
        preds_with_regex["category"] = cat

        logger.info("  Running no_regex (ml-hybrid only)...")
        preds_no_regex = predict_cascade(
            products, category=f"{cat}_stratified",
            use_hybrid=True, include_regex=False
        )
        preds_no_regex["category"] = cat

        acc_with = _compute_accuracy(preds_with_regex, test_gold)
        acc_with = acc_with.rename(columns={"n_cells": "n_cells_regex", "n_correct": "n_correct_regex", "accuracy": "acc_with_regex"})

        acc_no = _compute_accuracy(preds_no_regex, test_gold)
        acc_no = acc_no.rename(columns={"n_cells": "n_cells_no", "n_correct": "n_correct_no", "accuracy": "acc_no_regex"})

        merged = acc_with.merge(acc_no[["category", "attr", "acc_no_regex"]], on=["category", "attr"], how="outer")
        merged["delta_pp"] = (merged["acc_with_regex"] - merged["acc_no_regex"]) * 100
        all_rows.append(merged)

    if not all_rows:
        logger.error("No results produced!")
        return

    result = pd.concat(all_rows, ignore_index=True)
    out_path = Path(PROCESSED_DIR) / "regex_ablation.parquet"
    result.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(result))

    # Print pivot table
    print("\n=== REGEX ABLATION RESULTS ===")
    print("Positive delta_pp = regex helps; Negative = regex hurts\n")

    pivot = result[["category", "attr", "acc_with_regex", "acc_no_regex", "delta_pp"]].copy()
    pivot["acc_with_regex"] = (pivot["acc_with_regex"] * 100).round(1)
    pivot["acc_no_regex"] = (pivot["acc_no_regex"] * 100).round(1)
    pivot["delta_pp"] = pivot["delta_pp"].round(2)
    pivot = pivot.sort_values(["category", "attr"])
    print(pivot.to_string(index=False))

    # Summary stats
    print("\n=== SUMMARY ===")
    print(f"Attrs where regex helps (delta_pp > 0): {(result['delta_pp'] > 0).sum()}")
    print(f"Attrs where regex hurts (delta_pp < 0): {(result['delta_pp'] < 0).sum()}")
    print(f"Attrs with no change (delta_pp == 0): {(result['delta_pp'] == 0).sum()}")
    print(f"Mean delta_pp: {result['delta_pp'].mean():.2f}")
    print(f"Max delta_pp (regex most helpful): {result['delta_pp'].max():.2f} @ {result.loc[result['delta_pp'].idxmax(), 'attr']}")
    print(f"Min delta_pp (regex most harmful): {result['delta_pp'].min():.2f} @ {result.loc[result['delta_pp'].idxmin(), 'attr']}")


if __name__ == "__main__":
    main()
