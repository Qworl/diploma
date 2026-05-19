"""Per-brand fairness diagnostics with Wilson CI on hybrid v2 gold.

For each category, for top brands (>= 10 non-null gold cells):
  - Compute accuracy of hybrid cascade on v2 expanded gold
  - Compute Wilson 95% CI for each brand
  - Red-flag brand if CI does NOT overlap overall accuracy
    (i.e., brand is significantly better or worse than overall)

Output: datasets/processed/per_brand_fairness_v2.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.common import PROCESSED_DIR, setup_logging
from src.eval.cascade_predict import predict_cascade

logger = logging.getLogger(__name__)

OFF_CATS = ["pasta", "chocolate", "cheeses"]
MIN_CELLS_PER_BRAND = 10


def wilson_ci(n_correct: int, n_total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n_total == 0:
        return float("nan"), float("nan")
    p_hat = n_correct / n_total
    z = norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
    return float(center - margin), float(center + margin)


def _get_primary_brand(brands_str) -> str:
    return str(brands_str).split(",")[0].strip()


def main():
    setup_logging()

    gold = pd.read_parquet(Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet")
    gold["code"] = gold["code"].astype(str)

    all_results = []

    for cat in OFF_CATS:
        logger.info("Processing category: %s", cat)
        cat_gold = gold[(gold["category"] == cat) & (~gold["gold_is_null"])].copy()

        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        )
        silver["code"] = silver["code"].astype(str)
        products = silver[silver["code"].isin(cat_gold["code"].unique())].copy()

        if len(products) == 0:
            logger.warning("No products found for %s", cat)
            continue

        # Add brand info to gold
        products["brand_primary"] = products["brands"].fillna("").apply(_get_primary_brand)
        product_brands = products[["code", "brand_primary"]].drop_duplicates()

        # Run hybrid cascade predictions
        logger.info("  Running hybrid cascade for %s (%d products)", cat, len(products))
        preds = predict_cascade(products, category=f"{cat}_stratified", use_hybrid=True)
        preds["code"] = preds["code"].astype(str)

        # Merge all together
        merged = cat_gold.merge(
            preds[["code", "attr", "predicted"]].rename(columns={"predicted": "cascade_pred"}),
            on=["code", "attr"],
            how="left",
        )
        merged = merged.merge(product_brands, on="code", how="left")
        merged["correct"] = (
            merged["cascade_pred"].astype(object) == merged["gold_value"].astype(object)
        ).fillna(False).astype(int)

        # Overall accuracy for this category
        overall_acc = float(merged["correct"].mean())
        overall_n = len(merged)
        overall_lo, overall_hi = wilson_ci(int(merged["correct"].sum()), overall_n)
        logger.info("  %s overall accuracy: %.3f [%.3f, %.3f] (n=%d)", cat, overall_acc, overall_lo, overall_hi, overall_n)

        # Per-brand stats
        brand_groups = merged.groupby("brand_primary")
        for brand, g in brand_groups:
            n = len(g)
            if n < MIN_CELLS_PER_BRAND:
                continue
            n_correct = int(g["correct"].sum())
            brand_acc = float(g["correct"].mean())
            b_lo, b_hi = wilson_ci(n_correct, n)

            # Red flag: brand CI does not overlap overall accuracy
            # Brand CI [b_lo, b_hi] overlaps overall acc if b_lo <= overall_acc <= b_hi
            overlaps_overall = (b_lo <= overall_acc <= b_hi)
            red_flag = not overlaps_overall

            all_results.append({
                "category": cat,
                "brand": brand,
                "n_cells": n,
                "n_correct": n_correct,
                "brand_accuracy": round(brand_acc, 4),
                "wilson_ci_lo": round(b_lo, 4),
                "wilson_ci_hi": round(b_hi, 4),
                "overall_accuracy": round(overall_acc, 4),
                "overall_ci_lo": round(overall_lo, 4),
                "overall_ci_hi": round(overall_hi, 4),
                "ci_overlaps_overall": overlaps_overall,
                "red_flag": red_flag,
            })

    df_results = pd.DataFrame(all_results)

    if len(df_results) == 0:
        logger.error("No results!")
        return

    out_path = Path(PROCESSED_DIR) / "per_brand_fairness_v2.parquet"
    df_results.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d brands evaluated)", out_path, len(df_results))

    # Print summary
    print("\n=== PER-BRAND FAIRNESS (Wilson CI, top brands >=10 cells) ===")
    red_flagged = df_results[df_results["red_flag"]]
    print(f"\nTotal brands evaluated: {len(df_results)}")
    print(f"Red-flagged brands (CI not overlapping overall): {len(red_flagged)}")

    if len(red_flagged) > 0:
        print("\n--- Red-flagged brands ---")
        print(red_flagged[["category", "brand", "n_cells", "brand_accuracy",
                             "wilson_ci_lo", "wilson_ci_hi", "overall_accuracy"]
                           ].sort_values(["category", "brand_accuracy"]).to_string(index=False))
    else:
        print("No brands red-flagged.")

    print("\n=== SUMMARY BY CATEGORY ===")
    for cat in OFF_CATS:
        cat_df = df_results[df_results["category"] == cat]
        if len(cat_df) == 0:
            continue
        n_flagged = cat_df["red_flag"].sum()
        overall = cat_df["overall_accuracy"].iloc[0] if len(cat_df) > 0 else float("nan")
        print(f"  {cat}: {len(cat_df)} brands evaluated, {n_flagged} red-flagged, overall_acc={overall:.3f}")

    # Show worst brands per category
    print("\n=== WORST BRANDS (lowest accuracy, >=10 cells) ===")
    worst = df_results.nsmallest(10, "brand_accuracy")
    print(worst[["category", "brand", "n_cells", "brand_accuracy", "wilson_ci_lo", "wilson_ci_hi", "red_flag"]].to_string(index=False))


if __name__ == "__main__":
    main()
