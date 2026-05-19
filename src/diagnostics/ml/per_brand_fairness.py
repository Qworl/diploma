"""
Per-brand fairness — accuracy на топ-N брендов.

Phase 13 балансирует test set по языкам, но не по брендам. Возможно отдельные бренды
драматически хуже распознаются (rare brands → ML undertrained, but common brands могут
быть mis-pinned в class prior).

Этот скрипт читает experiment_per_product_<cat>.parquet (where pipeline test predictions
already saved) и считает accuracy на каждый из топ-15 брендов.

Если worst brand имеет accuracy << overall — есть fairness проблема в production:
для этих брендов нужен LLM fallback или отдельная модель.

Usage:
    python -m src.diagnostics.ml.per_brand_fairness
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from src.common import PROCESSED_DIR, RANDOM_STATE, TEST_SIZE, setup_logging, wilson_ci

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

CATEGORIES = ["pasta", "chocolate", "beverages",
              "pasta_stratified", "chocolate_stratified", "beverages_stratified",
              "cheeses_stratified", "cereals_stratified", "cosmetics_stratified"]
TOP_N_BRANDS = 12
MIN_PRODUCTS_PER_BRAND = 5  # below threshold — Wilson CI слишком широкий


def load_test_with_brands(category: str) -> pd.DataFrame:
    """Returns test_df slice with code, brands, and aligned via train_test_split."""
    ss = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{category}_silver_standard.parquet"))
    _, test_idx = train_test_split(np.arange(len(ss)), test_size=TEST_SIZE, random_state=RANDOM_STATE)
    test = ss.iloc[test_idx][["code", "brands"]].copy()
    test["code"] = test["code"].astype(str)
    test["brand_norm"] = test["brands"].fillna("unknown").astype(str) \
        .str.split(",").str[0].str.strip().str.lower()
    return test


def per_brand_accuracy(category: str) -> pd.DataFrame:
    pp_path = os.path.join(PROCESSED_DIR, f"experiment_per_product_{category}.parquet")
    if not os.path.exists(pp_path):
        logger.warning("missing %s — run run_experiments first", pp_path)
        return pd.DataFrame()

    pp = pd.read_parquet(pp_path)
    pp = pp[(pp["layer"] != "none") & pp["gt"].notna() & pp["pred"].notna()].copy()
    pp = pp[(pp["gt"] != "None") & (pp["pred"] != "None")]
    pp["correct"] = (pp["gt"].astype(str) == pp["pred"].astype(str)).astype(int)

    test_df = load_test_with_brands(category)
    pp = pp.merge(test_df, on="code", how="left")

    # Top-N brands by product count
    top_brands = pp["brand_norm"].value_counts().head(TOP_N_BRANDS).index.tolist()

    rows = []
    overall_acc = pp["correct"].mean()
    overall_n = len(pp)
    overall_n_correct = int(pp["correct"].sum())
    overall_lo, overall_hi = wilson_ci(overall_n_correct, overall_n)
    rows.append({
        "category": category.replace("_stratified", ""), "brand": "(OVERALL)",
        "n_predictions": overall_n,
        "accuracy": float(overall_acc),
        "ci_lo": float(overall_lo),
        "ci_hi": float(overall_hi),
    })

    for brand in top_brands:
        sub = pp[pp["brand_norm"] == brand]
        n = len(sub)
        if n < MIN_PRODUCTS_PER_BRAND:
            continue
        acc = sub["correct"].mean()
        n_correct = int(sub["correct"].sum())
        lo, hi = wilson_ci(n_correct, n)
        rows.append({
            "category": category.replace("_stratified", ""), "brand": brand,
            "n_predictions": n,
            "accuracy": float(acc),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "delta_from_overall_pp": float((acc - overall_acc) * 100),
        })
    return pd.DataFrame(rows)


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--category", choices=CATEGORIES, default=None)
    args = p.parse_args()

    cats = [args.category] if args.category else CATEGORIES
    all_rows = []
    for cat in cats:
        df = per_brand_accuracy(cat)
        if df.empty:
            continue
        all_rows.append(df)
        logger.info("\n" + "=" * 78)
        logger.info("PER-BRAND FAIRNESS — %s (config=off_ml_bayes)", cat)
        logger.info("=" * 78)
        for _, r in df.iterrows():
            delta = r.get("delta_from_overall_pp", 0)
            marker = ""
            if r["brand"] != "(OVERALL)" and abs(delta) >= 5:
                marker = " ← WORSE" if delta < 0 else " ← BETTER"
            logger.info("  %-25s n=%4d  acc=%.3f  [%4.1f, %4.1f]%s",
                        r["brand"], r["n_predictions"], r["accuracy"],
                        r["ci_lo"]*100, r["ci_hi"]*100, marker)

    full = pd.concat(all_rows, ignore_index=True)
    out = os.path.join(PROCESSED_DIR, "per_brand_fairness.parquet")
    full.to_parquet(out, index=False)
    logger.info("\nSaved -> %s", out)


if __name__ == "__main__":
    main()
