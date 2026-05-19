"""Brand-clustered bootstrap CI for hybrid cascade accuracy.

Per spec §4.6: cluster bootstrap by brand (samples whole brands at each bootstrap draw).
This produces wider CIs than naive Wilson when data is brand-clustered (same brand
products are correlated).

Functions:
  brand_clustered_ci(correct, brands, *, n_boot=5000, seed=42, alpha=0.05) -> (lo, hi)

For each (cat, attr) in OFF cats: compute CI on hybrid cascade predictions vs v2 gold.
Output: datasets/processed/bootstrap_ci_brand_clustered.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.common import PROCESSED_DIR, setup_logging
from src.eval.cascade_predict import predict_cascade
from src.manual_label.schemas_loader import load_domain_attrs

logger = logging.getLogger(__name__)

OFF_CATS = ["pasta", "chocolate", "cheeses"]


def brand_clustered_ci(
    correct: np.ndarray,
    brands: np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Cluster bootstrap confidence interval for mean accuracy.

    At each bootstrap draw, resample whole brands (with replacement), then
    collect all observations from those brands. Compute mean accuracy per draw.
    Returns (lo, hi) percentile CI at level (1-alpha).

    The resulting CI is wider than naive Wilson when observations within the same
    brand are correlated (i.e., one brand's products tend to be all correct or
    all wrong).
    """
    unique_brands = np.unique(brands)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        # Resample brands with replacement
        sampled_brands = rng.choice(unique_brands, size=len(unique_brands), replace=True)
        # Collect all observations from those brands
        indices = np.concatenate([np.where(brands == br)[0] for br in sampled_brands])
        if len(indices) == 0:
            boot_means[b] = float("nan")
        else:
            boot_means[b] = correct[indices].mean()

    lo = float(np.nanpercentile(boot_means, 100 * alpha / 2))
    hi = float(np.nanpercentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def wilson_ci(n_correct: int, n_total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval."""
    if n_total == 0:
        return float("nan"), float("nan")
    p_hat = n_correct / n_total
    z = norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
    return float(center - margin), float(center + margin)


def _get_primary_brand(brands_str: str) -> str:
    return str(brands_str).split(",")[0].strip()


def main():
    setup_logging()

    gold = pd.read_parquet(Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet")
    gold["code"] = gold["code"].astype(str)

    results = []

    for cat in OFF_CATS:
        logger.info("Processing category: %s", cat)
        cat_gold = gold[(gold["category"] == cat) & (~gold["gold_is_null"])].copy()

        # Load product data from silver standard (for features + brands)
        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        )
        silver["code"] = silver["code"].astype(str)
        products = silver[silver["code"].isin(cat_gold["code"].unique())].copy()

        if len(products) == 0:
            logger.warning("No products for %s", cat)
            continue

        # Run hybrid cascade
        logger.info("  Running hybrid cascade for %s (%d products)", cat, len(products))
        preds = predict_cascade(products, category=f"{cat}_stratified", use_hybrid=True)
        preds["code"] = preds["code"].astype(str)

        # Merge cascade predictions with gold
        merged = cat_gold.merge(
            preds[["code", "attr", "predicted"]].rename(columns={"predicted": "cascade_pred"}),
            on=["code", "attr"],
            how="left",
        )
        merged = merged.merge(
            products[["code", "brands"]],
            on="code",
            how="left",
        )
        merged["correct"] = (
            merged["cascade_pred"].astype(object) == merged["gold_value"].astype(object)
        ).fillna(False).astype(int)
        merged["brand_primary"] = merged["brands"].fillna("").apply(_get_primary_brand)

        attrs = list(load_domain_attrs(cat))
        for attr in attrs:
            attr_data = merged[merged["attr"] == attr].copy()
            if len(attr_data) < 10:
                logger.warning("  Skipping %s/%s: only %d non-null gold cells", cat, attr, len(attr_data))
                continue

            correct_arr = attr_data["correct"].values
            brand_arr = attr_data["brand_primary"].values

            n_total = len(attr_data)
            n_correct = int(correct_arr.sum())
            mean_acc = float(correct_arr.mean())

            # Brand-clustered bootstrap CI
            n_brands = len(np.unique(brand_arr))
            cl_lo, cl_hi = brand_clustered_ci(
                correct_arr, brand_arr, n_boot=5000, seed=42, alpha=0.05
            )

            # Naive Wilson CI for comparison
            w_lo, w_hi = wilson_ci(n_correct, n_total)

            results.append({
                "category": cat,
                "attr": attr,
                "n_cells": n_total,
                "n_brands": n_brands,
                "mean_accuracy": round(mean_acc, 4),
                "clustered_ci_lo": round(cl_lo, 4),
                "clustered_ci_hi": round(cl_hi, 4),
                "clustered_ci_width": round(cl_hi - cl_lo, 4),
                "wilson_ci_lo": round(w_lo, 4),
                "wilson_ci_hi": round(w_hi, 4),
                "wilson_ci_width": round(w_hi - w_lo, 4),
                "ci_inflation_factor": round((cl_hi - cl_lo) / (w_hi - w_lo), 3) if (w_hi - w_lo) > 0 else None,
            })

    df_results = pd.DataFrame(results)
    out_path = Path(PROCESSED_DIR) / "bootstrap_ci_brand_clustered.parquet"
    df_results.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(df_results))

    # Print summary
    print("\n=== BRAND-CLUSTERED BOOTSTRAP CI ===")
    print(df_results[["category", "attr", "n_cells", "n_brands", "mean_accuracy",
                       "clustered_ci_lo", "clustered_ci_hi", "wilson_ci_lo", "wilson_ci_hi",
                       "ci_inflation_factor"]].to_string(index=False))

    print("\n=== CI INFLATION (clustered wider than Wilson) ===")
    inflated = df_results[df_results["ci_inflation_factor"] > 1].sort_values("ci_inflation_factor", ascending=False)
    print(inflated[["category", "attr", "n_brands", "ci_inflation_factor"]].to_string(index=False))
    print(f"\nMean inflation factor: {df_results['ci_inflation_factor'].mean():.3f}")
    print(f"Max inflation factor: {df_results['ci_inflation_factor'].max():.3f} @ {df_results.loc[df_results['ci_inflation_factor'].idxmax(), 'attr']}")


if __name__ == "__main__":
    main()
