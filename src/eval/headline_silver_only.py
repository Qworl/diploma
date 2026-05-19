"""Headline cascade accuracy on silver labels for brand-disjoint test split.

Used for the 3 categories (beverages, cereals, cosmetics) that do not have
OFF-grounded v2 gold. Ground truth is the silver_value (from OFF tags).
Cells where silver_value is null are excluded from the denominator.

Output: one row per (category, attr) with accuracy + Wilson 95% CI.
"""
from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging, wilson_ci
from src.eval.cascade_predict import predict_cascade

logger = logging.getLogger(__name__)

SILVER_ONLY_CATS = ["beverages", "cereals", "cosmetics"]

# Map domain to (module_path, schema_attr_name) — mirrors schemas_loader but
# uses the actual attribute names (beverages schema is BEVERAGE_SCHEMA not BEVERAGES_SCHEMA).
_SCHEMA_MAP: dict[str, tuple[str, str]] = {
    "beverages": ("src.pipeline.schemas.beverages", "BEVERAGE_SCHEMA"),
    "cereals": ("src.pipeline.schemas.cereals", "CEREALS_SCHEMA"),
    "cosmetics": ("src.pipeline.schemas.cosmetics", "COSMETICS_SCHEMA"),
}


def _get_domain_attrs(domain: str) -> list[str]:
    """Return list of target attribute names for a domain."""
    mod_path, attr_name = _SCHEMA_MAP[domain]
    mod = importlib.import_module(mod_path)
    schema = getattr(mod, attr_name)
    return list(schema.keys())


def compute_silver_headline(
    silver: pd.DataFrame,
    cascade: pd.DataFrame,
    *,
    category: str,
) -> pd.DataFrame:
    """Compute per-(category, attr) silver-based accuracy.

    Parameters
    ----------
    silver:
        Long-format DataFrame with columns: code, attr, silver_value.
    cascade:
        Long-format predictions with columns: code, attr, predicted, layer.
    category:
        Category name used to tag output rows.

    Returns
    -------
    DataFrame with one row per attr.
    """
    silver = silver.copy()
    silver["code"] = silver["code"].astype(str)
    cascade = cascade.copy()
    cascade["code"] = cascade["code"].astype(str)

    merged = silver.merge(
        cascade[["code", "attr", "predicted", "layer"]],
        on=["code", "attr"],
        how="left",
    )

    # Non-null silver rows only
    nonnull = merged[~merged["silver_value"].isna()].copy()
    # Normalise both sides to str so bool silver values ("False"/"True") match
    # cascade string predictions. Abstained cascade rows keep None → miss.
    nonnull["correct"] = (
        nonnull["predicted"].map(lambda x: str(x) if x is not None else None)
        == nonnull["silver_value"].map(lambda x: str(x) if x is not None else None)
    ).astype(int)
    nonnull.loc[nonnull["predicted"].isna(), "correct"] = 0

    rows = []
    for attr, g in merged.groupby("attr"):
        sub = nonnull[nonnull["attr"] == attr]
        n_non_null = len(sub)
        n_correct = int(sub["correct"].sum())
        n_total = len(g)
        n_silver_null = n_total - n_non_null
        coverage = n_non_null / n_total if n_total else float("nan")
        acc = n_correct / n_non_null if n_non_null else float("nan")
        if n_non_null:
            lo, hi = wilson_ci(n_correct, n_non_null)
        else:
            lo, hi = float("nan"), float("nan")
        rows.append({
            "category": category,
            "attr": attr,
            "n_total_cells": n_total,
            "n_non_null_silver": n_non_null,
            "n_silver_null": n_silver_null,
            "silver_coverage_rate": coverage,
            "n_correct": n_correct,
            "accuracy": acc,
            "wilson_lower": lo,
            "wilson_upper": hi,
            "eval_set": "silver_brand_disjoint_test",
        })
    return pd.DataFrame(rows)


def main():
    setup_logging()
    p = argparse.ArgumentParser(
        description="Silver-only headline accuracy for beverages/cereals/cosmetics."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(PROCESSED_DIR) / "headline_results_silver_only.parquet",
    )
    p.add_argument("--cats", nargs="+", default=SILVER_ONLY_CATS)
    args = p.parse_args()

    all_results = []
    for cat in args.cats:
        # Load silver standard (wide format)
        silver_path = Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        silver_wide = pd.read_parquet(silver_path)
        silver_wide["code"] = silver_wide["code"].astype(str)

        # Filter to brand-disjoint test split
        split_path = Path(PROCESSED_DIR) / f"{cat}_gold_split.parquet"
        split = pd.read_parquet(split_path)
        test_codes = set(split.loc[split["split"] == "test", "code"].astype(str))
        silver_test = silver_wide[silver_wide["code"].isin(test_codes)].copy()

        n_codes = len(silver_test)
        logger.info("[%s] test split: %d codes", cat, n_codes)
        if n_codes == 0:
            logger.warning("[%s] No test codes found — skipping.", cat)
            continue

        # Melt wide → long on target attrs
        attrs = _get_domain_attrs(cat)
        available_attrs = [a for a in attrs if a in silver_test.columns]
        if len(available_attrs) < len(attrs):
            missing = set(attrs) - set(available_attrs)
            logger.warning("[%s] attrs missing from silver: %s", cat, missing)

        silver_long = silver_test[["code", *available_attrs]].melt(
            id_vars="code", var_name="attr", value_name="silver_value"
        )

        # Run cascade on test products
        feature_cols = [
            "code", "brands", "product_name", "ingredients_text",
            "quantity", "categories_tags",
        ]
        products = silver_test[[c for c in feature_cols if c in silver_test.columns]].copy()
        cascade = predict_cascade(products, category=f"{cat}_stratified")

        result = compute_silver_headline(silver_long, cascade, category=cat)
        mean_acc = result["accuracy"].mean()
        logger.info(
            "[%s] %d (cat, attr) rows; n=%d codes; mean accuracy=%.3f",
            cat, len(result), n_codes, mean_acc,
        )
        all_results.append(result)

    if not all_results:
        logger.error("No results — nothing written.")
        return

    final = pd.concat(all_results, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(args.out, index=False)
    logger.info("Wrote %s (%d rows)", args.out, len(final))


if __name__ == "__main__":
    main()
