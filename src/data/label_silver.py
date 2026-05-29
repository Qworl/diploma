"""
OFF-tag-only silver standard labeler (no LLM).

Аналог `label_with_llm.py`, но использует ТОЛЬКО `apply_off_labels()` —
никаких вызовов LLM. Это alignment с Phase 14 ground-truth-strategy
(Option B): silver standard = crowd-sourced consensus weak supervision
из OFF тегов и числовых полей, без LLM-noise floor.

Пропускает товары, где после `apply_off_labels` остаётся 0 заполненных
schema-атрибутов. Все остальные сохраняются с None-ами на пустых
атрибутах — downstream код (train/eval) уже умеет с NaN работать.

Usage:
    python src/eval/label_with_off_only.py --category pasta_stratified
    python src/eval/label_with_off_only.py --category chocolate_stratified
    python src/eval/label_with_off_only.py --category beverages_stratified
"""

import argparse
import logging
import os

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.pipeline.schemas import (
    PASTA_SCHEMA, CHOCOLATE_SCHEMA, BEVERAGE_SCHEMA,
    COSMETICS_SCHEMA, CHEESES_SCHEMA, CEREALS_SCHEMA,
)
from src.pipeline.off_labels import apply_off_labels

# Baby is not yet in the reorganized schemas; will be in T16
# For now, support only the 6 domains that have been migrated
try:
    from src.pipeline.schemas import BABY_SCHEMA
except ImportError:
    BABY_SCHEMA = None

logger = logging.getLogger(__name__)

CATEGORY_CONFIG = {
    "pasta_stratified": {
        "parquet": "pasta_stratified_raw.parquet",
        "schema": PASTA_SCHEMA,
        "output": "pasta_stratified_silver_standard.parquet",
    },
    "chocolate_stratified": {
        "parquet": "chocolate_stratified_raw.parquet",
        "schema": CHOCOLATE_SCHEMA,
        "output": "chocolate_stratified_silver_standard.parquet",
    },
    "beverages_stratified": {
        "parquet": "beverages_stratified_raw.parquet",
        "schema": BEVERAGE_SCHEMA,
        "output": "beverages_stratified_silver_standard.parquet",
    },
    "cosmetics_stratified": {
        "parquet": "cosmetics_stratified_raw.parquet",
        "schema": COSMETICS_SCHEMA,
        "output": "cosmetics_stratified_silver_standard.parquet",
    },
    "cheeses_stratified": {
        "parquet": "cheeses_stratified_raw.parquet",
        "schema": CHEESES_SCHEMA,
        "output": "cheeses_stratified_silver_standard.parquet",
    },
    "cereals_stratified": {
        "parquet": "cereals_stratified_raw.parquet",
        "schema": CEREALS_SCHEMA,
        "output": "cereals_stratified_silver_standard.parquet",
    },
}

# Add baby_stratified only if BABY_SCHEMA is available
if BABY_SCHEMA is not None:
    CATEGORY_CONFIG["baby_stratified"] = {
        "parquet": "baby_stratified_raw.parquet",
        "schema": BABY_SCHEMA,
        "output": "baby_stratified_silver_standard.parquet",
    }


def label_off_only(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Apply apply_off_labels per row, return DataFrame with code + schema attrs."""
    rows = []
    for _, row in df.iterrows():
        off = apply_off_labels(row.to_dict(), schema)
        off["code"] = row["code"]
        # ensure all schema attrs present (None if not derived)
        for attr in schema:
            off.setdefault(attr, None)
        rows.append(off)
    return pd.DataFrame(rows)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="OFF-tag-only silver labeling (no LLM)")
    parser.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()))
    parser.add_argument("--backup-suffix", default="_haiku",
                        help="Suffix for backing up existing silver standard")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip backup of existing silver standard")
    args = parser.parse_args()

    config = CATEGORY_CONFIG[args.category]
    input_path = os.path.join(PROCESSED_DIR, config["parquet"])
    output_path = os.path.join(PROCESSED_DIR, config["output"])

    df = pd.read_parquet(input_path)
    df = df[df["product_name"].notna()].copy()
    logger.info("Loaded %d products with product_name from %s", len(df), config["parquet"])

    # Backup existing silver standard
    if os.path.exists(output_path) and not args.no_backup:
        backup_path = output_path.replace(
            ".parquet", f"{args.backup_suffix}.parquet"
        )
        if not os.path.exists(backup_path):
            os.rename(output_path, backup_path)
            logger.info("Backed up existing %s → %s",
                        os.path.basename(output_path), os.path.basename(backup_path))
        else:
            logger.info("Backup already exists at %s, skipping rename", backup_path)
            os.remove(output_path)

    schema = config["schema"]
    logger.info("Labeling %d products with OFF-tag-only labeler...", len(df))
    result_df = label_off_only(df, schema)

    keep_cols = ["code", "product_name", "brands", "categories_tags",
                 "countries_tags", "labels_tags", "ingredients_text",
                 "ingredients_analysis_tags", "traces_tags",
                 "quantity",
                 "fat_100g", "sugars_100g", "proteins_100g", "carbohydrates_100g",
                 "alcohol_100g", "nutriscore_grade", "nova_group"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    merged = df[keep_cols].merge(result_df, on="code", how="inner")

    merged.to_parquet(output_path, index=False)
    logger.info("Saved %d labeled products to %s", len(merged), output_path)

    # Coverage report
    logger.info("=== Per-attribute coverage ===")
    for attr in schema:
        if attr in merged.columns:
            n_any = merged[attr].notna().sum()
            pct = n_any / len(merged) * 100 if len(merged) else 0
            tier = "A" if pct >= 80 else ("B" if pct >= 40 else "C")
            logger.info("  [%s] %-22s %4d/%d (%5.1f%%)",
                        tier, attr, n_any, len(merged), pct)

    # Value distribution
    logger.info("=== Label distribution ===")
    for attr in schema:
        if attr in merged.columns:
            dist = merged[attr].value_counts(dropna=False).head(8).to_string()
            logger.info("%s:\n%s", attr, dist)


if __name__ == "__main__":
    main()
