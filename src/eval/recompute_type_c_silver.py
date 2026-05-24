"""Recompute TYPE_C numeric-bucket attributes from raw nutriments.

silver_standard.parquet columns for TYPE_C attrs (fat_class, protein_class)
were computed with **old** bucket thresholds in an earlier run.
`rules.py:TYPE_C_RULES` has since been updated (cheeses fat_class: 15/25/32
→ 15/20/28 per Opus audit) and the saved columns are now stale.

This script applies current `_type_c_numeric` to raw nutriment columns
(`fat_100g`, `proteins_100g`) and writes a fresh long-format parquet
suitable for ingest into `consolidate_llm_gold.py` as a high-priority source.

Cocoa_percentage (regex-based) is also recomputed from product_name for completeness.

Output: datasets/processed/{cat}_silver_type_c_fresh.parquet
Columns: code, attr, value, source ('silver_type_c_fresh'), priority
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.common import MAIN_CATEGORIES, PROCESSED_DIR
from src.pipeline.off_labels.rules import _type_c_numeric

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

PROCESSED = Path(PROCESSED_DIR)

TYPE_C_ATTRS_PER_CAT = {
    "pasta": ["protein_class"],
    "chocolate": ["protein_class", "cocoa_percentage"],
    "cheeses": ["fat_class"],
}

FRESH_PRIORITY = 100


def recompute_cat(cat: str) -> pd.DataFrame:
    silver = pd.read_parquet(PROCESSED / f"{cat}_stratified_silver_standard.parquet")
    silver["code"] = silver["code"].astype(str)
    attrs = TYPE_C_ATTRS_PER_CAT.get(cat, [])
    rows = []
    for attr in attrs:
        n_computed = 0
        n_total = 0
        for _, row in silver.iterrows():
            n_total += 1
            value = _type_c_numeric(row.to_dict(), attr)
            if value is None:
                continue
            rows.append({
                "code": row["code"], "attr": attr, "value": str(value).lower().strip(),
                "source": "silver_type_c_fresh", "priority": FRESH_PRIORITY,
            })
            n_computed += 1
        logger.info("  %s/%s: %d computed of %d products", cat, attr, n_computed, n_total)
    return pd.DataFrame(rows)


def main():
    for cat in MAIN_CATEGORIES:
        logger.info("=== %s ===", cat.upper())
        df = recompute_cat(cat)
        if len(df) == 0:
            logger.info("  no TYPE_C attrs for this cat, skipping")
            continue
        out = PROCESSED / f"{cat}_silver_type_c_fresh.parquet"
        df.to_parquet(out, index=False)
        logger.info("  saved: %s (%d rows)", out, len(df))


if __name__ == "__main__":
    main()
