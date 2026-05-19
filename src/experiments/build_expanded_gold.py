"""Convert gpt-5.5 B-scale annotation parquets → long-format gold, append to v2 gold.

Input:  datasets/processed/gpt55_gold/{cat}_gpt55_gold.parquet
        (each row has code, parsed_json with attr→value dict)
Output: datasets/processed/consensus_gold_v2_expanded.parquet
        (same schema as consensus_gold_v2_off_grounded.parquet, with new rows appended)

Usage:
    python -m src.experiments.build_expanded_gold [--gold-dir ...] [--base-gold ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.eval.attribute_taxonomy import classify_attribute

logger = logging.getLogger(__name__)

CATS = ["pasta", "chocolate", "cheeses"]
DEFAULT_GOLD_DIR = Path("datasets/processed/gpt55_gold")
BASE_GOLD = Path(PROCESSED_DIR) / "consensus_gold_v2_off_grounded.parquet"
OUT_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet"


def gpt55_parquet_to_long(cat: str, parquet_path: Path) -> pd.DataFrame:
    """Convert a gpt-5.5 annotation parquet to long-format gold rows.

    Schema of output:
        category, code, attr, gold_value, gold_is_null, opus_reasoning, signal_type
    """
    df = pd.read_parquet(parquet_path)
    df["code"] = df["code"].astype(str)

    rows = []
    for _, row in df.iterrows():
        code = row["code"]
        try:
            parsed = json.loads(row["parsed_json"])
        except Exception:
            parsed = {}
        if not parsed:
            # LLM returned nothing / failed to parse
            logger.warning("[%s] empty parsed_json for code=%s", cat, code)
            continue
        for attr, val in parsed.items():
            is_null = val is None
            gold_value = None if is_null else str(val)
            rows.append({
                "category": cat,
                "code": code,
                "attr": attr,
                "gold_value": gold_value,
                "gold_is_null": is_null,
                "opus_reasoning": None,  # not available for gpt-5.5
                "signal_type": classify_attribute(cat, attr),
            })

    return pd.DataFrame(rows, columns=[
        "category", "code", "attr", "gold_value", "gold_is_null",
        "opus_reasoning", "signal_type",
    ])


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Build expanded gold parquet from gpt-5.5 annotations")
    ap.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    ap.add_argument("--base-gold", type=Path, default=BASE_GOLD)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--cats", nargs="+", default=CATS)
    args = ap.parse_args()

    # Load existing v2 gold (do NOT modify it)
    base = pd.read_parquet(args.base_gold)
    base["code"] = base["code"].astype(str)
    logger.info("Base v2 gold: %d rows, %d unique codes",
                len(base), base["code"].nunique())

    # Load gpt-5.5 annotations for each cat
    new_rows = []
    for cat in args.cats:
        p = args.gold_dir / f"{cat}_gpt55_gold.parquet"
        if not p.exists():
            logger.warning("[%s] gpt55_gold parquet not found: %s — skipping", cat, p)
            continue
        long_df = gpt55_parquet_to_long(cat, p)
        logger.info("[%s] converted: %d rows from %d unique codes",
                    cat, len(long_df), long_df["code"].nunique())
        new_rows.append(long_df)

    if not new_rows:
        logger.error("No new gpt-5.5 data found — nothing to append")
        return

    new_df = pd.concat(new_rows, ignore_index=True)

    # Check for overlapping codes (should be zero by design)
    base_codes = set(zip(base["category"], base["code"]))
    new_codes = set(zip(new_df["category"], new_df["code"]))
    overlap = base_codes & new_codes
    if overlap:
        logger.warning("%d (cat, code) pairs overlap with base gold — these will be deduplicated",
                       len(overlap))
        # Keep base gold for overlap codes (already have Opus annotation)
        new_df_filtered = new_df[
            ~new_df.apply(lambda r: (r["category"], r["code"]) in base_codes, axis=1)
        ]
        logger.info("After dedup: %d new rows remain", len(new_df_filtered))
        new_df = new_df_filtered

    # Combine
    expanded = pd.concat([base, new_df], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_parquet(args.out, index=False)
    logger.info("Expanded gold: %d total rows, %d unique codes — saved to %s",
                len(expanded), expanded["code"].nunique(), args.out)

    # Summary per category
    for cat in expanded["category"].unique():
        cat_df = expanded[expanded["category"] == cat]
        n_codes = cat_df["code"].nunique()
        n_base = base[base["category"] == cat]["code"].nunique()
        logger.info("[%s] base=%d codes → expanded=%d codes (+%d)",
                    cat, n_base, n_codes, n_codes - n_base)


if __name__ == "__main__":
    main()
