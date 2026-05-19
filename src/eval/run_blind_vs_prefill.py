"""CLI: load v1 prefill + v2 blind + silver, compute Checkpoint 1 metrics.

Outputs three parquets:
  - blind_vs_prefill_overall.parquet  (one row per category + 'ALL')
  - blind_vs_prefill_per_attr.parquet (one row per (category, attr))
  - blind_vs_prefill_flip.parquet     (one row per (category, attr) with flip counts)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.common import PROCESSED_DIR
from src.eval.blind_vs_prefill_analysis import (
    compute_agreement,
    compute_per_attr_metrics,
    compute_flip_direction,
)

logger = logging.getLogger(__name__)


def _load_prefill_for_category(cat: str) -> dict:
    """Load prefill-Opus decisions from legacy opus_batches.

    Pasta uses opus_decisions_all.json (per repo state), chocolate/cheeses
    use {cat}_decisions.json.
    """
    base = Path("datasets/manual_label/opus_batches")
    if cat == "pasta":
        path = base / "opus_decisions_all.json"
    else:
        path = base / f"{cat}_decisions.json"
    if not path.exists():
        raise FileNotFoundError(f"Prefill decisions not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_blind_for_category(cat: str) -> dict:
    path = Path(f"datasets/manual_label/opus_batches/blind_v2/{cat}_decisions.json")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_silver_for_category(cat: str) -> pd.DataFrame:
    """Load silver values in wide format with `code` column.

    Silver standards are stored in wide format with code as a regular column.
    """
    path = Path(f"{PROCESSED_DIR}/{cat}_stratified_silver_standard.parquet")
    df = pd.read_parquet(path)
    df["code"] = df["code"].astype(str)
    return df


def main(
    categories: tuple[str, ...] = ("pasta", "chocolate", "cheeses"),
    out_dir: str = "datasets/processed",
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(out_dir)

    overall_rows = []
    per_attr_dfs = []
    flip_dfs = []

    for cat in categories:
        try:
            prefill = _load_prefill_for_category(cat)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", cat, e)
            continue
        blind = _load_blind_for_category(cat)
        silver = _load_silver_for_category(cat)

        agreement = compute_agreement(prefill, blind)
        agreement["category"] = cat
        overall_rows.append(agreement)

        per_attr = compute_per_attr_metrics(prefill, blind)
        per_attr["category"] = cat
        per_attr_dfs.append(per_attr)

        flip = compute_flip_direction(prefill, blind, silver)
        flip["category"] = cat
        flip_dfs.append(flip)

        logger.info("[%s] overall_agreement=%.3f n_non_null=%d",
                    cat, agreement["overall_agreement"], agreement["n_non_null_cells"])

    overall_df = pd.DataFrame(overall_rows)
    # Add aggregate row
    if not overall_df.empty:
        agg = {
            "category": "ALL",
            "n_total_cells": int(overall_df["n_total_cells"].sum()),
            "n_non_null_cells": int(overall_df["n_non_null_cells"].sum()),
            "n_agree": int(overall_df["n_agree"].sum()),
        }
        agg["overall_agreement"] = (
            agg["n_agree"] / agg["n_non_null_cells"] if agg["n_non_null_cells"] else 0.0
        )
        overall_df = pd.concat([overall_df, pd.DataFrame([agg])], ignore_index=True)

    per_attr_combined = pd.concat(per_attr_dfs, ignore_index=True) if per_attr_dfs else pd.DataFrame()
    flip_combined = pd.concat(flip_dfs, ignore_index=True) if flip_dfs else pd.DataFrame()

    overall_df.to_parquet(out_dir / "blind_vs_prefill_overall.parquet", index=False)
    per_attr_combined.to_parquet(out_dir / "blind_vs_prefill_per_attr.parquet", index=False)
    flip_combined.to_parquet(out_dir / "blind_vs_prefill_flip.parquet", index=False)

    logger.info("=" * 60)
    logger.info("CHECKPOINT 1 SUMMARY:")
    logger.info("%s", overall_df.to_string(index=False))
    logger.info("=" * 60)
    logger.info("Per-attr written to blind_vs_prefill_per_attr.parquet")
    logger.info("Flip direction written to blind_vs_prefill_flip.parquet")


if __name__ == "__main__":
    main()
