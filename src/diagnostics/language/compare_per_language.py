"""
Side-by-side comparison of two per-language metrics CSV files.

Designed for Phase 13: compares baseline (FR-biased silver standard) against
stratified (balanced) variant for the same category. Output emphasizes deltas
so improvements/regressions are obvious.

Usage:
    python -m src.diagnostics.language.compare_per_language \\
        --baseline datasets/processed/pasta_per_language_metrics.csv \\
        --variant datasets/processed/pasta_stratified_per_language_metrics.csv \\
        --metric accuracy
"""

import argparse
import logging
import os
import sys

import pandas as pd

from src.common import setup_logging

logger = logging.getLogger(__name__)


def load_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"lang", "attr", "accuracy", "coverage"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def build_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot(index="lang", columns="attr", values=metric)


def format_diff(baseline: pd.DataFrame, variant: pd.DataFrame, label: str):
    common_idx = baseline.index.intersection(variant.index)
    common_cols = baseline.columns.intersection(variant.columns)
    base = baseline.loc[common_idx, common_cols]
    var = variant.loc[common_idx, common_cols]
    diff = (var - base) * 100  # percentage points

    logger.info("=" * 90)
    logger.info("BASELINE %s (× 100):", label)
    for line in (base * 100).round(1).fillna("—").to_string().split("\n"):
        logger.info("  %s", line)

    logger.info("=" * 90)
    logger.info("STRATIFIED %s (× 100):", label)
    for line in (var * 100).round(1).fillna("—").to_string().split("\n"):
        logger.info("  %s", line)

    logger.info("=" * 90)
    logger.info("DELTA (stratified − baseline, percentage points). + = improvement:")
    for line in diff.round(1).fillna("—").to_string().split("\n"):
        logger.info("  %s", line)

    # Flatten + sort by abs delta for headline numbers
    flat = diff.stack().reset_index()
    flat.columns = ["lang", "attr", "delta_pp"]
    flat = flat.dropna(subset=["delta_pp"])
    flat["abs_delta"] = flat["delta_pp"].abs()
    top_changes = flat.sort_values("abs_delta", ascending=False).head(15)
    logger.info("=" * 90)
    logger.info("Top 15 (lang, attr) changes by |Δ%s|:", label)
    for _, r in top_changes.iterrows():
        sign = "+" if r["delta_pp"] >= 0 else ""
        logger.info("  %-7s %-22s %s%.1f pp", r["lang"], r["attr"], sign, r["delta_pp"])


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--metric", choices=["accuracy", "coverage", "macro_f1", "both"],
                        default="both", help="Use 'both' to print accuracy + coverage tables")
    args = parser.parse_args()

    base_df = load_metrics(args.baseline)
    var_df = load_metrics(args.variant)

    logger.info("Baseline: %s (%d rows)", args.baseline, len(base_df))
    logger.info("Variant:  %s (%d rows)", args.variant, len(var_df))

    metrics_to_show = ["accuracy", "coverage"] if args.metric == "both" else [args.metric]
    for m in metrics_to_show:
        if m not in base_df.columns:
            logger.warning("Skipping %s — not in baseline", m)
            continue
        logger.info("\n" + "#" * 90 + f"\n###   METRIC: {m.upper()}\n" + "#" * 90)
        b = build_pivot(base_df, m)
        v = build_pivot(var_df, m)
        format_diff(b, v, m)


if __name__ == "__main__":
    main()
