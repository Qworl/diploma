"""Compare hybrid cascade headline numbers across gold dataset versions.

Prints a pivot table: per (cat, attr): silver_cascade vs hybrid_v1 (717) vs hybrid_v2 (2700).

Usage:
    python -m src.experiments.compare_gold_versions
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Compare gold versions headline accuracy")
    ap.add_argument("--silver", type=Path,
                    default=Path(PROCESSED_DIR) / "headline_results_off_grounded.parquet",
                    help="Silver cascade headline parquet")
    ap.add_argument("--hybrid-v1", type=Path,
                    default=Path(PROCESSED_DIR) / "headline_results_off_grounded_hybrid.parquet",
                    help="Hybrid v1 (717 gold) headline parquet")
    ap.add_argument("--hybrid-v2", type=Path,
                    default=Path(PROCESSED_DIR) / "headline_results_off_grounded_hybrid_v2.parquet",
                    help="Hybrid v2 (expanded gold) headline parquet")
    args = ap.parse_args()

    dfs = {}
    for name, path in [
        ("silver_cascade", args.silver),
        ("hybrid_v1_717", args.hybrid_v1),
        ("hybrid_v2_2700", args.hybrid_v2),
    ]:
        if path.exists():
            df = pd.read_parquet(path)
            df = df[["category", "attr", "accuracy"]].rename(columns={"accuracy": name})
            dfs[name] = df
            logger.info("Loaded %s: %d rows", name, len(df))
        else:
            logger.warning("Not found: %s", path)

    if not dfs:
        logger.error("No data found")
        return

    # Merge all available versions
    combined = None
    for name, df in dfs.items():
        if combined is None:
            combined = df
        else:
            combined = combined.merge(df, on=["category", "attr"], how="outer")

    combined = combined.sort_values(["category", "attr"])

    # Compute deltas where possible
    versions = list(dfs.keys())
    if "silver_cascade" in versions and "hybrid_v1_717" in versions:
        combined["v1_vs_silver"] = (combined["hybrid_v1_717"] - combined["silver_cascade"]).round(4)
    if "hybrid_v1_717" in versions and "hybrid_v2_2700" in versions:
        combined["v2_vs_v1"] = (combined["hybrid_v2_2700"] - combined["hybrid_v1_717"]).round(4)
    if "silver_cascade" in versions and "hybrid_v2_2700" in versions:
        combined["v2_vs_silver"] = (combined["hybrid_v2_2700"] - combined["silver_cascade"]).round(4)

    print("\n=== Accuracy comparison across gold versions ===")
    print(combined.to_string(index=False))

    # Summary per version
    print("\n=== Mean accuracy per version ===")
    for col in versions:
        if col in combined.columns:
            mean = combined[col].mean()
            print(f"  {col:25s}: {mean:.4f}")

    # Per-cat summary
    print("\n=== Mean accuracy per (cat, version) ===")
    for cat in sorted(combined["category"].unique()):
        cat_df = combined[combined["category"] == cat]
        row = f"  {cat:12s}"
        for col in versions:
            if col in cat_df.columns:
                row += f" | {col.split('_')[-1]}: {cat_df[col].mean():.3f}"
        print(row)


if __name__ == "__main__":
    main()
