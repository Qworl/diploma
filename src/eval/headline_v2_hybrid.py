"""Headline cascade accuracy on v2 OFF-grounded blind gold — HYBRID models.

Same logic as headline_v2.py but passes use_hybrid=True to predict_cascade.
Output: datasets/processed/headline_results_off_grounded_hybrid.parquet

NOTE: The hybrid models were trained on ALL v2 gold (no hold-out). The gold
evaluation set OVERLAPS with the training set. Numbers will be optimistically
biased vs. the honest 80/20 Task 5.5 estimate. Use Task 5.5 numbers for the
unbiased acc estimate; this file is for production deployment comparison.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.eval.cascade_predict import predict_cascade
from src.eval.headline_v2 import compute_headline

logger = logging.getLogger(__name__)

V2_GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_off_grounded.parquet"
OFF_GROUNDED_CATS = ["pasta", "chocolate", "cheeses"]


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--gold", type=Path, default=V2_GOLD_PATH)
    p.add_argument("--out", type=Path,
                   default=Path(PROCESSED_DIR) / "headline_results_off_grounded_hybrid.parquet")
    p.add_argument("--cats", nargs="+", default=OFF_GROUNDED_CATS)
    args = p.parse_args()

    gold = pd.read_parquet(args.gold)
    gold["code"] = gold["code"].astype(str)
    silver_codes_by_cat = {}
    for cat in args.cats:
        split = pd.read_parquet(Path(PROCESSED_DIR) / f"{cat}_gold_split.parquet")
        test_codes = set(split.loc[split["split"] == "test", "code"].astype(str))
        silver_codes_by_cat[cat] = test_codes

    all_results = []
    for cat in args.cats:
        cat_codes = silver_codes_by_cat[cat]
        cat_gold = gold[(gold["category"] == cat) & (gold["code"].isin(cat_codes))].copy()
        codes = sorted(cat_gold["code"].unique())
        if not codes:
            logger.warning("No overlap between v2 gold and brand-disjoint test for %s", cat)
            continue
        logger.info("[%s] intersection size = %d codes", cat, len(codes))
        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet")
        silver["code"] = silver["code"].astype(str)
        products = silver[silver["code"].isin(codes)].copy()
        # use_hybrid=True: loads hybrid models (silver + 5x v2 gold, all data)
        cascade = predict_cascade(products, category=f"{cat}_stratified", use_hybrid=True)
        result = compute_headline(cat_gold, cascade, category=cat)
        all_results.append(result)
        logger.info("[%s] %d (cat, attr) rows; n=%d codes; mean accuracy=%.3f",
                    cat, len(result), len(codes), result["accuracy"].mean())

    final = pd.concat(all_results, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(args.out, index=False)
    logger.info("Wrote %s (%d rows)", args.out, len(final))

    # Compare to silver cascade
    silver_path = Path(PROCESSED_DIR) / "headline_results_off_grounded.parquet"
    if silver_path.exists():
        silver_hl = pd.read_parquet(silver_path)
        merged = final.merge(
            silver_hl[["category", "attr", "accuracy"]].rename(
                columns={"accuracy": "silver_acc"}),
            on=["category", "attr"], how="left",
        )
        merged["delta_pp"] = (merged["accuracy"] - merged["silver_acc"]) * 100
        print("\n=== Per-(cat, attr) hybrid vs silver delta ===")
        print(merged[["category", "attr", "silver_acc", "accuracy", "delta_pp"]]
              .to_string(index=False))
        print(f"\nMean delta: {merged['delta_pp'].mean():.2f} pp")
        print(f"Hybrid mean acc: {final['accuracy'].mean():.4f}")
        print(f"Silver mean acc: {silver_hl['accuracy'].mean():.4f}")
    else:
        print(f"\nNo silver baseline at {silver_path}, skipping delta.")
        print(f"Hybrid mean acc: {final['accuracy'].mean():.4f}")


if __name__ == "__main__":
    main()
