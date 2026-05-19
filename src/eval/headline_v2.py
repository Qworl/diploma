"""Headline cascade accuracy on v2 OFF-grounded blind gold.

Refusal-aware: cells where blind Opus returned null are excluded from the
denominator (gold_coverage_rate reported separately). Cascade abstain counts
as a miss on non-null gold cells (production users need an answer).

Output: one row per (category, attr) with accuracy + Wilson 95% CI.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging, wilson_ci
from src.eval.cascade_predict import predict_cascade

logger = logging.getLogger(__name__)

V2_GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_off_grounded.parquet"
OFF_GROUNDED_CATS = ["pasta", "chocolate", "cheeses"]


def compute_headline(
    gold: pd.DataFrame,
    cascade: pd.DataFrame,
    *,
    category: str,
) -> pd.DataFrame:
    """Compute per-(cat, attr) refusal-aware accuracy on v2 gold."""
    gold = gold[gold["category"] == category].copy()
    gold["code"] = gold["code"].astype(str)
    cascade = cascade.copy()
    cascade["code"] = cascade["code"].astype(str)

    merged = gold.merge(
        cascade[["code", "attr", "predicted", "layer"]],
        on=["code", "attr"], how="left",
    )
    nonnull = merged[~merged["gold_is_null"]].copy()
    nonnull["correct"] = (
        nonnull["predicted"].astype(object) == nonnull["gold_value"].astype(object)
    ).astype(int)
    nonnull.loc[nonnull["predicted"].isna(), "correct"] = 0

    rows = []
    for attr, g in merged.groupby("attr"):
        sub = nonnull[nonnull["attr"] == attr]
        n_non_null = len(sub)
        n_correct = int(sub["correct"].sum())
        n_total = len(g)
        n_gold_null = n_total - n_non_null
        coverage = n_non_null / n_total if n_total else float("nan")
        acc = n_correct / n_non_null if n_non_null else float("nan")
        if n_non_null:
            lo, hi = wilson_ci(n_correct, n_non_null)
        else:
            lo, hi = float("nan"), float("nan")
        sig = sub["signal_type"].iloc[0] if n_non_null else g["signal_type"].iloc[0]
        rows.append({
            "category": category, "attr": attr,
            "n_total_cells": n_total, "n_non_null_gold": n_non_null,
            "n_gold_null": n_gold_null, "gold_coverage_rate": coverage,
            "n_correct": n_correct, "accuracy": acc,
            "wilson_lower": lo, "wilson_upper": hi,
            "signal_type": sig,
        })
    return pd.DataFrame(rows)


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--gold", type=Path, default=V2_GOLD_PATH)
    p.add_argument("--out", type=Path,
                   default=Path(PROCESSED_DIR) / "headline_results_off_grounded.parquet")
    p.add_argument("--cats", nargs="+", default=OFF_GROUNDED_CATS)
    args = p.parse_args()

    gold = pd.read_parquet(args.gold)
    gold["code"] = gold["code"].astype(str)
    silver_codes_by_cat = {}
    # Restrict to brand-disjoint test of the matching silver split, per §4.1.
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
        intersection_size = len(codes)
        if intersection_size < 50:
            logger.warning(
                "[%s] CONCERN: intersection size = %d < 50 — Wilson CI may be unreliable. "
                "v2 gold may not have been drawn from brand-disjoint split.",
                cat, intersection_size,
            )
        else:
            logger.info("[%s] intersection size = %d codes", cat, intersection_size)
        silver = pd.read_parquet(Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet")
        silver["code"] = silver["code"].astype(str)
        products = silver[silver["code"].isin(codes)].copy()
        cascade = predict_cascade(products, category=f"{cat}_stratified")
        result = compute_headline(cat_gold, cascade, category=cat)
        all_results.append(result)
        logger.info("[%s] %d (cat, attr) rows; n=%d codes; mean accuracy=%.3f",
                    cat, len(result), len(codes), result["accuracy"].mean())

    final = pd.concat(all_results, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(args.out, index=False)
    logger.info("Wrote %s (%d rows)", args.out, len(final))


if __name__ == "__main__":
    main()
