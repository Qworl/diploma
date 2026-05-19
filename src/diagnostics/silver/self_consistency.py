"""
Re-label a stratified sample of products with the same Sonnet prompt and
measure how often the LLM disagrees with its own prior labelling.

This estimates the noise floor of the silver standard. Any pipeline metric
above (1 - disagreement) is statistically indistinguishable from the LLM.

Usage:
    python -m src.diagnostics.silver.self_consistency --category pasta --sample 100
"""

import argparse
import logging
import os
import sys
import json
from collections import Counter

import numpy as np
import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.pipeline.schemas import PASTA_SCHEMA
from src.pipeline.llm_fallback import enrich_batch

logger = logging.getLogger(__name__)

CATEGORY_CONFIG = {
    "pasta": {
        "parquet": "pasta_silver_standard.parquet",
        "schema": PASTA_SCHEMA,
        "stratify": "grain_type",
        "attrs": ["grain_type", "pasta_shape", "is_whole_grain", "is_organic",
                  "is_gluten_free"],
    },
}


def stratified_sample(df: pd.DataFrame, by: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample n rows roughly equally across groups of `by`."""
    groups = df.groupby(by, dropna=False)
    n_groups = len(groups)
    per_group = max(1, n // n_groups)

    parts = []
    for _, grp in groups:
        take = min(per_group, len(grp))
        idx = rng.choice(grp.index.values, size=take, replace=False)
        parts.append(df.loc[idx])
    out = pd.concat(parts)
    if len(out) > n:
        out = out.sample(n=n, random_state=int(rng.integers(0, 2**31)))
    return out


def cohen_kappa_safe(y1, y2) -> float:
    """Cohen's kappa with NaN handling — returns nan if undefined."""
    a = pd.Series(y1).astype(str)
    b = pd.Series(y2).astype(str)
    mask = (a != "nan") & (b != "nan")
    if mask.sum() < 2:
        return float("nan")
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(a[mask], b[mask]))
    except Exception:
        return float("nan")


def normalize(v):
    """Normalise pandas/numpy bools/NaN to a comparable scalar."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
    return v


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()))
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--model", default="anthropic/claude-haiku-4.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Where to dump results parquet (default: datasets/processed/selfconsistency_<cat>.parquet)")
    args = parser.parse_args()

    cfg = CATEGORY_CONFIG[args.category]
    src_path = os.path.join(PROCESSED_DIR, cfg["parquet"])
    out_path = args.output or os.path.join(
        PROCESSED_DIR, f"selfconsistency_{args.category}.parquet"
    )

    df = pd.read_parquet(src_path)
    logger.info("Loaded %d products from %s", len(df), cfg["parquet"])

    rng = np.random.default_rng(args.seed)
    sample = stratified_sample(df, cfg["stratify"], args.sample, rng).reset_index(drop=True)
    logger.info("Stratified sample size: %d (by %s)", len(sample), cfg["stratify"])
    logger.info("Distribution in sample by %s: %s",
                cfg["stratify"], sample[cfg["stratify"]].value_counts(dropna=False).to_dict())

    # Re-label (or load cached if exists)
    relabel_path = out_path.replace(".parquet", "_relabel.parquet")
    if os.path.exists(relabel_path):
        logger.info("Loading cached relabel from %s", relabel_path)
        relabel = pd.read_parquet(relabel_path)
    else:
        logger.info("Calling %s on %d products...", args.model, len(sample))
        relabel = enrich_batch(
            sample, cfg["schema"], backend="openrouter", model=args.model,
            max_workers=10,
        )
        # Save raw LLM result IMMEDIATELY before any comparison work
        relabel.to_parquet(relabel_path, index=False)
        logger.info("Cached LLM relabel to %s", relabel_path)
    relabel = relabel.set_index("code")

    # Compare
    sample = sample.set_index("code")
    rows = []
    per_attr_counts = {a: {"agree": 0, "disagree": 0, "missing": 0} for a in cfg["attrs"]}
    examples = {a: [] for a in cfg["attrs"]}

    for code in sample.index:
        if code not in relabel.index:
            continue
        for attr in cfg["attrs"]:
            v_orig = normalize(sample.at[code, attr]) if attr in sample.columns else None
            v_new = normalize(relabel.at[code, attr]) if attr in relabel.columns else None
            row = {
                "code": code,
                "product_name": sample.at[code, "product_name"],
                "attr": attr,
                "v_orig": v_orig,
                "v_new": v_new,
            }
            if v_new is None and v_orig is None:
                continue  # both null
            if v_new is None:
                per_attr_counts[attr]["missing"] += 1
                continue
            if v_orig == v_new:
                per_attr_counts[attr]["agree"] += 1
            else:
                per_attr_counts[attr]["disagree"] += 1
                if len(examples[attr]) < 6:
                    examples[attr].append(row)
            rows.append(row)

    # Save raw — cast values to strings (mixed bool/str/None breaks parquet)
    rows_save = pd.DataFrame(rows)
    if len(rows_save):
        for col in ("v_orig", "v_new"):
            rows_save[col] = rows_save[col].apply(lambda v: "<NA>" if v is None else str(v))
    rows_save.to_parquet(out_path, index=False)
    logger.info("Saved per-attr comparison to %s", out_path)

    # Report
    print(f"\n{'='*78}")
    print(f"SELF-CONSISTENCY ({args.category}, n={len(sample)}, model={args.model})")
    print(f"{'='*78}")
    print(f"{'Attribute':<22} {'Agree':>7} {'Disagree':>9} {'Missing':>8} {'Agree %':>9} {'Kappa':>7}")
    print("-" * 78)
    for attr in cfg["attrs"]:
        c = per_attr_counts[attr]
        total = c["agree"] + c["disagree"]
        agree_pct = c["agree"] / total * 100 if total else 0
        # Kappa uses only rows where both are non-null
        pairs_orig = []
        pairs_new = []
        for r in rows:
            if r["attr"] == attr and r["v_orig"] is not None and r["v_new"] is not None:
                pairs_orig.append(r["v_orig"])
                pairs_new.append(r["v_new"])
        kappa = cohen_kappa_safe(pairs_orig, pairs_new)
        print(f"{attr:<22} {c['agree']:>7} {c['disagree']:>9} {c['missing']:>8} "
              f"{agree_pct:>8.1f}% {kappa:>7.3f}")

    print()
    print("Sample disagreements:")
    for attr in cfg["attrs"]:
        if not examples[attr]:
            continue
        print(f"\n  [{attr}]")
        for r in examples[attr]:
            print(f"    orig={str(r['v_orig']):<10} new={str(r['v_new']):<10} | {str(r['product_name'])[:70]}")


if __name__ == "__main__":
    main()
