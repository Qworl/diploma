"""
Compare silver standard (Sonnet, post-fix) against alternative-model relabels
(GPT-4o-mini, Gemini-2.5-flash-lite). Reports pairwise agreement, Cohen's kappa,
3-way consensus rate, and writes a "consensus subset" parquet for downstream use.

Apply labels_tags-fix to alt-model outputs first so all three sit on the same
ground (otherwise Sonnet "wins" trivially on the rows where we already
post-processed organic/gluten-free from labels_tags).

Usage:
    python -m src.diagnostics.silver.compare --category pasta
"""

import argparse
import logging
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.pipeline.schemas import PASTA_SCHEMA
from src.diagnostics.silver.fix_labels import (
    ORGANIC_TAGS, ORGANIC_PATTERNS, GLUTEN_FREE_TAGS, VEGAN_TAGS,
    LACTOSE_FREE_TAGS, _split_tags, has_any,
)

logger = logging.getLogger(__name__)

CATEGORY_CONFIG = {
    "pasta": {
        "parquet": "pasta_silver_standard.parquet",
        "schema": PASTA_SCHEMA,
        "attrs": ["grain_type", "pasta_shape", "is_whole_grain", "is_organic",
                  "is_gluten_free"],
    },
}

ALT_MODELS = [
    ("gpt-4o-mini", "openai/gpt-4o-mini"),
    ("gemini-flash", "google/gemini-2.5-flash-lite"),
]


def model_tag(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def apply_labels_fix(df: pd.DataFrame, ss_df: pd.DataFrame) -> pd.DataFrame:
    """Mirror fix_labels_from_tags but on a relabel df. Joins labels_tags from ss_df."""
    df = df.merge(ss_df[["code", "labels_tags"]], on="code", how="left")

    rules = [
        ("is_organic", ORGANIC_TAGS, ORGANIC_PATTERNS),
        ("is_gluten_free", GLUTEN_FREE_TAGS, ()),
        ("is_vegan", VEGAN_TAGS, ()),
        ("is_lactose_free", LACTOSE_FREE_TAGS, ()),
    ]
    for attr, tags, patterns in rules:
        if attr not in df.columns:
            continue
        for idx in df.index:
            t = _split_tags(df.at[idx, "labels_tags"])
            if has_any(t, tags, patterns):
                cur = df.at[idx, attr]
                is_true = cur is True or (isinstance(cur, bool) and cur) or (
                    hasattr(cur, "item") and cur is not None
                    and not pd.isna(cur) and bool(cur.item()) is True
                )
                if not is_true:
                    df.at[idx, attr] = True
    return df.drop(columns=["labels_tags"])


def normalize(v):
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


def to_str(v):
    v = normalize(v)
    return "<NA>" if v is None else str(v)


def cohen_kappa(a, b):
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(a, b))
    except Exception:
        return float("nan")


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()))
    parser.add_argument("--save-consensus", action="store_true",
                        help="Write consensus subset parquet for downstream use")
    args = parser.parse_args()

    cfg = CATEGORY_CONFIG[args.category]
    ss = pd.read_parquet(os.path.join(PROCESSED_DIR, cfg["parquet"]))
    n_before = len(ss)
    ss = ss.drop_duplicates(subset="code", keep="first")
    if len(ss) != n_before:
        logger.warning("Silver standard had %d duplicate codes — kept first", n_before - len(ss))
    logger.info("Silver standard: %d rows", len(ss))

    # Load alt models
    alts = {}
    for short, full in ALT_MODELS:
        path = os.path.join(PROCESSED_DIR, f"relabel_{args.category}__{model_tag(full)}.parquet")
        if not os.path.exists(path):
            logger.warning("Missing relabel file: %s", path)
            continue
        df = pd.read_parquet(path)
        df = df.drop_duplicates(subset="code", keep="first")
        logger.info("%s: %d rows from %s", short, len(df), os.path.basename(path))
        df = apply_labels_fix(df, ss)
        alts[short] = df

    if not alts:
        logger.error("No alt-model relabels found — run relabel_with_model.py first.")
        return

    # Merge on code: get one wide DF with sonnet_*, gpt_*, gemini_* columns per attribute
    ss = ss.set_index("code")
    wide = ss[cfg["attrs"]].copy()
    wide.columns = [f"sonnet__{c}" for c in wide.columns]

    for short, df in alts.items():
        df = df.set_index("code")
        for a in cfg["attrs"]:
            if a in df.columns:
                wide[f"{short}__{a}"] = df[a]

    # Pairwise agreement per attribute
    print(f"\n{'='*100}")
    print(f"INTER-MODEL AGREEMENT — {args.category} (n={len(wide)})")
    print(f"{'='*100}")

    sources = ["sonnet"] + list(alts.keys())

    # Pairwise agreement table
    print(f"\n{'Attribute':<22} {'pair':<24} {'agree %':>9} {'kappa':>8} {'n':>6}")
    print("-" * 80)
    for attr in cfg["attrs"]:
        for a, b in combinations(sources, 2):
            ca = f"{a}__{attr}"
            cb = f"{b}__{attr}"
            if ca not in wide.columns or cb not in wide.columns:
                continue
            sa = wide[ca].apply(to_str)
            sb = wide[cb].apply(to_str)
            mask = (sa != "<NA>") & (sb != "<NA>")
            n = int(mask.sum())
            if n == 0:
                continue
            agree = (sa[mask] == sb[mask]).mean() * 100
            k = cohen_kappa(sa[mask], sb[mask])
            print(f"{attr:<22} {a:>10} vs {b:<10} {agree:>8.1f}% {k:>8.3f} {n:>6}")
        print()

    # 3-way consensus
    print(f"\n{'='*100}")
    print("3-WAY CONSENSUS (all three models give the same non-null answer)")
    print(f"{'='*100}")
    print(f"{'Attribute':<22} {'Consensus':>10} {'2-of-3':>10} {'All differ':>11} {'Coverage':>9}")
    print("-" * 80)

    consensus_records = []
    for attr in cfg["attrs"]:
        cols = [f"{s}__{attr}" for s in sources if f"{s}__{attr}" in wide.columns]
        if len(cols) != 3:
            continue
        cons_count = 0
        two_of_three = 0
        all_diff = 0
        non_null_n = 0
        for code, row in wide.iterrows():
            vals = [to_str(row[c]) for c in cols]
            non_null = [v for v in vals if v != "<NA>"]
            if len(non_null) < 3:
                continue
            non_null_n += 1
            unique = set(non_null)
            if len(unique) == 1:
                cons_count += 1
                consensus_records.append({"code": code, "attr": attr, "value": non_null[0]})
            elif len(unique) == 2:
                two_of_three += 1
            else:
                all_diff += 1
        n = non_null_n
        if n == 0:
            continue
        print(f"{attr:<22} {cons_count:>5} ({cons_count/n*100:>4.1f}%) "
              f"{two_of_three:>4} ({two_of_three/n*100:>4.1f}%) "
              f"{all_diff:>4} ({all_diff/n*100:>4.1f}%) {n:>5}/{len(wide):<5}")

    # Save consensus subset
    if args.save_consensus and consensus_records:
        cons_df = pd.DataFrame(consensus_records)
        cons_pivot = cons_df.pivot(index="code", columns="attr", values="value").reset_index()
        # Add product_name for readability
        names = ss[["product_name"]].reset_index() if "product_name" in ss.columns else None
        if names is not None:
            cons_pivot = cons_pivot.merge(names, on="code", how="left")
        out = os.path.join(PROCESSED_DIR, f"consensus_{args.category}.parquet")
        cons_pivot.to_parquet(out, index=False)
        logger.info("Saved consensus subset to %s (%d rows × %d attrs)",
                    out, len(cons_pivot), len(cfg["attrs"]))

    # Save full comparison wide table
    out = os.path.join(PROCESSED_DIR, f"compare_{args.category}.parquet")
    wide.reset_index().to_parquet(out, index=False)
    logger.info("Saved wide comparison to %s", out)

    # Show some interesting disagreement examples
    print(f"\n{'='*100}")
    print("DISAGREEMENT EXAMPLES (3 different answers across models)")
    print(f"{'='*100}")
    name_map = ss["product_name"].to_dict() if "product_name" in ss.columns else {}
    shown = 0
    for code, row in wide.iterrows():
        if shown >= 12:
            break
        any_attr_3way = False
        for attr in cfg["attrs"]:
            cols = [f"{s}__{attr}" for s in sources if f"{s}__{attr}" in wide.columns]
            if len(cols) != 3:
                continue
            vals = [to_str(row[c]) for c in cols]
            non_null = [v for v in vals if v != "<NA>"]
            if len(non_null) == 3 and len(set(non_null)) == 3:
                any_attr_3way = True
                if shown < 12:
                    pname = str(name_map.get(code, "?"))[:70]
                    pairs = " | ".join(f"{s}={to_str(row[f'{s}__{attr}'])}" for s in sources)
                    print(f"  [{attr}] {pname}")
                    print(f"      {pairs}")
                    shown += 1
                    break


if __name__ == "__main__":
    main()
