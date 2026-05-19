"""
Replace silver-standard attributes with majority vote across (Sonnet, GPT-4o-mini,
Gemini-2.5-flash-lite). Apply labels_tags-fix to alt-model outputs first so all
three sit on the same ground.

Decision rule per row:
- If 2 or 3 models agree (non-null) → use that value
- If all 3 differ OR all 3 null → keep Sonnet's value (don't make worse)
- Null votes are abstentions (counted in non-null subset only)

Usage:
    python -m src.diagnostics.silver.ensemble --category pasta --attrs grain_type
    python -m src.diagnostics.silver.ensemble --category pasta --attrs all
    python -m src.diagnostics.silver.ensemble --category pasta --attrs all --dry-run
"""

import argparse
import logging
import os
import sys
from collections import Counter

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.pipeline.schemas import PASTA_SCHEMA
from src.diagnostics.silver.compare import apply_labels_fix, normalize, ALT_MODELS, model_tag

logger = logging.getLogger(__name__)

CATEGORY_CONFIG = {
    "pasta": {
        "parquet": "pasta_silver_standard.parquet",
        "schema": PASTA_SCHEMA,
        "attrs": ["grain_type", "pasta_shape", "is_whole_grain", "is_organic",
                  "is_gluten_free", "is_vegan"],
    },
}


def majority_vote(values: list) -> tuple[object, str]:
    """Return (chosen_value, reason).

    reason ∈ {"unanimous", "majority", "split", "all_null"}
    """
    non_null = [v for v in values if v is not None]
    if not non_null:
        return None, "all_null"
    counts = Counter(non_null)
    top, top_n = counts.most_common(1)[0]
    if top_n >= 2:
        return top, "unanimous" if top_n == 3 else "majority"
    return None, "split"


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()))
    parser.add_argument("--attrs", required=True,
                        help='Comma-separated attribute list, or "all"')
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = CATEGORY_CONFIG[args.category]
    if args.attrs == "all":
        target_attrs = cfg["attrs"]
    else:
        target_attrs = [a.strip() for a in args.attrs.split(",")]
        unknown = [a for a in target_attrs if a not in cfg["attrs"]]
        if unknown:
            logger.error("Unknown attrs for %s: %s", args.category, unknown)
            return

    src = os.path.join(PROCESSED_DIR, cfg["parquet"])
    ss = pd.read_parquet(src)
    n_before = len(ss)
    ss = ss.drop_duplicates(subset="code", keep="first").reset_index(drop=True)
    if len(ss) != n_before:
        logger.warning("Dropped %d duplicate-code rows from silver standard", n_before - len(ss))
    logger.info("Loaded silver standard %s (%d rows)", cfg["parquet"], len(ss))

    # Load alt models, apply labels_tags-fix
    alts = {}
    for short, full in ALT_MODELS:
        path = os.path.join(PROCESSED_DIR, f"relabel_{args.category}__{model_tag(full)}.parquet")
        if not os.path.exists(path):
            logger.warning("Missing relabel: %s — skipping %s", path, short)
            continue
        df = pd.read_parquet(path).drop_duplicates(subset="code", keep="first")
        df = apply_labels_fix(df, ss)
        alts[short] = df.set_index("code")
    if len(alts) < 2:
        logger.error("Need at least 2 alt models — only have %d", len(alts))
        return

    ss = ss.set_index("code")

    for attr in target_attrs:
        if attr not in ss.columns:
            logger.warning("Skipping %s (not in silver standard)", attr)
            continue

        stats = Counter()
        changes = 0
        examples_change = []
        examples_split = []

        for code in ss.index:
            sonnet_v = normalize(ss.at[code, attr])
            alt_vs = []
            for short, df in alts.items():
                if code in df.index and attr in df.columns:
                    alt_vs.append(normalize(df.at[code, attr]))
                else:
                    alt_vs.append(None)

            all_three = [sonnet_v] + alt_vs
            chosen, reason = majority_vote(all_three)
            stats[reason] += 1

            if reason in ("unanimous", "majority") and chosen != sonnet_v:
                ss.at[code, attr] = chosen
                changes += 1
                if len(examples_change) < 5:
                    name = ss.at[code, "product_name"] if "product_name" in ss.columns else "?"
                    examples_change.append(
                        (str(name)[:65], sonnet_v, chosen, alt_vs)
                    )
            elif reason == "split" and len(examples_split) < 3:
                name = ss.at[code, "product_name"] if "product_name" in ss.columns else "?"
                examples_split.append((str(name)[:65], all_three))

        logger.info("[%s] %s — voting: %s | changed: %d", args.category, attr, dict(stats), changes)
        for name, prev, new, alts_v in examples_change:
            logger.info("    %r → %r  (votes: sonnet=%r, gpt=%r, gemini=%r) | %s",
                        prev, new, alts_v[0] if False else None, alts_v[0], alts_v[1], name)

    if args.dry_run:
        logger.info("[dry-run] not saving")
        return

    ss = ss.reset_index()
    ss.to_parquet(src, index=False)
    logger.info("Saved %s", src)


if __name__ == "__main__":
    main()
