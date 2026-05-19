"""
Override silver-standard boolean attributes when labels_tags carries a deterministic
signal that the LLM didn't see (labels_tags is not part of the LLM input).

Conservative direction only: flip False/None → True. Absence of a positive label
does not mean negative, so we never flip True → False.

Usage:
    python -m src.diagnostics.silver.fix_labels --dry-run
    python -m src.diagnostics.silver.fix_labels
"""

import argparse
import logging
import os
import sys

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)

# Only positive certifications. NOT included: en:eu-agriculture / en:non-eu-agriculture
# (those are origin markers, not organic claims).
ORGANIC_TAGS = {
    "en:organic",
    "en:eu-organic",
    "fr:ab-agriculture-biologique",
    "de:eg-öko-verordnung",
    "en:bio",
    "en:bio-suisse",
    "en:demeter",
    "en:naturland",
}

# Country/regulator-specific organic codes follow the pattern xx:yy-bio-NNN.
# Match by suffix to be exhaustive.
ORGANIC_PATTERNS = ("-bio-", ":bio-")

GLUTEN_FREE_TAGS = {
    "en:gluten-free",
    "en:no-gluten",
    "en:sans-gluten",
    "en:senza-glutine",
    "en:sin-gluten",
    "en:glutenfrei",
    "en:dzg-gluten-free",
    "en:crossed-grain-trademark",
    "en:afdiag",
}

VEGAN_TAGS = {
    "en:vegan",
    "fr:vegan",
    "en:vegan-society",
}

LACTOSE_FREE_TAGS = {
    "en:no-lactose",
    "en:lactose-free",
    "en:sans-lactose",
    "en:laktosefrei",
    "en:senza-lattosio",
    "en:sin-lactosa",
}


def _split_tags(value) -> list[str]:
    """labels_tags can be a comma-separated string or a list — normalise to lower-case list."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        return [t.strip().lower() for t in value.split(",") if t.strip()]
    if hasattr(value, "__iter__"):
        return [str(t).strip().lower() for t in value if str(t).strip()]
    return []


def has_any(tags: list[str], wanted: set[str], patterns: tuple[str, ...] = ()) -> bool:
    if any(t in wanted for t in tags):
        return True
    if patterns and any(any(p in t for p in patterns) for t in tags):
        return True
    return False


def apply_fix(
    df: pd.DataFrame, attr: str, wanted: set[str], patterns: tuple[str, ...] = ()
) -> int:
    """Flip df[attr] to True for rows whose labels_tags contain a positive marker.

    Returns number of flipped rows.
    """
    if attr not in df.columns:
        return 0
    if "labels_tags" not in df.columns:
        return 0

    flipped = 0
    examples = []
    for idx in df.index:
        tags = _split_tags(df.at[idx, "labels_tags"])
        if not has_any(tags, wanted, patterns):
            continue
        current = df.at[idx, attr]
        # Treat numpy bools, python bools, None, NaN uniformly.
        is_true = current is True or (isinstance(current, (bool,)) and current) or (
            hasattr(current, "item") and current is not None and not pd.isna(current) and bool(current.item()) is True
        )
        if is_true:
            continue
        df.at[idx, attr] = True
        flipped += 1
        if len(examples) < 5:
            examples.append((str(df.at[idx, "product_name"])[:70], current))

    if flipped:
        logger.info("  %s: %d rows flipped → True", attr, flipped)
        for name, prev in examples:
            logger.info("    [%r → True] %s", prev, name)
    return flipped


def fix_file(path: str, dry_run: bool = False) -> dict:
    df = pd.read_parquet(path)
    n = len(df)
    logger.info("=== %s (n=%d) ===", os.path.basename(path), n)

    counts = {}
    df_work = df.copy()

    counts["is_organic"] = apply_fix(df_work, "is_organic", ORGANIC_TAGS, ORGANIC_PATTERNS)
    counts["is_gluten_free"] = apply_fix(df_work, "is_gluten_free", GLUTEN_FREE_TAGS)
    counts["is_vegan"] = apply_fix(df_work, "is_vegan", VEGAN_TAGS)
    counts["is_lactose_free"] = apply_fix(df_work, "is_lactose_free", LACTOSE_FREE_TAGS)

    total = sum(counts.values())
    logger.info("Total flips: %d (%.1f%% of rows)", total, total / n * 100)

    if not dry_run and total > 0:
        df_work.to_parquet(path, index=False)
        logger.info("Saved %s", path)
    elif dry_run:
        logger.info("[dry-run] not saving")

    return counts


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = [
        "baby_merged_silver_standard.parquet",
        "pasta_silver_standard.parquet",
        "milk_substitutes_silver_standard.parquet",
        "mixed_cereals_silver_standard.parquet",
    ]

    grand_total = 0
    for t in targets:
        path = os.path.join(PROCESSED_DIR, t)
        if not os.path.exists(path):
            logger.warning("Missing: %s", path)
            continue
        counts = fix_file(path, dry_run=args.dry_run)
        grand_total += sum(counts.values())

    logger.info("\n=== GRAND TOTAL: %d label flips ===", grand_total)


if __name__ == "__main__":
    main()
