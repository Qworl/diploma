"""
Re-label the full silver standard with an alternative LLM.

The output parquet contains the same `code` index and the schema's attributes,
written verbatim from the LLM (no labels_tags fixup applied — those are kept
in a separate post-processing layer).

Usage:
    python -m src.diagnostics.silver.relabel --category pasta --model openai/gpt-4o-mini
    python -m src.diagnostics.silver.relabel --category pasta --model google/gemini-flash-1.5
    python -m src.diagnostics.silver.relabel --category pasta --model openai/gpt-4o-mini --resume
"""

import argparse
import logging
import os
import sys

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.pipeline.schemas import PASTA_SCHEMA
from src.pipeline.llm_fallback import enrich_batch

logger = logging.getLogger(__name__)

CATEGORY_CONFIG = {
    "pasta": {
        "parquet": "pasta_silver_standard.parquet",
        "schema": PASTA_SCHEMA,
    },
}


def model_tag(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()))
    parser.add_argument("--model", required=True,
                        help="OpenRouter model ID, e.g. openai/gpt-4o-mini, google/gemini-flash-1.5")
    parser.add_argument("--limit", type=int, default=None, help="Only label first N rows (for smoke tests)")
    parser.add_argument("--resume", action="store_true", help="Skip rows already in output")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    cfg = CATEGORY_CONFIG[args.category]
    src = os.path.join(PROCESSED_DIR, cfg["parquet"])
    out = os.path.join(PROCESSED_DIR, f"relabel_{args.category}__{model_tag(args.model)}.parquet")

    df = pd.read_parquet(src)
    logger.info("Loaded %d products from %s", len(df), cfg["parquet"])

    if args.limit:
        df = df.head(args.limit)
        logger.info("Limited to %d", len(df))

    if args.resume and os.path.exists(out):
        existing = pd.read_parquet(out)
        done_codes = set(existing["code"].astype(str))
        df = df[~df["code"].astype(str).isin(done_codes)].copy()
        logger.info("Resuming: %d remaining (%d already done)", len(df), len(done_codes))

    if len(df) == 0:
        logger.info("Nothing to do.")
        return

    logger.info("Calling %s on %d products with %d workers...", args.model, len(df), args.workers)
    result = enrich_batch(
        df, cfg["schema"], backend="openrouter", model=args.model,
        max_workers=args.workers,
    )
    logger.info("Got %d results", len(result))

    if args.resume and os.path.exists(out):
        existing = pd.read_parquet(out)
        result = pd.concat([existing, result], ignore_index=True)
        result = result.drop_duplicates(subset="code", keep="last")

    result.to_parquet(out, index=False)
    logger.info("Saved %s (%d rows)", out, len(result))

    # Coverage stats per attribute
    schema = cfg["schema"]
    logger.info("--- Coverage per attribute ---")
    for attr in schema:
        if attr in result.columns:
            non_null = result[attr].notna().sum()
            logger.info("  %s: %d/%d (%.1f%%)", attr, non_null, len(result), non_null / len(result) * 100)


if __name__ == "__main__":
    main()
