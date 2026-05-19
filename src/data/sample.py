"""
Language-stratified sampling for silver standard generation.

Phase 13 — multilingual stabilization. Baseline silver standard is biased toward
French (~60-70% of all examples) because OFF dump itself is FR-heavy. This script
takes a category's pre-filtered raw parquet and produces an N-per-language sample
across the top European languages, so downstream classifier sees balanced input.

Pipeline:
  1. Load raw filtered parquet (pasta_cereals / chocolate_raw / beverages_raw)
  2. Drop rows without product_name and with very low completeness (likely junk)
  3. Detect language on product_name + ingredients_text via langdetect
  4. For each target language, take top-N by completeness (best-curated rows first)
  5. Concat → save as {category}_stratified_raw.parquet

Usage:
    python scripts/sample_stratified.py --category pasta --per-lang 250
    python scripts/sample_stratified.py --category chocolate --per-lang 250 --langs fr,en,es,de,it
"""

import argparse
import logging
import os

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from tqdm import tqdm

from src.common import PROCESSED_DIR, setup_logging

DetectorFactory.seed = 0
logger = logging.getLogger(__name__)

# Reuse CATEGORY_CLEANUP outputs as inputs
INPUT_PARQUETS = {
    "pasta": "pasta_raw.parquet",
    "chocolate": "chocolate_raw.parquet",
    "beverages": "beverages_raw.parquet",
    "baby": "baby_raw.parquet",
    "cheeses": "cheeses_raw.parquet",
    "cereals": "cereals_raw.parquet",
    "cosmetics": "cosmetics_raw.parquet",
}

DEFAULT_LANGS = ["fr", "en", "es", "de", "it"]
DEFAULT_PER_LANG = 250
MIN_COMPLETENESS = 0.4
MIN_TEXT_LEN = 8

# Per-category overrides (см. domain_proposal_2026-05.md):
# pet_food имеет ~2.5K продуктов после фильтра, поэтому per_lang=150
# и MIN_COMPLETENESS=0.2 (ниже общего, иначе слишком мало семплов).
CATEGORY_DEFAULTS = {
    "cosmetics": {"per_lang": 250, "min_completeness": 0.3,
                  "langs": ["fr", "en", "de", "es", "it"]},
    "baby": {"per_lang": 250, "min_completeness": 0.4,
             "langs": ["fr", "en", "de", "es", "it"]},
    # Cheeses: 89K в OFF, FR-heavy (37K France). 250×5 = 1250 — easy.
    "cheeses": {"per_lang": 250, "min_completeness": 0.4,
                "langs": ["fr", "en", "de", "es", "it"]},
    # Cereals: 25K в OFF, breakfast subset поменьше. 200×5 = 1000.
    "cereals": {"per_lang": 200, "min_completeness": 0.4,
                "langs": ["fr", "en", "de", "es", "it"]},
}


def detect_lang(row: pd.Series) -> str:
    parts = []
    for col in ("product_name", "ingredients_text"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    text = " ".join(parts)
    if len(text) < MIN_TEXT_LEN:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=list(INPUT_PARQUETS.keys()))
    parser.add_argument("--per-lang", type=int, default=None)
    parser.add_argument("--langs", default=None,
                        help="Comma-separated ISO 639-1 codes")
    parser.add_argument("--min-completeness", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Per-category defaults take effect when CLI flag is not given
    cat_defaults = CATEGORY_DEFAULTS.get(args.category, {})
    per_lang = args.per_lang if args.per_lang is not None else cat_defaults.get("per_lang", DEFAULT_PER_LANG)
    min_completeness = (args.min_completeness if args.min_completeness is not None
                        else cat_defaults.get("min_completeness", MIN_COMPLETENESS))
    langs_str = args.langs if args.langs is not None else ",".join(cat_defaults.get("langs", DEFAULT_LANGS))
    target_langs = [l.strip().lower() for l in langs_str.split(",")]
    in_path = os.path.join(PROCESSED_DIR, INPUT_PARQUETS[args.category])
    out_path = args.output or os.path.join(
        PROCESSED_DIR, f"{args.category}_stratified_raw.parquet"
    )

    logger.info("Loading %s ...", in_path)
    df = pd.read_parquet(in_path)
    logger.info("Loaded %d products", len(df))

    df = df[df["product_name"].notna()].copy()
    if "completeness" in df.columns:
        df["_completeness"] = pd.to_numeric(df["completeness"], errors="coerce").fillna(0.0)
    else:
        df["_completeness"] = 0.5
    n_before_quality = len(df)
    df = df[df["_completeness"] >= min_completeness]
    logger.info("Quality filter (completeness>=%.2f, name notna): %d -> %d",
                min_completeness, n_before_quality, len(df))

    # Language detection — slow part, ~5-10 min on 100k rows.
    logger.info("Detecting language on %d products (this takes a few minutes)...", len(df))
    tqdm.pandas(desc="langdetect")
    df["_lang"] = df.progress_apply(detect_lang, axis=1)

    full_dist = df["_lang"].value_counts()
    logger.info("=== Language distribution in source pool ===")
    for lang, n in full_dist.head(15).items():
        marker = " <- target" if lang in target_langs else ""
        logger.info("  %-10s %6d (%.1f%%)%s", lang, n, 100 * n / len(df), marker)

    # Stratified sample: top-N per language by completeness
    parts = []
    summary_rows = []
    for lang in target_langs:
        pool = df[df["_lang"] == lang].sort_values("_completeness", ascending=False)
        n_available = len(pool)
        n_taken = min(per_lang, n_available)
        if n_taken < per_lang:
            logger.warning("  %s: only %d available (wanted %d) — taking all",
                           lang, n_available, per_lang)
        parts.append(pool.head(n_taken))
        summary_rows.append({"lang": lang, "available": n_available, "taken": n_taken})

    sample = pd.concat(parts, ignore_index=True)
    sample = sample.drop(columns=["_completeness", "_lang"])
    logger.info("=== Stratified sample summary ===")
    for r in summary_rows:
        logger.info("  %s: taken %d / available %d", r["lang"], r["taken"], r["available"])
    logger.info("Total: %d products", len(sample))

    sample.to_parquet(out_path, index=False)
    logger.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
