"""
Per-language refresh на extended LLM-consensus gold (n=16360 cascade-valid, 2026-05-27).

Источники:
  - datasets/processed/manual_gold_consensus.parquet  (22207 cells × 3 cats)
  - datasets/processed/cascade_preds_{cat}_gold.parquet  (предсказания каскада на gold)
  - datasets/processed/{cat}_stratified_silver_standard.parquet  (тексты product_name/ingredients_text)

Логика:
  1. Берём cascade_preds_{cat}_gold (in_scope=True, cascade_pred not null) — cascade-valid срез.
  2. Join по code → manual_gold_consensus[gold_value] (по сути там уже есть, но используем как
     verification + получаем agreement_ratio при необходимости).
  3. Join по code → silver_standard[product_name + ingredients_text] для langdetect.
  4. agg metrics: (category, attr, language) → n_cells, accuracy.

Выход: datasets/processed/per_language_eval.parquet (полностью заменяет старый файл,
который был на n=3257 cascade-valid).

Usage:
    python -m src.diagnostics.language.per_language_refresh
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from tqdm import tqdm

from src.common import PROCESSED_DIR, setup_logging

DetectorFactory.seed = 0
logger = logging.getLogger(__name__)

REPORTED_LANGS = {"fr", "en", "de", "it", "es"}
CATEGORIES = ("pasta", "chocolate", "cheeses")


def detect_lang(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) < 8:
        return "unknown"
    try:
        return detect(text.strip())
    except LangDetectException:
        return "unknown"


def build_lang_lookup() -> pd.DataFrame:
    """Code → detected language (one row per code per category)."""
    rows = []
    for cat in CATEGORIES:
        sil_path = os.path.join(PROCESSED_DIR, f"{cat}_stratified_silver_standard.parquet")
        sil = pd.read_parquet(sil_path)
        sil["code"] = sil["code"].astype(str)
        sil = sil.drop_duplicates(subset=["code"])
        logger.info("Detecting language for %d %s codes...", len(sil), cat)
        texts = (sil["product_name"].fillna("") + " " + sil["ingredients_text"].fillna("")).str.strip()
        langs = [detect_lang(t) for t in tqdm(texts, desc=f"langdetect/{cat}", file=sys.stdout)]
        rows.append(pd.DataFrame({"code": sil["code"].values, "category": cat, "_lang_raw": langs}))
    out = pd.concat(rows, ignore_index=True)
    out["language"] = out["_lang_raw"].where(out["_lang_raw"].isin(REPORTED_LANGS), "other")
    return out


def load_cascade_valid() -> pd.DataFrame:
    """Concat cascade_preds for 3 cats, filter to in_scope=True & cascade_pred not null."""
    parts = []
    for cat in CATEGORIES:
        fp = os.path.join(PROCESSED_DIR, f"cascade_preds_{cat}_gold.parquet")
        df = pd.read_parquet(fp)
        df["code"] = df["code"].astype(str)
        df["category"] = cat
        before = len(df)
        df = df[df["in_scope"] & df["cascade_pred"].notna()].copy()
        logger.info("%s: %d cells (in_scope+cascade_valid) of %d total", cat, len(df), before)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["correct"] = (out["cascade_pred"].astype(str) == out["gold_value"].astype(str)).astype(int)
    return out


def aggregate(cascade_df: pd.DataFrame, lang_lookup: pd.DataFrame) -> pd.DataFrame:
    """Join cascade ⨝ lang, group by (category, attr, language)."""
    merged = cascade_df.merge(lang_lookup[["code", "category", "language"]],
                              on=["code", "category"], how="left")
    missing = merged["language"].isna().sum()
    if missing:
        logger.warning("%d cells (%d%%) have no language (code missing in silver_standard); "
                       "tagging as 'unknown'", missing, 100 * missing // len(merged))
        merged["language"] = merged["language"].fillna("unknown")
    # Drop 'unknown' bucket for final reporting (langdetect failure on short strings)
    rep = merged[merged["language"] != "unknown"].copy()
    g = (rep.groupby(["category", "attr", "language"])
            .agg(n_cells=("correct", "size"), accuracy=("correct", "mean"))
            .reset_index())
    return g, merged


def main():
    setup_logging()
    out_path = os.path.join(PROCESSED_DIR, "per_language_eval.parquet")

    cascade = load_cascade_valid()
    logger.info("TOTAL cascade-valid cells: %d", len(cascade))

    lang_lookup = build_lang_lookup()
    metrics, merged = aggregate(cascade, lang_lookup)

    # Headline: per-language totals (across cats × attrs)
    headline = (merged[merged["language"] != "unknown"]
                .groupby("language")
                .agg(n_cells=("correct", "size"), accuracy=("correct", "mean"))
                .reset_index()
                .sort_values("accuracy", ascending=False))
    logger.info("=" * 70)
    logger.info("Headline per-language accuracy (cascade-only, all cats):")
    for _, r in headline.iterrows():
        logger.info("  %-8s n=%5d  acc=%.4f (%.1f%%)",
                    r["language"], int(r["n_cells"]), r["accuracy"], 100 * r["accuracy"])
    logger.info("=" * 70)

    # Worst (cat, attr, lang) tuples
    worst = metrics[metrics["n_cells"] >= 10].sort_values("accuracy").head(10)
    logger.info("Bottom-10 (cat, attr, lang) with n>=10:")
    for _, r in worst.iterrows():
        logger.info("  %-9s %-25s %-6s n=%4d acc=%.3f",
                    r["category"], r["attr"], r["language"],
                    int(r["n_cells"]), r["accuracy"])

    metrics.to_parquet(out_path, index=False)
    logger.info("Saved → %s (%d rows)", out_path, len(metrics))

    # Headline TeX-friendly summary
    summary_path = os.path.join(PROCESSED_DIR, "per_language_headline.json")
    import json
    payload = {
        "source": "manual_gold_consensus.parquet (extended 2026-05-27) × cascade_preds_*_gold.parquet",
        "n_cells_total": int(len(merged[merged["language"] != "unknown"])),
        "n_cells_unknown_dropped": int((merged["language"] == "unknown").sum()),
        "per_language": headline.to_dict(orient="records"),
        "categories": list(CATEGORIES),
        "reported_langs": sorted(REPORTED_LANGS) + ["other"],
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved headline → %s", summary_path)


if __name__ == "__main__":
    main()
