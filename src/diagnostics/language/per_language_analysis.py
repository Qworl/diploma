"""
Per-language accuracy breakdown for the hybrid pipeline.

Detects language from product_name + ingredients_text via langdetect, then computes
accuracy/macro-F1 per (language, attribute). Reveals whether the multilingual
embedding model serves all languages equally — historically French (~60% of data)
gets the best signal.

Usage:
    python -m src.diagnostics.language.per_language_analysis --category pasta
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.common import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    setup_logging,
)
from src.eval.run_experiments import CATEGORY_CONFIG, load_ml_models, load_thresholds

# Languages we report explicitly; rest aggregated as "other"
REPORTED_LANGS = {"fr", "en", "de", "it", "es", "nl", "pt"}

# Deterministic detection across runs
DetectorFactory.seed = 0
logger = logging.getLogger(__name__)


def detect_lang(row: pd.Series) -> str:
    """Best-effort lang detection from product_name + ingredients_text. Returns ISO 639-1
    code or 'unknown'. Short strings are unreliable, so we concat.
    """
    parts = []
    for col in ("product_name", "ingredients_text"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    text = " ".join(parts)
    if len(text) < 8:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def predict_ml(test_df: pd.DataFrame, embeddings: np.ndarray, test_idx: np.ndarray,
               models: dict, attrs: list[str], thresholds: dict) -> pd.DataFrame:
    """Run ML layer only (no regex/Bayes/LLM) on test subset, return predictions DF."""
    rows = []
    for i, (df_idx, _) in enumerate(test_df.iterrows()):
        emb_idx = test_idx[i]
        X = embeddings[emb_idx:emb_idx + 1]
        out = {"_test_pos": i}
        for attr in attrs:
            xgb_key = f"{attr}_xgb"
            le_key = f"{attr}_le"
            if xgb_key not in models:
                out[f"{attr}_pred"] = None
                out[f"{attr}_conf"] = 0.0
                continue
            clf = models[xgb_key]
            proba = clf.predict_proba(X)[0]
            max_idx = int(proba.argmax())
            conf = float(proba[max_idx])
            threshold = thresholds.get(attr, DEFAULT_CONFIDENCE_THRESHOLD)
            if conf < threshold:
                out[f"{attr}_pred"] = None
                out[f"{attr}_conf"] = conf
                continue
            if le_key in models:
                value = models[le_key].inverse_transform([max_idx])[0]
            else:
                value = bool(max_idx)
            out[f"{attr}_pred"] = value
            out[f"{attr}_conf"] = conf
        rows.append(out)
    return pd.DataFrame(rows)


def aggregate_metrics(test_df: pd.DataFrame, preds: pd.DataFrame,
                       attrs: list[str]) -> pd.DataFrame:
    """For each (lang, attr) compute accuracy, macro_F1, coverage, n_samples."""
    test_df = test_df.reset_index(drop=True)
    test_df["_lang"] = test_df.apply(detect_lang, axis=1)
    test_df["_lang_bucket"] = test_df["_lang"].where(
        test_df["_lang"].isin(REPORTED_LANGS), "other"
    )
    merged = test_df.join(preds.set_index("_test_pos"), how="left")

    rows = []
    for lang, group in merged.groupby("_lang_bucket"):
        for attr in attrs:
            pred_col = f"{attr}_pred"
            if pred_col not in group.columns or attr not in group.columns:
                continue
            valid = group[[attr, pred_col]].dropna()
            n_total = len(group)
            n_covered = len(valid)
            if n_covered == 0:
                rows.append({"lang": lang, "attr": attr, "n_total": n_total,
                             "n_covered": 0, "coverage": 0.0,
                             "accuracy": None, "macro_f1": None})
                continue
            y_true = valid[attr].astype(str)
            y_pred = valid[pred_col].astype(str)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            rows.append({
                "lang": lang, "attr": attr,
                "n_total": n_total, "n_covered": n_covered,
                "coverage": n_covered / n_total if n_total else 0.0,
                "accuracy": float(acc), "macro_f1": float(f1),
            })
    return pd.DataFrame(rows)


def log_summary(metrics_df: pd.DataFrame, lang_counts: pd.Series, attrs: list[str]):
    logger.info("=" * 70)
    logger.info("Language distribution (test set):")
    for lang, n in lang_counts.items():
        logger.info("  %-10s %4d (%.1f%%)", lang, n, 100 * n / lang_counts.sum())
    logger.info("=" * 70)

    pivot = metrics_df.pivot(index="lang", columns="attr", values="accuracy")
    pivot = pivot.reindex(columns=attrs)
    logger.info("Accuracy by (language, attribute):")
    for line in pivot.round(3).fillna("—").to_string().split("\n"):
        logger.info("  %s", line)

    logger.info("=" * 70)
    cov_pivot = metrics_df.pivot(index="lang", columns="attr", values="coverage")
    cov_pivot = cov_pivot.reindex(columns=attrs)
    logger.info("Coverage by (language, attribute):")
    for line in (cov_pivot * 100).round(1).fillna(0).to_string().split("\n"):
        logger.info("  %s", line)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()))
    parser.add_argument("--output", default=None,
                        help="Path to save per-language metrics CSV (default: under datasets/processed/)")
    args = parser.parse_args()
    cfg = CATEGORY_CONFIG[args.category]

    ss_path = os.path.join(PROCESSED_DIR, cfg["silver_standard"])
    df = pd.read_parquet(ss_path)
    logger.info("Loaded %d rows from %s", len(df), ss_path)

    _, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    test_df = df.iloc[test_idx]
    logger.info("Test set: %d rows (aligned with training split)", len(test_df))

    emb_path = os.path.join(PROCESSED_DIR, cfg["embeddings_cache"])
    embeddings = np.load(emb_path)

    models = load_ml_models(args.category, cfg["ml_attrs"])
    thresholds = load_thresholds(args.category)
    if not models:
        logger.error("No ML models found for %s — train first via train_classifiers", args.category)
        return

    preds = predict_ml(test_df, embeddings, test_idx, models, cfg["ml_attrs"], thresholds)

    test_df = test_df.reset_index(drop=True)
    test_df["_lang"] = test_df.apply(detect_lang, axis=1)
    test_df["_lang_bucket"] = test_df["_lang"].where(
        test_df["_lang"].isin(REPORTED_LANGS), "other"
    )
    lang_counts = test_df["_lang_bucket"].value_counts()

    metrics = aggregate_metrics(test_df, preds, cfg["ml_attrs"])
    log_summary(metrics, lang_counts, cfg["ml_attrs"])

    out_path = args.output or os.path.join(
        PROCESSED_DIR, f"{args.category}_per_language_metrics.csv"
    )
    metrics.to_csv(out_path, index=False)
    logger.info("Saved per-language metrics → %s", out_path)


if __name__ == "__main__":
    main()
