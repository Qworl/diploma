"""Trek A1 — produce long-format parquet of validator signals on pasta gold.

For each (code, attr) where ``code`` is in the audited Trek D pasta gold,
this script re-runs the regex + ML layers (Bayes excluded — out of scope
per plan Task 0), records the four candidate validator signals, joins
audited manual labels, and writes a long-format parquet.

Usage
-----
    python -m src.eval.validator_comparison \\
        --gold datasets/manual_label/pasta_gold_250.csv \\
        --out  datasets/processed/validator_comparison_pasta.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from src.common import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MODELS_DIR,
    PROCESSED_DIR,
    setup_logging,
)
from src.eval.cascade_vs_audited_gold import (
    AUDITED_MODES,
    AUDITED_STATUSES,
    PASTA_ATTRS,
    _norm,
    _values_equal,
    build_audited_long,
    load_bayesian,
    load_ml_models,
    load_thresholds,
    regex_layer,
)
from src.eval.validator_mahalanobis import (
    fit_per_attr_mahalanobis,
    score_per_attr_mahalanobis,
)
from src.eval.validator_signals import (
    layer_disagreement_score,
    load_per_attr_ece,
    per_attr_ece_score,
    xgb_uncertainty_score,
)
from src.pipeline.regex.extractor import RegexExtractor

logger = logging.getLogger(__name__)


def _run_ml_with_signals(
    embeddings: np.ndarray,
    idx: int,
    ml_models: dict,
    thresholds: dict,
) -> dict[str, dict]:
    """Run ML on a single row, returning prediction + raw signals per attr.

    Returns mapping attr -> {"value", "confidence", "proba_max", "thresholded"}.
    ``thresholded`` is True iff the cell would survive the conf threshold.
    """
    X = embeddings[idx : idx + 1]
    out: dict[str, dict] = {}
    for attr in PASTA_ATTRS:
        xgb_key = f"{attr}_xgb"
        if xgb_key not in ml_models:
            out[attr] = {
                "value": None, "confidence": None, "proba_max": None,
                "thresholded": False,
            }
            continue
        clf = ml_models[xgb_key]
        proba = clf.predict_proba(X)[0]
        max_idx = int(proba.argmax())
        confidence = float(proba[max_idx])
        le_key = f"{attr}_le"
        if le_key in ml_models:
            value = ml_models[le_key].inverse_transform([max_idx])[0]
        else:
            value = bool(max_idx)
        threshold = thresholds.get(attr, DEFAULT_CONFIDENCE_THRESHOLD)
        out[attr] = {
            "value": value,
            "confidence": confidence,
            "proba_max": confidence,
            "thresholded": confidence >= threshold,
        }
    return out


def build_long_table(
    silver_df: pd.DataFrame,
    embeddings: np.ndarray,
    gold_codes: set[str],
    category: str = "pasta_stratified",
) -> pd.DataFrame:
    """Run cascade + record validator signals for rows with code in gold_codes."""
    rx = RegexExtractor()
    ml_models = load_ml_models(category, PASTA_ATTRS)
    thresholds = load_thresholds(category)

    # Fit Mahalanobis on the *complement* (silver rows NOT in gold) to avoid
    # leakage. Silver labels provide the class assignment per attr.
    complement_mask = (~silver_df["code"].astype(str).isin(gold_codes)).values
    X_train = embeddings[complement_mask]
    train_labels: dict[str, np.ndarray] = {}
    for attr in PASTA_ATTRS:
        col = attr  # silver columns are flat in pasta_stratified parquet
        if col in silver_df.columns:
            train_labels[attr] = silver_df.loc[complement_mask, col].values
    maha_fits = fit_per_attr_mahalanobis(X_train, train_labels, PASTA_ATTRS)
    logger.info("Fitted Mahalanobis for attrs: %s", sorted(maha_fits.keys()))

    ece_table = load_per_attr_ece(category, tuple(PASTA_ATTRS), MODELS_DIR)
    logger.info("Loaded per-attr ECE: %s", sorted(ece_table.keys()))

    # Subset to gold codes
    mask = silver_df["code"].astype(str).isin(gold_codes).values
    sub = silver_df.loc[mask].reset_index(drop=True)
    sub_emb = embeddings[mask]

    # Pre-compute Mahalanobis scores per attr for all sub-rows
    maha_scores = score_per_attr_mahalanobis(sub_emb, maha_fits)

    rows: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(sub.iterrows()):
        code = str(row.get("code"))
        regex_out = regex_layer(row, rx)  # {attr: (value, conf)}
        ml_out = _run_ml_with_signals(sub_emb, i, ml_models, thresholds)

        for attr in PASTA_ATTRS:
            regex_val = regex_out.get(attr, (None, None))[0]
            ml_info = ml_out[attr]
            ml_val = ml_info["value"] if ml_info["thresholded"] else None
            xgb_max = ml_info["proba_max"]
            cascade_val: Any = None
            cascade_layer = "none"
            if regex_val is not None:
                cascade_val, cascade_layer = regex_val, "regex"
            elif ml_val is not None:
                cascade_val, cascade_layer = ml_val, "ml"
            rows.append({
                "code": code,
                "attr": attr,
                "cascade_pred": None if cascade_val is None else str(cascade_val),
                "cascade_layer": cascade_layer,
                "regex_pred": None if regex_val is None else str(regex_val),
                "ml_pred": None if ml_val is None else str(ml_val),
                "xgb_max_prob": xgb_max,
                "xgb_uncertainty": xgb_uncertainty_score(
                    np.array([xgb_max, 1 - xgb_max]) if xgb_max is not None else None
                ),
                "mahalanobis": (
                    float(maha_scores[attr][i]) if attr in maha_scores else None
                ),
                "layer_disagree": layer_disagreement_score(regex_val, ml_info["value"]),
                "ece_attr": per_attr_ece_score(attr, ece_table),
            })
    return pd.DataFrame(rows)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold", default="datasets/manual_label/pasta_gold_250.csv",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(PROCESSED_DIR, "validator_comparison_pasta.parquet"),
    )
    parser.add_argument("--category", default="pasta_stratified")
    args = parser.parse_args()

    gold = pd.read_csv(args.gold, dtype={"code": str})
    gold_codes = set(gold["code"].astype(str))
    logger.info("Gold: %d products", len(gold_codes))

    silver_path = os.path.join(PROCESSED_DIR, f"{args.category}_silver_standard.parquet")
    emb_path = os.path.join(PROCESSED_DIR, f"{args.category}_embeddings.npy")
    silver = pd.read_parquet(silver_path).reset_index(drop=True)
    silver["code"] = silver["code"].astype(str)
    emb = np.load(emb_path)
    if len(emb) != len(silver):
        raise RuntimeError(f"Embeddings/silver length mismatch: {len(emb)} vs {len(silver)}")

    long_signals = build_long_table(silver, emb, gold_codes, category=args.category)

    # Join audited manual labels
    audited_long = build_audited_long(gold)
    joined = audited_long.merge(long_signals, on=["code", "attr"], how="left")

    # Normalise types
    for col in ("manual_value", "silver_value", "cascade_pred"):
        joined[col] = joined[col].apply(
            lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
        )

    joined["has_manual"] = joined["manual_value"].notna() & joined["status"].isin(AUDITED_STATUSES)
    joined["is_error"] = joined.apply(
        lambda r: bool(
            r["has_manual"]
            and r["cascade_pred"] is not None
            and not _values_equal(r["cascade_pred"], r["manual_value"])
        ),
        axis=1,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joined.to_parquet(args.out, index=False)
    logger.info("Wrote %d rows → %s", len(joined), args.out)

    # Mini-summary
    n_err = int(joined["is_error"].sum())
    n_with_manual = int(joined["has_manual"].sum())
    logger.info("is_error=%d / has_manual=%d (%.1f%%)",
                n_err, n_with_manual, 100 * n_err / max(n_with_manual, 1))


if __name__ == "__main__":
    main()
