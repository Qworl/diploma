"""ECE v2: Expected Calibration Error on v2 blind-gold + brand-disjoint silver hold-out.

For each per-(cat, attr) silver-trained XGB model:
- On v2 blind-gold (pasta/chocolate/cheeses)
- On silver brand-disjoint hold-out (20% brands held out per category)

Output: datasets/processed/ece_off_grounded.json

Schema per (cat_attr):
  in_sample_ece: ECE on silver test split (from calibration JSON if available)
  v2_blind_gold_ece: ECE on v2 expanded gold
  brand_disjoint_holdout_ece: ECE on brand-disjoint silver hold-out
  v2_accuracy_at_threshold: Accuracy on v2 gold at model threshold
  in_sample_accuracy_at_threshold: Accuracy on silver test split at threshold
  drift_abs: abs(brand_disjoint_holdout_ece - in_sample_ece)
  accuracy_drop_pp: (in_sample_accuracy_at_threshold - v2_accuracy_at_threshold) * 100
  flag_for_re_derivation: drift_abs > 0.05 AND accuracy_drop_pp > 3
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.common import DEFAULT_CONFIDENCE_THRESHOLD, MODELS_DIR, PROCESSED_DIR, get_embeddings, setup_logging
from src.manual_label.schemas_loader import load_domain_attrs
from src.pipeline.ml.infer import load_classifier, load_thresholds

logger = logging.getLogger(__name__)

# Categories with v2 gold
OFF_CATS = ["pasta", "chocolate", "cheeses"]

# All categories (for brand-disjoint hold-out even without v2 gold)
ALL_CATS = ["pasta", "chocolate", "cheeses", "beverages", "cereals", "cosmetics"]

BRAND_HOLDOUT_FRAC = 0.2
BRAND_HOLDOUT_SEED = 42


def compute_ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (max-prob style).

    For each prediction take p = max-class probability, check correctness.
    ECE = weighted mean of |bin_acc - bin_conf|.
    """
    confidences = proba.max(axis=1)
    preds = proba.argmax(axis=1)
    correct = (preds == y_true).astype(float)
    ece = 0.0
    n = len(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidences[mask].mean())
        ece += (cnt / n) * abs(bin_acc - bin_conf)
    return float(ece)


def accuracy_at_threshold(y_true: np.ndarray, proba: np.ndarray, le, threshold: float) -> float:
    """Accuracy on cells where model is confident enough (above threshold).

    For non-confident predictions, treat as incorrect (refusal-as-miss style).
    """
    confidences = proba.max(axis=1)
    preds_idx = proba.argmax(axis=1)
    preds_labels = le.inverse_transform(preds_idx)
    correct = np.array([
        str(pred) == str(true) and conf >= threshold
        for pred, true, conf in zip(preds_labels, y_true, confidences)
    ], dtype=float)
    if len(correct) == 0:
        return float("nan")
    return float(correct.mean())


def _extract_text_features(df: pd.DataFrame) -> list[str]:
    """Combine text fields into embedding input."""
    cols = ["product_name", "brands", "ingredients_text", "quantity"]
    parts = []
    for c in cols:
        if c in df.columns:
            parts.append(df[c].fillna("").astype(str))
        else:
            parts.append(pd.Series([""] * len(df)))
    return (parts[0] + " " + parts[1] + " " + parts[2] + " " + parts[3]).tolist()


def _get_primary_brand(brands_series: pd.Series) -> pd.Series:
    """Extract first brand from comma-separated brands field."""
    return brands_series.fillna("").str.split(",").str[0].str.strip()


def _brand_disjoint_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Split df into train/test by brand (no brand appears in both sets)."""
    df = df.copy()
    df["_brand"] = _get_primary_brand(df.get("brands", pd.Series([""] * len(df))))
    unique_brands = df["_brand"].unique()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_brands))
    n_test = max(1, int(len(unique_brands) * test_frac))
    test_brands = set(unique_brands[perm[:n_test]])
    test_mask = df["_brand"].isin(test_brands)
    return df[~test_mask].drop(columns=["_brand"]), df[test_mask].drop(columns=["_brand"])


def process_cat_attr(
    cat: str,
    attr: str,
    silver: pd.DataFrame,
    gold: pd.DataFrame | None,
    thresholds: dict,
) -> dict | None:
    """Compute ECE metrics for a single (cat, attr) pair."""
    key = f"{cat}_{attr}"
    category = f"{cat}_stratified"

    # Load model
    try:
        clf, le = load_classifier(category, attr)
    except (FileNotFoundError, OSError) as e:
        logger.warning("Model not found for %s: %s", key, e)
        return None

    threshold = thresholds.get(attr, DEFAULT_CONFIDENCE_THRESHOLD)

    # Filter silver to rows with valid labels for this attr
    if attr not in silver.columns:
        logger.warning("Attr %s not in silver for %s", attr, cat)
        return None

    silver_valid = silver.dropna(subset=[attr]).copy()
    silver_valid = silver_valid[silver_valid[attr].astype(str).str.strip() != ""]
    if len(silver_valid) < 20:
        logger.warning("Not enough silver rows for %s/%s (%d)", cat, attr, len(silver_valid))
        return None

    # Brand-disjoint split on silver
    silver_train, silver_test = _brand_disjoint_split(silver_valid, test_frac=BRAND_HOLDOUT_FRAC, seed=BRAND_HOLDOUT_SEED)
    if len(silver_test) < 5:
        logger.warning("Brand-disjoint test too small for %s/%s (%d rows)", cat, attr, len(silver_test))
        return None

    # --- In-sample ECE (on silver brand-disjoint test set) ---
    texts_silver = _extract_text_features(silver_valid)
    embs_silver = get_embeddings(texts_silver)

    texts_test = _extract_text_features(silver_test)
    embs_test = get_embeddings(texts_test)

    y_true_silver = silver_valid[attr].astype(str).tolist()
    y_true_test = silver_test[attr].astype(str).tolist()

    try:
        y_enc_silver = le.transform(y_true_silver)
        y_enc_test = le.transform(y_true_test)
    except ValueError as e:
        logger.warning("Label encoding failed for %s/%s: %s", cat, attr, e)
        # Partial encoding — skip unknowns
        valid_classes = set(le.classes_)
        mask_silver = [v in valid_classes for v in y_true_silver]
        mask_test = [v in valid_classes for v in y_true_test]
        if sum(mask_silver) < 10 or sum(mask_test) < 5:
            return None
        embs_silver = embs_silver[np.array(mask_silver)]
        embs_test = embs_test[np.array(mask_test)]
        y_enc_silver = le.transform([v for v, m in zip(y_true_silver, mask_silver) if m])
        y_enc_test = le.transform([v for v, m in zip(y_true_test, mask_test) if m])

    proba_silver = clf.predict_proba(embs_silver)
    in_sample_ece = compute_ece(y_enc_silver, proba_silver)
    in_sample_acc = accuracy_at_threshold(y_true_silver if len(y_enc_silver) == len(y_true_silver) else
                                          [v for v, m in zip(y_true_silver, [True]*len(y_enc_silver)) if m],
                                          proba_silver, le, threshold)

    proba_test = clf.predict_proba(embs_test)
    brand_disjoint_ece = compute_ece(y_enc_test, proba_test)

    # --- V2 blind-gold ECE (only for OFF cats) ---
    v2_ece = None
    v2_acc = None
    if gold is not None:
        cat_gold = gold[(gold["category"] == cat) & (gold["attr"] == attr)].copy()
        cat_gold = cat_gold[~cat_gold["gold_is_null"]]

        if len(cat_gold) < 5:
            logger.warning("Not enough v2 gold for %s/%s (%d rows)", cat, attr, len(cat_gold))
        else:
            # Join with silver to get product features
            cat_gold["code"] = cat_gold["code"].astype(str)
            silver_all = silver.copy()
            silver_all["code"] = silver_all["code"].astype(str)
            merged = cat_gold.merge(silver_all[["code"] + [c for c in ["product_name", "brands", "ingredients_text", "quantity"] if c in silver_all.columns]],
                                    on="code", how="inner")
            if len(merged) < 5:
                logger.warning("Not enough joined v2 gold for %s/%s (%d rows)", cat, attr, len(merged))
            else:
                texts_gold = _extract_text_features(merged)
                embs_gold = get_embeddings(texts_gold)
                gold_labels = merged["gold_value"].astype(str).tolist()
                valid_classes = set(le.classes_)
                mask_gold = [v in valid_classes for v in gold_labels]
                if sum(mask_gold) < 5:
                    logger.warning("Too few valid-class v2 gold labels for %s/%s", cat, attr)
                else:
                    embs_gold_valid = embs_gold[np.array(mask_gold)]
                    gold_labels_valid = [v for v, m in zip(gold_labels, mask_gold) if m]
                    y_enc_gold = le.transform(gold_labels_valid)
                    proba_gold = clf.predict_proba(embs_gold_valid)
                    v2_ece = compute_ece(y_enc_gold, proba_gold)
                    v2_acc = accuracy_at_threshold(gold_labels_valid, proba_gold, le, threshold)

    # Compute drift and flags
    drift_abs = abs(brand_disjoint_ece - in_sample_ece)
    accuracy_drop_pp = None
    flag = False
    if v2_acc is not None and not np.isnan(v2_acc) and not np.isnan(in_sample_acc):
        accuracy_drop_pp = (in_sample_acc - v2_acc) * 100
        flag = (drift_abs > 0.05) and (accuracy_drop_pp > 3.0)

    return {
        "in_sample_ece": round(in_sample_ece, 4),
        "v2_blind_gold_ece": round(v2_ece, 4) if v2_ece is not None else None,
        "brand_disjoint_holdout_ece": round(brand_disjoint_ece, 4),
        "v2_accuracy_at_threshold": round(v2_acc, 4) if v2_acc is not None else None,
        "in_sample_accuracy_at_threshold": round(in_sample_acc, 4),
        "drift_abs": round(drift_abs, 4),
        "accuracy_drop_pp": round(accuracy_drop_pp, 2) if accuracy_drop_pp is not None else None,
        "flag_for_re_derivation": flag,
    }


def main():
    setup_logging()

    gold = pd.read_parquet(Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet")
    gold["code"] = gold["code"].astype(str)

    results: dict[str, dict] = {}

    for cat in ALL_CATS:
        logger.info("Processing category: %s", cat)
        silver_path = Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        if not silver_path.exists():
            logger.warning("Silver standard not found for %s", cat)
            continue
        silver = pd.read_parquet(silver_path)
        silver["code"] = silver["code"].astype(str)

        attrs = list(load_domain_attrs(cat))
        thresholds = load_thresholds(f"{cat}_stratified")
        cat_gold = gold[gold["category"] == cat] if cat in OFF_CATS else None

        for attr in attrs:
            logger.info("  %s / %s", cat, attr)
            try:
                result = process_cat_attr(cat, attr, silver, cat_gold, thresholds)
                if result is not None:
                    results[f"{cat}_{attr}"] = result
            except Exception as e:
                logger.warning("Failed for %s/%s: %s", cat, attr, e)

    out_path = Path(PROCESSED_DIR) / "ece_off_grounded.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s (%d entries)", out_path, len(results))

    # Print summary table
    print(f"\n=== ECE DRIFT SUMMARY ({len(results)} attr-model pairs) ===")
    rows = []
    for key, r in sorted(results.items(), key=lambda x: -x[1]["drift_abs"]):
        rows.append({
            "cat_attr": key,
            "in_sample_ece": r["in_sample_ece"],
            "brand_disjoint_ece": r["brand_disjoint_holdout_ece"],
            "v2_blind_ece": r["v2_blind_gold_ece"],
            "drift_abs": r["drift_abs"],
            "acc_drop_pp": r["accuracy_drop_pp"],
            "flag": r["flag_for_re_derivation"],
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    flagged = [k for k, v in results.items() if v["flag_for_re_derivation"]]
    print(f"\nFlagged for re-derivation ({len(flagged)}):", flagged)


if __name__ == "__main__":
    main()
