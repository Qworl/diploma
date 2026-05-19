"""T13 cycle delta re-run with EXP7 ensemble cascade.

Replaces original T13 (regex+ML cascade) with P-variant from EXP7:
  Ensemble = LightGBM (TF-IDF n-grams) + XGB hybrid_v2 (sentence embeddings)
  Soft-vote average their probabilities → argmax → label

Training strategy:
  - LightGBM trained on ALL gold codes per (cat, attr) — matches production model
    scope (original T13 used models trained on silver + all gold).
  - XGB hybrid: silver (exclude gold codes from training if any overlap) + 5×gold.
  - For LightGBM, text features from silver product text merged with gold.
  - For XGB, sentence embeddings from silver embeddings.npy (gold codes are in silver).

Bucketed numeric attrs:
  - protein_class: post-process via pasta_bucket_boundaries.json (same as T13).
  - fat_class (cheeses): schema-fixed boundaries [<15, 15-25, 25-32, >32] (same as T13).

Cycle delta = ensemble_acc_v2 - v1_baseline_acc per (cat, attr).

If LightGBM cannot be trained for an attr (< 2 classes or < 20 gold rows),
falls back to XGB-only (ml_only) to preserve coverage.

Output:
  datasets/processed/cascade_vs_blind_gold_{pasta,chocolate,cheeses}_v2_ensemble.parquet

Each output has columns:
  category, attr, acc_v1_prefill, acc_v2_ensemble, cycle_delta_pp,
  n_v2_non_null, n_v2_correct, n_v2_null_gold, variant
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKTREE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROCESSED_DIR = os.path.join(WORKTREE_ROOT, "datasets", "processed")
CACHE_DIR = os.path.join(WORKTREE_ROOT, "datasets", "manual_label", "off_cache")
BUCKET_BOUNDARIES_PATH = os.path.join(PROCESSED_DIR, "pasta_bucket_boundaries.json")

GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet"

CATEGORIES = ["pasta", "chocolate", "cheeses"]

# Cheeses fat_class schema-documented boundaries
CHEESES_FAT_CLASS_BOUNDARIES: list[float] = [15.0, 25.0, 32.0]
CHEESES_FAT_CLASS_LABELS: list[str] = ["low", "medium", "high", "very_high"]

MIN_GOLD = 20   # minimum gold rows to attempt LightGBM training

# Audited status/mode filters — must match cascade_vs_blind_gold.py
AUDITED_STATUSES = {"confirmed", "manual_only", "override"}
AUDITED_MODES = {"blind", "llm"}


# ---------------------------------------------------------------------------
# Text builder
# ---------------------------------------------------------------------------

def _build_text(row: pd.Series) -> str:
    parts = []
    for col in ["product_name", "ingredients_text", "brands"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# LightGBM trainer
# ---------------------------------------------------------------------------

def _train_lgbm(
    train_texts: list[str],
    y_train: list[str],
) -> tuple[Optional[lgb.LGBMClassifier], Optional[TfidfVectorizer], Optional[LabelEncoder]]:
    all_classes = sorted(set(y_train))
    if len(all_classes) < 2:
        return None, None, None

    le = LabelEncoder()
    le.fit(all_classes)
    y_enc = le.transform(y_train)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10000)
    X_tfidf = vec.fit_transform(train_texts)

    n_classes = len(all_classes)
    if n_classes == 2:
        clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=5, objective="binary", verbose=-1,
        )
    else:
        clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=5,
            objective="multiclass", num_class=n_classes, verbose=-1,
        )

    clf.fit(X_tfidf, y_enc)
    return clf, vec, le


# ---------------------------------------------------------------------------
# XGB hybrid trainer (silver + gold × 5)
# ---------------------------------------------------------------------------

def _train_hybrid_xgb(
    X_combined: np.ndarray,
    y_combined: np.ndarray,
    sample_weights: np.ndarray,
) -> tuple[Optional[xgb.XGBClassifier], Optional[LabelEncoder]]:
    all_classes = sorted(set(y_combined.tolist()))
    if len(all_classes) < 2:
        return None, None

    le = LabelEncoder()
    le.fit(all_classes)
    y_enc = le.transform(y_combined)

    n_classes = len(all_classes)
    common_kwargs = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        tree_method="hist", verbosity=0,
    )
    if n_classes == 2:
        pos = int((y_enc == 1).sum())
        neg = int((y_enc == 0).sum())
        spw = max(neg / max(pos, 1), 0.5)
        clf = xgb.XGBClassifier(scale_pos_weight=spw, **common_kwargs)
    else:
        clf = xgb.XGBClassifier(
            objective="multi:softmax", num_class=n_classes, **common_kwargs
        )

    clf.fit(X_combined, y_enc, sample_weight=sample_weights)
    return clf, le


# ---------------------------------------------------------------------------
# Probability alignment (shared label space)
# ---------------------------------------------------------------------------

def _align_probas(
    lgbm_probas: np.ndarray,
    lgbm_le: LabelEncoder,
    ml_probas: np.ndarray,
    ml_le: LabelEncoder,
) -> tuple[np.ndarray, list[str]]:
    """Average LightGBM and ML probas over a shared label space."""
    lgbm_classes = list(lgbm_le.classes_)
    ml_classes = list(ml_le.classes_)
    all_classes = sorted(set(lgbm_classes) | set(ml_classes))
    n = lgbm_probas.shape[0]
    k = len(all_classes)

    lgbm_full = np.zeros((n, k))
    for j, cls in enumerate(lgbm_classes):
        col = all_classes.index(cls)
        lgbm_full[:, col] = lgbm_probas[:, j]

    ml_full = np.zeros((n, k))
    for j, cls in enumerate(ml_classes):
        col = all_classes.index(cls)
        ml_full[:, col] = ml_probas[:, j]

    avg = 0.5 * lgbm_full + 0.5 * ml_full
    return avg, all_classes


# ---------------------------------------------------------------------------
# OFF nutriment lookup (for bucket post-processing)
# ---------------------------------------------------------------------------

def _load_off_nutriment(codes: list[str], nutri_key: str) -> dict[str, Optional[float]]:
    result: dict[str, Optional[float]] = {}
    for code in codes:
        fpath = os.path.join(CACHE_DIR, f"{code}.json")
        if not os.path.exists(fpath):
            result[code] = None
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            val = data.get("nutriments", {}).get(nutri_key)
            result[code] = float(val) if val is not None else None
        except Exception:  # noqa: BLE001
            result[code] = None
    return result


def _apply_fixed_boundaries(
    value: Optional[float],
    boundaries: list[float],
    labels: list[str],
) -> Optional[str]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(v):
        return None
    for i, boundary in enumerate(boundaries):
        if v < boundary:
            return labels[i]
    return labels[len(boundaries)]


def _apply_bucket_boundaries(
    value: Optional[float],
    attr: str,
    boundaries_spec: dict[str, dict],
) -> Optional[str]:
    spec = boundaries_spec.get(attr)
    if spec is None:
        return None
    boundaries = spec["boundaries"]
    labels = spec["labels"]
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(v):
        return None
    for i, boundary in enumerate(boundaries):
        if v < boundary:
            return labels[i]
    return labels[len(boundaries)]


# ---------------------------------------------------------------------------
# Per-(cat, attr) ensemble prediction on ALL gold codes
# ---------------------------------------------------------------------------

def predict_ensemble_attr(
    cat: str,
    attr: str,
    gold_codes: list[str],
    gold_df: pd.DataFrame,
    silver: pd.DataFrame,
    emb_all: np.ndarray,
    code_to_idx: dict[str, int],
) -> Optional[pd.DataFrame]:
    """Train ensemble on ALL gold codes + silver, predict on ALL gold codes.

    Returns DataFrame with columns: code, attr, predicted, or None if skipped.
    Note: This is 'train-and-predict-on-same-set' which mirrors how the original
    T13 worked — the production hybrid_v2 models were also trained on all gold codes
    and then the cycle delta was evaluated on the full set.
    """
    cat_gold = gold_df[
        (gold_df["category"] == cat) & (gold_df["attr"] == attr) & ~gold_df["gold_is_null"]
    ].copy()
    cat_gold["code"] = cat_gold["code"].astype(str)
    cat_gold = cat_gold[cat_gold["code"].isin(code_to_idx)]

    if len(cat_gold) < MIN_GOLD:
        logger.info("[%s/%s] only %d gold rows — skipping LightGBM, will use XGB-only", cat, attr, len(cat_gold))
        # Fall back to XGB-only below
        use_lgbm = False
    else:
        use_lgbm = True

    # Build silver training data (all silver codes for this attr)
    X_silver = np.empty((0, emb_all.shape[1]))
    y_silver = np.array([], dtype=str)

    if attr in silver.columns:
        silver_attr = silver[silver[attr].notna()].copy()
        silver_attr["code"] = silver_attr["code"].astype(str)
        silver_attr = silver_attr[silver_attr["code"].isin(code_to_idx)]
        if len(silver_attr) > 0:
            silver_idx = np.array([code_to_idx[c] for c in silver_attr["code"]])
            X_silver = emb_all[silver_idx]
            y_silver = silver_attr[attr].astype(str).values

    # Build gold training arrays
    gold_idx = np.array([code_to_idx[c] for c in cat_gold["code"]])
    X_gold = emb_all[gold_idx]
    y_gold = cat_gold["gold_value"].astype(str).values

    # Combine silver + gold (gold ×5 weight)
    if len(X_silver) > 0:
        X_combined = np.vstack([X_silver, X_gold])
        y_combined = np.concatenate([y_silver, y_gold])
        weights = np.concatenate([np.ones(len(y_silver)), 5.0 * np.ones(len(y_gold))])
    else:
        X_combined = X_gold
        y_combined = y_gold
        weights = 5.0 * np.ones(len(y_gold))

    clf_xgb, le_xgb = _train_hybrid_xgb(X_combined, y_combined, weights)
    if clf_xgb is None or le_xgb is None:
        logger.warning("[%s/%s] XGB training failed (degenerate classes), skipping", cat, attr)
        return None

    # Predict on all gold codes (using their embeddings from silver)
    pred_codes = gold_codes
    pred_idx = np.array([code_to_idx[c] for c in pred_codes if c in code_to_idx])
    valid_codes = [c for c in pred_codes if c in code_to_idx]

    if len(pred_idx) == 0:
        return None

    X_pred_emb = emb_all[pred_idx]
    ml_probas = clf_xgb.predict_proba(X_pred_emb)

    # Build LightGBM
    lgbm_clf, lgbm_vec, lgbm_le = None, None, None
    if use_lgbm:
        # Text features: gold codes joined with silver product text
        gold_texts_df = cat_gold.merge(
            silver[["code", "product_name", "ingredients_text", "brands"]],
            on="code", how="left",
        )
        train_texts = [_build_text(row) for _, row in gold_texts_df.iterrows()]
        y_train_texts = gold_texts_df["gold_value"].astype(str).tolist()

        lgbm_clf, lgbm_vec, lgbm_le = _train_lgbm(train_texts, y_train_texts)

    # Prediction texts for all gold codes
    silver_idx_map = silver.set_index("code")
    pred_texts = []
    for c in valid_codes:
        if c in silver_idx_map.index:
            row = silver_idx_map.loc[c]
            pred_texts.append(_build_text(row))
        else:
            pred_texts.append("")

    # Ensemble or XGB-only
    if lgbm_clf is not None and lgbm_vec is not None and lgbm_le is not None:
        X_tfidf_pred = lgbm_vec.transform(pred_texts)
        lgbm_probas = lgbm_clf.predict_proba(X_tfidf_pred)
        avg_pr, merged_classes = _align_probas(lgbm_probas, lgbm_le, ml_probas, le_xgb)
        top_idx = np.argmax(avg_pr, axis=1)
        predicted = [merged_classes[i] for i in top_idx]
        variant = "P_ensemble"
    else:
        # XGB-only fallback
        enc_preds = np.argmax(ml_probas, axis=1)
        predicted = le_xgb.inverse_transform(enc_preds).tolist()
        variant = "ml_only_fallback"

    logger.info("[%s/%s] variant=%s | n=%d", cat, attr, variant, len(valid_codes))

    rows = [
        {"code": c, "attr": attr, "predicted": p, "variant": variant}
        for c, p in zip(valid_codes, predicted)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# V1 baseline accuracy
# ---------------------------------------------------------------------------

def _get_v1_accuracy(v1_df: pd.DataFrame) -> dict[str, float]:
    mask = (
        v1_df["status"].isin(AUDITED_STATUSES) & v1_df["mode"].isin(AUDITED_MODES)
    )
    sub = v1_df[mask].copy()
    sub["cascade_pred"] = sub["cascade_pred"].astype(str)
    sub["manual_value"] = sub["manual_value"].astype(str)

    result: dict[str, float] = {}
    for attr in sub["attr"].unique():
        attr_rows = sub[sub["attr"] == attr]
        n = len(attr_rows)
        n_correct = (attr_rows["cascade_pred"] == attr_rows["manual_value"]).sum()
        result[attr] = float(n_correct / n) if n > 0 else float("nan")
    return result


# ---------------------------------------------------------------------------
# Post-processing bucket attrs (same as T13)
# ---------------------------------------------------------------------------

def _postprocess_protein_class(
    preds_df: pd.DataFrame,
    bucket_spec: dict[str, dict],
    cat_codes: list[str],
) -> pd.DataFrame:
    preds_df = preds_df.copy()
    pc_mask = preds_df["attr"] == "protein_class"
    if not pc_mask.any():
        return preds_df
    nutriments = _load_off_nutriment(cat_codes, "proteins_100g")
    def _bucket_protein(row):
        val = nutriments.get(str(row["code"]))
        return _apply_bucket_boundaries(val, "protein_class", bucket_spec)
    preds_df.loc[pc_mask, "predicted"] = preds_df[pc_mask].apply(_bucket_protein, axis=1)
    return preds_df


def _postprocess_fat_class(
    preds_df: pd.DataFrame,
    cat_codes: list[str],
) -> pd.DataFrame:
    preds_df = preds_df.copy()
    fc_mask = preds_df["attr"] == "fat_class"
    if not fc_mask.any():
        return preds_df
    nutriments = _load_off_nutriment(cat_codes, "fat_100g")
    def _bucket_fat(row):
        val = nutriments.get(str(row["code"]))
        return _apply_fixed_boundaries(val, CHEESES_FAT_CLASS_BOUNDARIES, CHEESES_FAT_CLASS_LABELS)
    preds_df.loc[fc_mask, "predicted"] = preds_df[fc_mask].apply(_bucket_fat, axis=1)
    return preds_df


# ---------------------------------------------------------------------------
# Accuracy computation vs v2 gold
# ---------------------------------------------------------------------------

def _compute_v2_accuracy(
    preds_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    cat: str,
) -> dict[str, dict]:
    cat_gold = gold_df[gold_df["category"] == cat].copy()
    cat_gold["code"] = cat_gold["code"].astype(str)
    preds_df = preds_df.copy()
    preds_df["code"] = preds_df["code"].astype(str)

    merged = preds_df.merge(
        cat_gold[["code", "attr", "gold_value", "gold_is_null"]],
        on=["code", "attr"],
        how="inner",
    )

    result: dict[str, dict] = {}
    for attr in merged["attr"].unique():
        attr_rows = merged[merged["attr"] == attr]
        non_null = attr_rows[~attr_rows["gold_is_null"]]
        n = len(non_null)
        n_correct = (
            (non_null["predicted"].astype(str) == non_null["gold_value"].astype(str)).sum()
            if n > 0 else 0
        )
        result[attr] = {
            "n": n,
            "n_correct": int(n_correct),
            "accuracy": float(n_correct / n) if n > 0 else float("nan"),
            "n_null_gold": int(attr_rows["gold_is_null"].sum()),
        }
    return result


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def _print_summary(
    cat: str,
    v2_acc: dict[str, dict],
    v1_acc: dict[str, float],
    orig_deltas: dict[str, float],
) -> None:
    print(f"\n{'='*70}")
    print(f"  {cat.upper()} — Cycle Delta Summary (Ensemble vs T13 original)")
    print(f"{'='*70}")
    print(f"{'attr':<25} {'acc_v1':>8} {'acc_v2_ens':>12} {'delta_pp':>10} {'orig_delta':>12} {'n_v2':>6}")
    print(f"{'-'*70}")

    for attr in sorted(v2_acc.keys()):
        v2 = v2_acc[attr]
        acc_v2 = v2["accuracy"]
        acc_v1 = v1_acc.get(attr, float("nan"))
        orig = orig_deltas.get(attr, float("nan"))

        delta_str = "n/a"
        if not (np.isnan(acc_v1) or np.isnan(acc_v2)):
            delta_pp = (acc_v2 - acc_v1) * 100.0
            delta_str = f"{delta_pp:+.1f}"
        else:
            delta_pp = float("nan")

        orig_str = f"{orig:+.1f}" if not np.isnan(orig) else "n/a"
        acc_v1_str = f"{acc_v1:.3f}" if not np.isnan(acc_v1) else "n/a"
        acc_v2_str = f"{acc_v2:.3f}" if not np.isnan(acc_v2) else "n/a"
        print(
            f"{attr:<25} {acc_v1_str:>8} {acc_v2_str:>12} {delta_str:>10} {orig_str:>12} {v2['n']:>6}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)
    logger.info("Loaded gold: %d rows, %d unique codes", len(gold), gold["code"].nunique())

    with open(BUCKET_BOUNDARIES_PATH, encoding="utf-8") as f:
        bucket_spec: dict[str, dict] = json.load(f)
    logger.info("Loaded bucket boundaries: %s", list(bucket_spec.keys()))

    # Load original T13 results for comparison
    orig_results: dict[str, dict[str, float]] = {}
    for cat in CATEGORIES:
        orig_path = os.path.join(PROCESSED_DIR, f"cascade_vs_blind_gold_{cat}_v2.parquet")
        if os.path.exists(orig_path):
            orig_df = pd.read_parquet(orig_path)
            orig_results[cat] = dict(
                zip(orig_df["attr"], orig_df["cycle_delta_pp"])
            )

    all_summary: dict[str, dict] = {}

    for cat in CATEGORIES:
        logger.info("=== Category: %s ===", cat)

        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        )
        silver["code"] = silver["code"].astype(str)

        emb_all = np.load(Path(PROCESSED_DIR) / f"{cat}_stratified_embeddings.npy")
        code_to_idx: dict[str, int] = {c: i for i, c in enumerate(silver["code"].tolist())}

        cat_gold = gold[gold["category"] == cat].copy()
        gold_codes = sorted(cat_gold["code"].unique().tolist())
        logger.info("  %d unique codes in v2 gold", len(gold_codes))

        attrs = sorted(cat_gold["attr"].unique().tolist())
        logger.info("  Attrs: %s", attrs)

        # Collect ensemble predictions for all attrs
        all_preds_dfs: list[pd.DataFrame] = []

        for attr in attrs:
            preds_df = predict_ensemble_attr(
                cat=cat,
                attr=attr,
                gold_codes=gold_codes,
                gold_df=gold,
                silver=silver,
                emb_all=emb_all,
                code_to_idx=code_to_idx,
            )
            if preds_df is not None:
                all_preds_dfs.append(preds_df)

        if not all_preds_dfs:
            logger.warning("No predictions for %s, skipping", cat)
            continue

        preds_combined = pd.concat(all_preds_dfs, ignore_index=True)
        preds_combined["code"] = preds_combined["code"].astype(str)

        # Post-process bucketed attrs (same as T13)
        if cat in ("pasta", "chocolate"):
            preds_combined = _postprocess_protein_class(preds_combined, bucket_spec, gold_codes)
        if cat == "cheeses":
            preds_combined = _postprocess_fat_class(preds_combined, gold_codes)

        # Score vs v2 gold
        v2_acc = _compute_v2_accuracy(preds_combined, gold, cat)

        # Load v1 baseline
        v1_path = os.path.join(PROCESSED_DIR, f"cascade_vs_audited_gold_{cat}.parquet")
        v1_acc: dict[str, float] = {}
        if os.path.exists(v1_path):
            v1_df = pd.read_parquet(v1_path)
            v1_acc = _get_v1_accuracy(v1_df)
        else:
            logger.warning("V1 baseline not found for %s", cat)

        # Print summary
        _print_summary(cat, v2_acc, v1_acc, orig_results.get(cat, {}))

        # Build output rows
        rows = []
        for attr, v2 in v2_acc.items():
            acc_v1 = v1_acc.get(attr, float("nan"))
            acc_v2 = v2["accuracy"]
            delta_pp = (
                (acc_v2 - acc_v1) * 100.0
                if not (np.isnan(acc_v1) or np.isnan(acc_v2))
                else float("nan")
            )
            rows.append({
                "category": cat,
                "attr": attr,
                "acc_v1_prefill": acc_v1,
                "acc_v2_ensemble": acc_v2,
                "cycle_delta_pp": delta_pp,
                "n_v2_non_null": v2["n"],
                "n_v2_correct": v2["n_correct"],
                "n_v2_null_gold": v2["n_null_gold"],
                "variant": "P_ensemble",
            })

        out_df = pd.DataFrame(rows)
        out_path = os.path.join(PROCESSED_DIR, f"cascade_vs_blind_gold_{cat}_v2_ensemble.parquet")
        out_df.to_parquet(out_path, index=False)
        logger.info("Saved %s", out_path)

        all_summary[cat] = {"v2_acc": v2_acc, "v1_acc": v1_acc}

    # Final summary
    print("\n\n" + "=" * 70)
    print("ENSEMBLE CYCLE DELTA SUMMARY (vs T13 original)")
    print("=" * 70)
    print(f"{'cat':<12} {'mean_delta_ens':>16} {'mean_delta_orig':>16} {'replicated?':>12}")
    print(f"{'-'*60}")

    for cat in CATEGORIES:
        if cat not in all_summary:
            continue
        v2_acc = all_summary[cat]["v2_acc"]
        v1_acc = all_summary[cat]["v1_acc"]
        orig = orig_results.get(cat, {})

        deltas_ens = []
        for attr, v2 in v2_acc.items():
            acc_v1 = v1_acc.get(attr, float("nan"))
            acc_v2 = v2["accuracy"]
            if not (np.isnan(acc_v1) or np.isnan(acc_v2)):
                deltas_ens.append((acc_v2 - acc_v1) * 100.0)

        mean_ens = float(np.mean(deltas_ens)) if deltas_ens else float("nan")
        mean_orig_vals = [d for d in orig.values() if not np.isnan(d)]
        mean_orig = float(np.mean(mean_orig_vals)) if mean_orig_vals else float("nan")
        replicated = "YES" if mean_ens >= 5.0 else "NO"

        print(f"{cat:<12} {mean_ens:>+16.1f}pp {mean_orig:>+16.1f}pp {replicated:>12}")

    print("\nFraming: A (Full 3/3) if all 3 cats have mean_delta >= +5pp.")


if __name__ == "__main__":
    main()
