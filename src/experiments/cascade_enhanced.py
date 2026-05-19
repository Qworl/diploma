"""P2-EXP9: 3 architectural improvements to the 2-layer ensemble cascade (P_2 = 86.87%).

Variants tested (all honest 80/20 split, seed=42):
  Q_ml      - LightGBM with brand prior features  + XGB embeddings soft-vote 50/50
  R_ml      - LightGBM with numeric features      + XGB embeddings soft-vote 50/50
  QR_ml     - LightGBM with brand+numeric features + XGB embeddings soft-vote 50/50
  S         - Per-attr routing: tag→XGB, text→LGBM, nutri→XGB-on-numerics
  All       - (Q+R LightGBM + XGB + S) / 3 ensemble (3-way soft-vote)

Reference baseline P_2 = P_ensemble from EXP7 = 86.87%.

Output: datasets/processed/enhanced_ensemble.parquet
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
import scipy.sparse as sp
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.common import PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CATEGORIES = ["pasta", "chocolate", "cheeses"]
SEED = 42
TEST_SIZE = 0.2
MIN_GOLD = 20
GOLD_WEIGHT = 5.0

GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet"
TAXONOMY_PATH = Path(PROCESSED_DIR) / "attribute_signal_taxonomy.parquet"
OFF_CACHE_DIR = Path("datasets/manual_label/off_cache")
OUT_PATH = Path(PROCESSED_DIR) / "enhanced_ensemble.parquet"

NUMERIC_COLS = ["fat_100g", "proteins_100g", "sugars_100g", "salt_100g", "fiber_100g", "energy_100g"]

# ---------------------------------------------------------------------------
# Nutriments loader
# ---------------------------------------------------------------------------

def _load_nutriments_cache(codes: list[str]) -> dict[str, dict]:
    """Load nutriments from off_cache JSON files for given codes. Returns code→nutriments dict."""
    result: dict[str, dict] = {}
    for code in codes:
        # Try exact match or with leading zeros
        candidates = [
            OFF_CACHE_DIR / f"{code}.json",
            OFF_CACHE_DIR / f"{code.lstrip('0')}.json",
        ]
        for fpath in candidates:
            if fpath.exists():
                try:
                    with open(fpath) as fp:
                        d = json.load(fp)
                    nut = d.get("nutriments", {})
                    if isinstance(nut, dict):
                        result[code] = nut
                except Exception:
                    pass
                break
    return result


def _extract_numerics(codes: list[str], nutriments_cache: dict[str, dict], medians: dict[str, float]) -> np.ndarray:
    """Extract numeric nutriment features for a list of codes.
    Missing values filled with median computed on training data.
    Returns float32 array shape (n, len(NUMERIC_COLS)).
    """
    rows = []
    for code in codes:
        nut = nutriments_cache.get(code, {})
        row = []
        for col in NUMERIC_COLS:
            val = nut.get(col, None)
            if val is None or not isinstance(val, (int, float)):
                val = medians.get(col, 0.0)
            row.append(float(val))
        rows.append(row)
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Brand prior builder
# ---------------------------------------------------------------------------

def _compute_brand_norm(brands_series: pd.Series) -> pd.Series:
    """Extract first brand, lowercase, strip."""
    def _norm(v):
        if pd.isna(v) or str(v).strip() == "":
            return "__unknown__"
        return str(v).split(",")[0].strip().lower()
    return brands_series.map(_norm)


def _build_brand_prior(
    train_codes: list[str],
    y_train: np.ndarray,
    brand_norm_map: dict[str, str],
    all_classes: list[str],
    alpha: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute Laplace-smoothed P(class | brand) from training set.
    Returns dict: brand_norm → probability_vector (len = n_classes).
    """
    n_classes = len(all_classes)
    class_idx = {c: i for i, c in enumerate(all_classes)}

    # Accumulate counts per brand
    brand_counts: dict[str, np.ndarray] = {}
    for code, label in zip(train_codes, y_train):
        brand = brand_norm_map.get(code, "__unknown__")
        if brand not in brand_counts:
            brand_counts[brand] = np.zeros(n_classes, dtype=np.float32)
        idx = class_idx.get(str(label), -1)
        if idx >= 0:
            brand_counts[brand][idx] += 1.0

    # Laplace smoothed probabilities
    brand_prior: dict[str, np.ndarray] = {}
    for brand, counts in brand_counts.items():
        total = counts.sum() + alpha * n_classes
        brand_prior[brand] = (counts + alpha) / total

    # Uniform prior for unknown brands
    uniform = np.full(n_classes, 1.0 / n_classes, dtype=np.float32)
    brand_prior["__unknown__"] = uniform

    return brand_prior


def _get_brand_prior_features(
    codes: list[str],
    brand_norm_map: dict[str, str],
    brand_prior: dict[str, np.ndarray],
    n_classes: int,
) -> np.ndarray:
    """Get brand prior feature vectors for codes."""
    uniform = np.full(n_classes, 1.0 / n_classes, dtype=np.float32)
    rows = []
    for code in codes:
        brand = brand_norm_map.get(code, "__unknown__")
        vec = brand_prior.get(brand, brand_prior.get("__unknown__", uniform))
        rows.append(vec)
    return np.array(rows, dtype=np.float32)


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
# XGB trainer (embeddings-based, same as EXP7)
# ---------------------------------------------------------------------------

def _train_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    all_classes: list[str],
) -> tuple[Optional[xgb.XGBClassifier], Optional[LabelEncoder]]:
    """Train XGBoost on embedding features."""
    if len(all_classes) < 2:
        return None, None
    le = LabelEncoder()
    le.fit(all_classes)
    y_enc = le.transform(y_tr)
    n_classes = len(all_classes)
    common = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        tree_method="hist", verbosity=0,
    )
    if n_classes == 2:
        pos = int((y_enc == 1).sum())
        neg = int((y_enc == 0).sum())
        spw = max(neg / max(pos, 1), 0.5)
        clf = xgb.XGBClassifier(scale_pos_weight=spw, **common)
    else:
        clf = xgb.XGBClassifier(objective="multi:softmax", num_class=n_classes, **common)
    clf.fit(X_tr, y_enc)
    return clf, le


def _xgb_probas(clf: xgb.XGBClassifier, le: LabelEncoder, X: np.ndarray, all_classes: list[str]) -> np.ndarray:
    """Get class probability matrix aligned to all_classes order."""
    p = clf.predict_proba(X)
    # map columns: clf.classes_ order may differ from all_classes
    n_classes = len(all_classes)
    class_to_col = {c: i for i, c in enumerate(all_classes)}
    le_classes = le.classes_.tolist()
    aligned = np.zeros((len(X), n_classes), dtype=np.float32)
    for i, cls in enumerate(le_classes):
        j = class_to_col.get(str(cls), -1)
        if j >= 0:
            aligned[:, j] = p[:, i]
    return aligned


# ---------------------------------------------------------------------------
# LightGBM trainer (TF-IDF + optional extra features)
# ---------------------------------------------------------------------------

def _train_lgbm(
    train_texts: list[str],
    y_train: list[str],
    extra_train: Optional[np.ndarray] = None,
) -> tuple[Optional[lgb.LGBMClassifier], Optional[TfidfVectorizer], Optional[LabelEncoder]]:
    """Train LightGBM on TF-IDF + optional extra dense features."""
    all_classes = sorted(set(y_train))
    if len(all_classes) < 2:
        return None, None, None

    le = LabelEncoder()
    le.fit(all_classes)
    y_enc = le.transform(y_train)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10000)
    X_tfidf = vec.fit_transform(train_texts)

    if extra_train is not None and extra_train.shape[0] == X_tfidf.shape[0]:
        X_extra_sp = sp.csr_matrix(extra_train)
        X_train_feat = sp.hstack([X_tfidf, X_extra_sp], format="csr")
    else:
        X_train_feat = X_tfidf

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
    clf.fit(X_train_feat, y_enc)
    return clf, vec, le


def _lgbm_probas(
    clf: lgb.LGBMClassifier,
    vec: TfidfVectorizer,
    le: LabelEncoder,
    texts: list[str],
    all_classes: list[str],
    extra_test: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Get class probability matrix aligned to all_classes order."""
    X_tfidf = vec.transform(texts)
    if extra_test is not None and extra_test.shape[0] == X_tfidf.shape[0]:
        X_extra_sp = sp.csr_matrix(extra_test)
        X_feat = sp.hstack([X_tfidf, X_extra_sp], format="csr")
    else:
        X_feat = X_tfidf

    p = clf.predict_proba(X_feat)
    n_classes = len(all_classes)
    class_to_col = {c: i for i, c in enumerate(all_classes)}
    le_classes = le.classes_.tolist()
    aligned = np.zeros((X_tfidf.shape[0], n_classes), dtype=np.float32)
    for i, cls in enumerate(le_classes):
        j = class_to_col.get(str(cls), -1)
        if j >= 0:
            aligned[:, j] = p[:, i]
    return aligned


# ---------------------------------------------------------------------------
# XGB on numerics only (for nutri_derived attrs)
# ---------------------------------------------------------------------------

def _train_xgb_numeric(
    X_num_tr: np.ndarray,
    y_tr: np.ndarray,
    all_classes: list[str],
) -> tuple[Optional[xgb.XGBClassifier], Optional[LabelEncoder]]:
    """Train XGBoost on numeric nutriment features only."""
    if len(all_classes) < 2 or X_num_tr.shape[0] < 5:
        return None, None
    # Check that we actually have non-trivial numeric data
    if np.all(X_num_tr == 0):
        return None, None
    return _train_xgb(X_num_tr, y_tr, all_classes)


# ---------------------------------------------------------------------------
# Main per-(cat, attr) function
# ---------------------------------------------------------------------------

def run_one_attr(
    cat: str,
    attr: str,
    gold: pd.DataFrame,
    silver: pd.DataFrame,
    emb_all: np.ndarray,
    code_to_idx: dict[str, int],
    train_codes_set: set[str],
    test_codes_set: set[str],
    test_products: pd.DataFrame,
    brand_norm_map: dict[str, str],
    nutriments_cache: dict[str, dict],
    signal_type: str,
) -> list[dict]:
    """Run all variants for one (cat, attr)."""

    # Gold for this attr
    cat_gold = gold[
        (gold["category"] == cat) & (gold["attr"] == attr) & ~gold["gold_is_null"]
    ].copy()
    cat_gold["code"] = cat_gold["code"].astype(str)
    cat_gold = cat_gold[cat_gold["code"].isin(code_to_idx)]

    if len(cat_gold) < MIN_GOLD:
        return []

    train_gold = cat_gold[cat_gold["code"].isin(train_codes_set)]
    test_gold = cat_gold[cat_gold["code"].isin(test_codes_set)]

    if len(train_gold) < 10 or len(test_gold) < 5:
        return []

    # Embedding arrays
    train_idx = np.array([code_to_idx[c] for c in train_gold["code"]])
    test_idx = np.array([code_to_idx[c] for c in test_gold["code"]])

    X_gold_train_emb = emb_all[train_idx]
    y_gold_train = train_gold["gold_value"].astype(str).values
    X_test_emb = emb_all[test_idx]
    y_test = test_gold["gold_value"].astype(str).values
    test_codes_list = test_gold["code"].tolist()
    train_codes_list = train_gold["code"].tolist()

    # Silver data (exclude test codes)
    X_silver_emb = np.empty((0, emb_all.shape[1]))
    y_silver = np.array([], dtype=str)
    silver_codes: list[str] = []

    if attr in silver.columns:
        silver_attr = silver[silver[attr].notna()].copy()
        silver_attr["code"] = silver_attr["code"].astype(str)
        silver_attr = silver_attr[~silver_attr["code"].isin(test_codes_set)]
        silver_attr = silver_attr[~silver_attr["code"].isin(train_codes_set)]
        silver_attr = silver_attr[silver_attr["code"].isin(code_to_idx)]
        if len(silver_attr) > 0:
            s_idx = np.array([code_to_idx[c] for c in silver_attr["code"]])
            X_silver_emb = emb_all[s_idx]
            y_silver = silver_attr[attr].astype(str).values
            silver_codes = silver_attr["code"].tolist()

    # Combined train for XGB hybrid (silver+gold weighted)
    if len(X_silver_emb) > 0:
        X_hybrid_emb = np.vstack([X_silver_emb, X_gold_train_emb])
        y_hybrid = np.concatenate([y_silver, y_gold_train])
        w_hybrid = np.concatenate([
            np.ones(len(y_silver)),
            GOLD_WEIGHT * np.ones(len(y_gold_train)),
        ])
    else:
        X_hybrid_emb = X_gold_train_emb
        y_hybrid = y_gold_train
        w_hybrid = GOLD_WEIGHT * np.ones(len(y_gold_train))

    all_classes = sorted(set(y_hybrid.tolist()))
    if len(all_classes) < 2:
        return []

    # ---- Train base XGB (embeddings, same as P_2) ----
    le_xgb_full = LabelEncoder()
    le_xgb_full.fit(all_classes)
    y_hybrid_enc = le_xgb_full.transform(y_hybrid)
    n_classes = len(all_classes)

    common_xgb = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        tree_method="hist", verbosity=0,
    )
    if n_classes == 2:
        pos = int((y_hybrid_enc == 1).sum())
        neg = int((y_hybrid_enc == 0).sum())
        spw = max(neg / max(pos, 1), 0.5)
        clf_xgb = xgb.XGBClassifier(scale_pos_weight=spw, **common_xgb)
    else:
        clf_xgb = xgb.XGBClassifier(
            objective="multi:softmax", num_class=n_classes, **common_xgb
        )
    clf_xgb.fit(X_hybrid_emb, y_hybrid_enc, sample_weight=w_hybrid)

    p_xgb_test = _xgb_probas(clf_xgb, le_xgb_full, X_test_emb, all_classes)

    # ---- Build text features ----
    # For LGBM: use gold train texts + silver texts from silver df
    # Combine train codes (gold) with silver codes for LGBM training
    lgbm_train_codes = train_codes_list + silver_codes
    lgbm_train_labels = list(y_gold_train) + list(y_silver)
    lgbm_weights = (
        [GOLD_WEIGHT] * len(train_codes_list) + [1.0] * len(silver_codes)
    )

    # Build text for LGBM train
    silver_indexed = silver.set_index("code")
    def _text_for_code(code: str) -> str:
        if code in silver_indexed.index:
            return _build_text(silver_indexed.loc[code])
        return ""

    lgbm_train_texts = [_text_for_code(c) for c in lgbm_train_codes]

    # Test texts
    test_products_idx = test_products.set_index("code")
    test_texts = []
    for c in test_codes_list:
        if c in test_products_idx.index:
            test_texts.append(_build_text(test_products_idx.loc[c]))
        else:
            test_texts.append("")

    # ---- Numeric features ----
    # Compute medians from training codes (gold + silver)
    all_train_nut_codes = lgbm_train_codes  # use same pool
    train_nut_cache = {c: nutriments_cache.get(c, {}) for c in all_train_nut_codes}

    medians: dict[str, float] = {}
    for col in NUMERIC_COLS:
        vals = []
        for nut in train_nut_cache.values():
            v = nut.get(col, None)
            if v is not None and isinstance(v, (int, float)):
                vals.append(float(v))
        medians[col] = float(np.median(vals)) if vals else 0.0

    X_num_train = _extract_numerics(lgbm_train_codes, nutriments_cache, medians)
    X_num_test = _extract_numerics(test_codes_list, nutriments_cache, medians)

    # ---- Brand prior features ----
    brand_prior_map = _build_brand_prior(
        lgbm_train_codes, np.array(lgbm_train_labels), brand_norm_map, all_classes
    )
    X_brand_train = _get_brand_prior_features(lgbm_train_codes, brand_norm_map, brand_prior_map, n_classes)
    X_brand_test = _get_brand_prior_features(test_codes_list, brand_norm_map, brand_prior_map, n_classes)

    # ---- Train LightGBM variants ----

    # Q: LGBM with brand prior only
    lgbm_q, vec_q, le_q = _train_lgbm(lgbm_train_texts, lgbm_train_labels, extra_train=X_brand_train)
    p_q_test = None
    if lgbm_q is not None:
        p_q_test = _lgbm_probas(lgbm_q, vec_q, le_q, test_texts, all_classes, extra_test=X_brand_test)

    # R: LGBM with numeric only
    lgbm_r, vec_r, le_r = _train_lgbm(lgbm_train_texts, lgbm_train_labels, extra_train=X_num_train)
    p_r_test = None
    if lgbm_r is not None:
        p_r_test = _lgbm_probas(lgbm_r, vec_r, le_r, test_texts, all_classes, extra_test=X_num_test)

    # QR: LGBM with brand + numeric
    X_qr_train = np.hstack([X_brand_train, X_num_train])
    X_qr_test = np.hstack([X_brand_test, X_num_test])
    lgbm_qr, vec_qr, le_qr = _train_lgbm(lgbm_train_texts, lgbm_train_labels, extra_train=X_qr_train)
    p_qr_test = None
    if lgbm_qr is not None:
        p_qr_test = _lgbm_probas(lgbm_qr, vec_qr, le_qr, test_texts, all_classes, extra_test=X_qr_test)

    # Baseline LGBM (TF-IDF only, same as EXP7)
    lgbm_base, vec_base, le_base = _train_lgbm(lgbm_train_texts, lgbm_train_labels, extra_train=None)
    p_lgbm_base_test = None
    if lgbm_base is not None:
        p_lgbm_base_test = _lgbm_probas(lgbm_base, vec_base, le_base, test_texts, all_classes, extra_test=None)

    # ---- S: per-attr routing ----
    # tag_derived → XGB embeddings
    # text_derived → LGBM TF-IDF (baseline, no extras)
    # nutri_derived → XGB on numerics
    # multi_source / missing → use P_2 ensemble (lgbm_base + xgb) / 2

    # Train XGB on numerics only (for nutri_derived)
    clf_xgb_num, le_xgb_num = _train_xgb_numeric(X_num_train, np.array(lgbm_train_labels), all_classes)
    p_xgb_num_test = None
    if clf_xgb_num is not None:
        p_xgb_num_test = _xgb_probas(clf_xgb_num, le_xgb_num, X_num_test, all_classes)

    def _predict_labels(prob_matrix: np.ndarray) -> list[str]:
        idxs = prob_matrix.argmax(axis=1)
        return [all_classes[i] for i in idxs]

    def _acc(preds: list[str]) -> float:
        n = len(y_test)
        if n == 0:
            return float("nan")
        return sum(p == g for p, g in zip(preds, y_test)) / n

    # ---- Compute P_2 baseline (LGBM base + XGB) / 2 ----
    if p_lgbm_base_test is not None:
        p_p2 = (p_lgbm_base_test + p_xgb_test) / 2.0
    else:
        p_p2 = p_xgb_test
    acc_p2 = _acc(_predict_labels(p_p2))

    # ---- Q_ml: (Q LGBM + XGB) / 2 ----
    if p_q_test is not None:
        p_q_ml = (p_q_test + p_xgb_test) / 2.0
    else:
        p_q_ml = p_xgb_test
    acc_q_ml = _acc(_predict_labels(p_q_ml))

    # ---- R_ml: (R LGBM + XGB) / 2 ----
    if p_r_test is not None:
        p_r_ml = (p_r_test + p_xgb_test) / 2.0
    else:
        p_r_ml = p_xgb_test
    acc_r_ml = _acc(_predict_labels(p_r_ml))

    # ---- QR_ml: (QR LGBM + XGB) / 2 ----
    if p_qr_test is not None:
        p_qr_ml = (p_qr_test + p_xgb_test) / 2.0
    else:
        p_qr_ml = p_xgb_test
    acc_qr_ml = _acc(_predict_labels(p_qr_ml))

    # ---- S: per-attr routing ----
    if signal_type == "tag_derived":
        p_s = p_xgb_test
    elif signal_type == "text_derived":
        p_s = p_lgbm_base_test if p_lgbm_base_test is not None else p_xgb_test
    elif signal_type == "nutri_derived":
        p_s = p_xgb_num_test if p_xgb_num_test is not None else p_xgb_test
    else:
        # multi_source or missing → P_2 ensemble
        p_s = p_p2
    acc_s = _acc(_predict_labels(p_s))

    # ---- All: (QR LGBM + XGB + S) / 3 ----
    # Use S probabilities (which already encodes routing)
    if p_qr_test is not None:
        p_all = (p_qr_test + p_xgb_test + p_s) / 3.0
    else:
        p_all = (p_xgb_test + p_s) / 2.0
    acc_all = _acc(_predict_labels(p_all))

    n = len(y_test)
    logger.info(
        "[%s/%s/%s] n=%d | P_2=%.3f | Q=%.3f | R=%.3f | QR=%.3f | S=%.3f | All=%.3f",
        cat, attr, signal_type, n, acc_p2, acc_q_ml, acc_r_ml, acc_qr_ml, acc_s, acc_all,
    )

    rows = []
    for variant, acc in [
        ("P_2_baseline", acc_p2),
        ("Q_ml", acc_q_ml),
        ("R_ml", acc_r_ml),
        ("QR_ml", acc_qr_ml),
        ("S_routing", acc_s),
        ("All_ensemble", acc_all),
    ]:
        rows.append({
            "category": cat,
            "attr": attr,
            "signal_type": signal_type,
            "variant": variant,
            "accuracy": acc,
            "n_test": n,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)
    logger.info("Loaded gold: %d rows, %d codes", len(gold), gold["code"].nunique())

    taxonomy = pd.read_parquet(TAXONOMY_PATH)
    # Build (cat, attr) → signal_type lookup
    tax_map: dict[tuple[str, str], str] = {
        (row["category"], row["attr"]): row["signal_type"]
        for _, row in taxonomy.iterrows()
    }

    all_rows: list[dict] = []

    for cat in CATEGORIES:
        logger.info("=== Category: %s ===", cat)

        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        )
        silver["code"] = silver["code"].astype(str)

        emb_all = np.load(Path(PROCESSED_DIR) / f"{cat}_stratified_embeddings.npy")
        code_to_idx: dict[str, int] = {c: i for i, c in enumerate(silver["code"].tolist())}

        # Build brand_norm_map: code → brand_norm string
        brand_norm_col = _compute_brand_norm(silver["brands"])
        brand_norm_map: dict[str, str] = dict(zip(silver["code"].tolist(), brand_norm_col.tolist()))

        # Load nutriments from off_cache for all codes in this category
        all_cat_codes = list(code_to_idx.keys())
        logger.info("  Loading nutriments for %d codes...", len(all_cat_codes))
        nutriments_cache = _load_nutriments_cache(all_cat_codes)
        logger.info("  Nutriments found for %d / %d codes", len(nutriments_cache), len(all_cat_codes))

        # 80/20 split on gold codes
        cat_gold = gold[gold["category"] == cat].copy()
        unique_codes = sorted(cat_gold["code"].unique().tolist())
        train_codes, test_codes = train_test_split(
            unique_codes, test_size=TEST_SIZE, random_state=SEED
        )
        train_codes_set = set(train_codes)
        test_codes_set = set(test_codes)
        logger.info("  Split: %d train / %d test codes", len(train_codes), len(test_codes))

        test_products = silver[silver["code"].isin(test_codes_set)].copy()

        attrs = sorted(cat_gold["attr"].unique().tolist())
        for attr in attrs:
            signal_type = tax_map.get((cat, attr), "text_derived")  # default text_derived
            rows = run_one_attr(
                cat=cat,
                attr=attr,
                gold=cat_gold,
                silver=silver,
                emb_all=emb_all,
                code_to_idx=code_to_idx,
                train_codes_set=train_codes_set,
                test_codes_set=test_codes_set,
                test_products=test_products,
                brand_norm_map=brand_norm_map,
                nutriments_cache=nutriments_cache,
                signal_type=signal_type,
            )
            all_rows.extend(rows)

    result = pd.DataFrame(all_rows)
    result.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(result), OUT_PATH)

    _print_summary(result)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

VARIANTS_ORDER = ["P_2_baseline", "Q_ml", "R_ml", "QR_ml", "S_routing", "All_ensemble"]


def _print_summary(result: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("P2-EXP9: Enhanced Ensemble — Grand Mean Accuracy")
    print("=" * 72)
    grand = result.groupby("variant")["accuracy"].mean()
    p2 = grand.get("P_2_baseline", float("nan"))
    for v in VARIANTS_ORDER:
        acc = grand.get(v, float("nan"))
        delta = (acc - p2) * 100
        marker = "  <-- BEST" if acc == grand.max() else ""
        if v == "P_2_baseline":
            print(f"  {v:20s}: {acc*100:.2f}%  (reference){marker}")
        else:
            print(f"  {v:20s}: {acc*100:.2f}%  ({delta:+.2f} pp){marker}")

    print("\n" + "=" * 72)
    print("Per-variant accuracy by signal_type")
    print("=" * 72)
    pt = result.groupby(["signal_type", "variant"])["accuracy"].mean().unstack("variant")
    for col in VARIANTS_ORDER:
        if col not in pt.columns:
            pt[col] = float("nan")
    print(pt[VARIANTS_ORDER].round(4).to_string())

    print("\n" + "=" * 72)
    print("Per-(category, attr) accuracy — all variants")
    print("=" * 72)
    pt2 = result.pivot_table(
        index=["category", "attr", "signal_type"],
        columns="variant",
        values="accuracy",
        aggfunc="mean",
    )
    for col in VARIANTS_ORDER:
        if col not in pt2.columns:
            pt2[col] = float("nan")
    print(pt2[VARIANTS_ORDER].round(4).to_string())

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    best_var = grand.idxmax()
    best_acc = grand.max()
    delta_best = (best_acc - p2) * 100
    print(f"  Best variant: {best_var} at {best_acc*100:.2f}% ({delta_best:+.2f} pp vs P_2)")
    print(f"  P_2 baseline: {p2*100:.2f}%")


if __name__ == "__main__":
    main()
