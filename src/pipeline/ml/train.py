"""
ML classifiers (Layer 2 of hybrid system).

Multilingual embeddings (sentence-transformers) + XGBoost.
Labels from silver standard (LLM-generated ground truth).

Usage:
    python -m src.pipeline.ml.train --category pasta
"""

import argparse
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.common import (
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    build_text,
    get_embeddings,
    setup_logging,
)

logger = logging.getLogger(__name__)

CATEGORY_CONFIG = {
    "pasta": {
        "silver_standard": "pasta_silver_standard.parquet",
        "embeddings_cache": "pasta_embeddings.npy",
        "classifiers": {
            "grain_type": {"type": "multiclass", "min_samples": 5},
            "pasta_shape": {"type": "multiclass", "min_samples": 10},
            "is_whole_grain": {"type": "binary", "min_positive": 10},
            "is_organic": {"type": "binary", "min_positive": 10},
            "is_gluten_free": {"type": "binary", "min_positive": 10},
            "is_vegan": {"type": "binary", "min_positive": 10},
            "nutri_score_grade": {"type": "multiclass", "min_samples": 5},
            "protein_class": {"type": "multiclass", "min_samples": 5},
        },
    },
    "pasta_stratified": {
        "silver_standard": "pasta_stratified_silver_standard.parquet",
        "embeddings_cache": "pasta_stratified_embeddings.npy",
        "classifiers": {
            "grain_type": {"type": "multiclass", "min_samples": 5},
            "pasta_shape": {"type": "multiclass", "min_samples": 10},
            # is_whole_grain заменён на is_filled (degenerate было 99.5% False)
            "is_filled": {"type": "binary", "min_positive": 5},
            "is_organic": {"type": "binary", "min_positive": 10},
            "is_gluten_free": {"type": "binary", "min_positive": 10},
            "is_vegan": {"type": "binary", "min_positive": 10},
            "nutri_score_grade": {"type": "multiclass", "min_samples": 5},
            "protein_class": {"type": "multiclass", "min_samples": 5},
        },
    },
    "chocolate": {
        "silver_standard": "chocolate_silver_standard.parquet",
        "embeddings_cache": "chocolate_embeddings.npy",
        "classifiers": {
            "chocolate_type": {"type": "multiclass", "min_samples": 5},
            "cocoa_percentage": {"type": "multiclass", "min_samples": 5},
            "contains_nuts": {"type": "binary", "min_positive": 10},
            "palm_oil_status": {"type": "multiclass", "min_samples": 5},
            "is_organic": {"type": "binary", "min_positive": 10},
            "nutri_score_grade": {"type": "multiclass", "min_samples": 5},
            "protein_class": {"type": "multiclass", "min_samples": 5},
        },
    },
    "chocolate_stratified": {
        "silver_standard": "chocolate_stratified_silver_standard.parquet",
        "embeddings_cache": "chocolate_stratified_embeddings.npy",
        "classifiers": {
            "chocolate_type": {"type": "multiclass", "min_samples": 5},
            "cocoa_percentage": {"type": "multiclass", "min_samples": 5},
            "contains_nuts": {"type": "binary", "min_positive": 10},
            # palm_oil_status удалён (95% palm-oil-free — degenerate),
            # заменён на chocolate_extra (plain/with_nuts/with_fruit/...)
            "chocolate_extra": {"type": "multiclass", "min_samples": 5},
            "is_organic": {"type": "binary", "min_positive": 10},
            "nutri_score_grade": {"type": "multiclass", "min_samples": 5},
            "protein_class": {"type": "multiclass", "min_samples": 5},
        },
    },
    "beverages": {
        "silver_standard": "beverages_silver_standard.parquet",
        "embeddings_cache": "beverages_embeddings.npy",
        "classifiers": {
            "beverage_type": {"type": "multiclass", "min_samples": 5},
            "sugar_class": {"type": "multiclass", "min_samples": 5},
            "is_organic": {"type": "binary", "min_positive": 10},
            "is_no_added_sugar": {"type": "binary", "min_positive": 10},
            "nutri_score_grade": {"type": "multiclass", "min_samples": 5},
            "nova_group": {"type": "multiclass", "min_samples": 5},
            "protein_class": {"type": "multiclass", "min_samples": 5},
        },
    },
    "beverages_stratified": {
        "silver_standard": "beverages_stratified_silver_standard.parquet",
        "embeddings_cache": "beverages_stratified_embeddings.npy",
        "classifiers": {
            "beverage_type": {"type": "multiclass", "min_samples": 5},
            "sugar_class": {"type": "multiclass", "min_samples": 5},
            "is_organic": {"type": "binary", "min_positive": 10},
            # is_no_added_sugar заменён на is_carbonated (был 94% False — degenerate)
            "is_carbonated": {"type": "binary", "min_positive": 10},
            "nutri_score_grade": {"type": "multiclass", "min_samples": 5},
            "nova_group": {"type": "multiclass", "min_samples": 5},
            "protein_class": {"type": "multiclass", "min_samples": 5},
            "is_vegan": {"type": "binary", "min_positive": 10},
        },
    },
    "electronics": {
        "silver_standard": "electronics_silver_standard.parquet",
        "embeddings_cache": "electronics_embeddings.npy",
        "classifiers": {
            # Per audit: 5 ML_READY, 2 BAYESIAN_ONLY (skip ML), 1 DROP (price_tier).
            "brand": {"type": "multiclass", "min_samples": 5},
            "screen_size_class": {"type": "multiclass", "min_samples": 5},
            "ram_class": {"type": "multiclass", "min_samples": 5},
            "storage_class": {"type": "multiclass", "min_samples": 5},
            "release_year_class": {"type": "multiclass", "min_samples": 5},
        },
    },
    # baby_stratified removed 2026-05-11: domain слишком узкий после filter to baby_milks
    # (96% milk_type=cow, 95% format=powder, 94% feeding_purpose=regular). ML побеждает
    # majority class, Bayes Δ ≈ 0. Перенесён в Приложение как «narrow-domain контр-пример».
    "cheeses_stratified": {
        "silver_standard": "cheeses_stratified_silver_standard.parquet",
        "embeddings_cache": "cheeses_stratified_embeddings.npy",
        "classifiers": {
            "milk_source": {"type": "multiclass", "min_samples": 5},
            "texture": {"type": "multiclass", "min_samples": 5},
            "country_of_origin": {"type": "multiclass", "min_samples": 5},
            "fat_class": {"type": "multiclass", "min_samples": 5},
            "is_pdo": {"type": "binary", "min_positive": 10},
            "is_organic": {"type": "binary", "min_positive": 10},
            "is_ultra_processed": {"type": "binary", "min_positive": 10},
        },
    },
    "cereals_stratified": {
        "silver_standard": "cereals_stratified_silver_standard.parquet",
        "embeddings_cache": "cereals_stratified_embeddings.npy",
        "classifiers": {
            "cereal_type": {"type": "multiclass", "min_samples": 5},
            "grain_type": {"type": "multiclass", "min_samples": 5},
            "is_low_sugar": {"type": "binary", "min_positive": 10},
            "is_high_fibre": {"type": "binary", "min_positive": 10},
            "nova_class": {"type": "multiclass", "min_samples": 5},
            "is_vegan": {"type": "binary", "min_positive": 10},
            "is_whole_grain": {"type": "binary", "min_positive": 10},
            "is_organic": {"type": "binary", "min_positive": 10},
        },
    },
    "cosmetics_stratified": {
        "silver_standard": "cosmetics_stratified_silver_standard.parquet",
        "embeddings_cache": "cosmetics_stratified_embeddings.npy",
        "classifiers": {
            "product_type": {"type": "multiclass", "min_samples": 5},
            "form_factor": {"type": "multiclass", "min_samples": 5},
            "body_area": {"type": "multiclass", "min_samples": 5},
            "has_sulfates": {"type": "binary", "min_positive": 10},
            "has_silicones": {"type": "binary", "min_positive": 10},
            "is_organic": {"type": "binary", "min_positive": 10},
        },
    },
    # gold v4: DeepSeek-V4-flash relabel (v5_relabel) + rules-derived TYPE_C.
    # Schema cleanup 2026-05-24: cuisine_origin/flavor_profile/aging added;
    # TYPE_C (fat_class, nutri_score_grade, protein_class, cocoa_percentage)
    # populated deterministically from src/pipeline/off_labels/rules.py.
    # TYPE_C attrs (protein_class, cocoa_percentage, nutri_score_grade, fat_class)
    # исключены из ML — они deterministic из nutriments через TYPE_C_RULES (rules.py).
    # ML на rules-derived labels хуже самих rules; для 3% продуктов без nutriments
    # есть LLM fallback (Layer 4).
    "pasta_v4": {
        "silver_standard": "pasta_gold_v4_wide.parquet",
        "embeddings_cache": "pasta_v4_embeddings.npy",
        "classifiers": {
            "grain_type": {"type": "multiclass", "min_samples": 5},
            "pasta_shape": {"type": "multiclass", "min_samples": 10},
            "is_filled": {"type": "binary", "min_positive": 5},
            "is_organic": {"type": "binary", "min_positive": 10},
            "is_gluten_free": {"type": "binary", "min_positive": 10},
            "is_vegan": {"type": "binary", "min_positive": 10},
            "cuisine_origin": {"type": "multiclass", "min_samples": 10},
        },
    },
    "chocolate_v4": {
        "silver_standard": "chocolate_gold_v4_wide.parquet",
        "embeddings_cache": "chocolate_v4_embeddings.npy",
        "classifiers": {
            # Schema refactor: chocolate_type (base type) is orthogonal to is_filled (structural form).
            # Old schema conflated them — a "milk + filled" product had to choose between milk and filled.
            "chocolate_type": {"type": "multiclass", "min_samples": 5,
                               "exclude_classes": ["filled", "other"]},
            "is_filled": {"type": "binary", "min_positive": 10},
            "contains_nuts": {"type": "binary", "min_positive": 10},
            # chocolate_extra now strictly about mix-ins (orthogonal to is_filled).
            "chocolate_extra": {"type": "multiclass", "min_samples": 5,
                                "exclude_classes": ["filled", "other", "with_alcohol", "with_coffee"]},
            "is_organic": {"type": "binary", "min_positive": 10},
            # flavor_profile.other: n=72/5000 (1.4%), recall=0.11 — catch-all class without coherent signal.
            "flavor_profile": {"type": "multiclass", "min_samples": 10,
                               "exclude_classes": ["other"]},
        },
    },
    "cheeses_v4": {
        "silver_standard": "cheeses_gold_v4_wide.parquet",
        "embeddings_cache": "cheeses_v4_embeddings.npy",
        "classifiers": {
            "milk_source": {"type": "multiclass", "min_samples": 5},
            # texture.other: n=14/2815 (0.5%), recall=0.14 — unlearnable catch-all class.
            "texture": {"type": "multiclass", "min_samples": 5,
                        "exclude_classes": ["other"]},
            "country_of_origin": {"type": "multiclass", "min_samples": 5},
            "is_pdo": {"type": "binary", "min_positive": 10},
            "is_organic": {"type": "binary", "min_positive": 10},
            "is_ultra_processed": {"type": "binary", "min_positive": 10},
            "aging": {"type": "multiclass", "min_samples": 10},
        },
    },
}

# XGBoost hyperparameters. n_jobs=-1 uses all available cores
# (на macOS должно работать через OMP_NUM_THREADS=1 чтобы избежать libomp segfault;
# на linux VM лимита нет, можно жить с дефолтом).
import os as _os_for_xgb
_XGB_N_JOBS = int(_os_for_xgb.environ.get("XGB_N_JOBS", "-1"))

MULTICLASS_PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=30,
    n_jobs=_XGB_N_JOBS,
)
BINARY_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=30,
    n_jobs=_XGB_N_JOBS,
)


def find_best_threshold(clf, X_val, y_val):
    """Sweep confidence thresholds on validation set, return optimal."""
    proba = clf.predict_proba(X_val)
    best_t, best_score = 0.5, 0.0
    for t in np.arange(0.5, 0.95, 0.05):
        mask = proba.max(axis=1) >= t
        if mask.sum() < 10:
            continue
        preds = proba[mask].argmax(axis=1)
        f1 = f1_score(y_val[mask], preds, average="macro", zero_division=0)
        score = f1 * (mask.mean() ** 0.3)
        if score > best_score:
            best_score = score
            best_t = t
    return round(best_t, 2)


def compute_ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Expected Calibration Error (max-prob style) + per-bin breakdown for plotting.

    For each prediction we take p = max-class probability and check whether the model
    was correct. Bin by p; ECE is the weighted mean of |bin_acc - bin_conf|.
    Returns (ece, bins) where bins is a list of {p_lo, p_hi, count, acc, mean_conf}.
    """
    confidences = proba.max(axis=1)
    preds = proba.argmax(axis=1)
    correct = (preds == y_true).astype(float)
    bins_info = []
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
            bins_info.append({"p_lo": float(lo), "p_hi": float(hi),
                              "count": 0, "acc": None, "mean_conf": None})
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidences[mask].mean())
        bins_info.append({"p_lo": float(lo), "p_hi": float(hi),
                          "count": cnt, "acc": bin_acc, "mean_conf": bin_conf})
        ece += (cnt / n) * abs(bin_acc - bin_conf)
    return float(ece), bins_info


def calibrate(clf, X_calib, y_calib, method: str = "sigmoid"):
    """Wrap a fitted classifier in CalibratedClassifierCV using held-out data.

    Uses FrozenEstimator (sklearn 1.6+) so the underlying XGB is not re-fit.
    Default is method="sigmoid" (Platt scaling, 2 parameters per class) — robust on
    small calibration sets (<200). Use method="isotonic" when calibration set ≥500.

    Falls back gracefully (returns the original clf) if the calibration set is too
    small or imbalanced.
    """
    if len(y_calib) < 30:
        logger.warning("Calibration set too small (%d) — skipping calibration", len(y_calib))
        return clf, False
    unique, counts = np.unique(y_calib, return_counts=True)
    cv = min(3, int(counts.min()))
    if cv < 2:
        logger.warning("Calibration set has class with <2 samples — skipping calibration")
        return clf, False
    try:
        cal = CalibratedClassifierCV(FrozenEstimator(clf), method=method, cv=cv)
        cal.fit(X_calib, y_calib)
    except ValueError as e:
        logger.warning("Calibration failed (%s) — returning uncalibrated clf", e)
        return clf, False
    return cal, True


def _stratified_split_safe(X, y, *extras, test_size, random_state, stratify):
    """train_test_split with graceful fallback if some classes have <2 samples."""
    try:
        return train_test_split(X, y, *extras, test_size=test_size,
                                random_state=random_state, stratify=stratify)
    except ValueError:
        return train_test_split(X, y, *extras, test_size=test_size,
                                random_state=random_state)


def _save_calibration_data(prefix: str, attr_name: str, ece_raw: float,
                            ece_cal: float | None, bins_raw: list, bins_cal: list | None):
    """Persist ECE/bins JSON for notebook reliability diagrams."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    out = {
        "attr": attr_name,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "bins_raw": bins_raw,
        "bins_calibrated": bins_cal,
    }
    path = os.path.join(MODELS_DIR, f"{prefix}_{attr_name}_calibration.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def train_multiclass(X_train, X_test, y_train, y_test, attr_name: str, prefix: str,
                     do_calibrate: bool = True, calibration_method: str = "sigmoid"):
    """Train multiclass classifier with regularization, early stopping, and calibration.

    Splits train into:
      - 90% fit_pool (inside which 15% goes to early-stop val)
      - 10% calib (held out, used only for CalibratedClassifierCV)
    """
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    n_classes = len(le.classes_)
    class_counts = np.bincount(y_train_enc, minlength=n_classes)
    weights = len(y_train_enc) / (n_classes * np.maximum(class_counts, 1))
    sample_weights = weights[y_train_enc]

    if do_calibrate:
        X_fit, X_calib, y_fit, y_calib, sw_fit, _ = _stratified_split_safe(
            X_train, y_train_enc, sample_weights,
            test_size=0.10, random_state=RANDOM_STATE, stratify=y_train_enc,
        )
    else:
        X_fit, y_fit, sw_fit = X_train, y_train_enc, sample_weights
        X_calib, y_calib = None, None

    X_tr, X_val, y_tr, y_val, sw_tr, _ = _stratified_split_safe(
        X_fit, y_fit, sw_fit,
        test_size=0.15, random_state=RANDOM_STATE, stratify=y_fit,
    )

    clf = XGBClassifier(
        **MULTICLASS_PARAMS,
        eval_metric="mlogloss",
    )
    # Двойной internal split (calibration 90/10 + val 85/15) на разреженных
    # классах может оставить val без некоторых классов из train. XGBoost
    # mlogloss требует одинаковый набор классов в train/val. При ошибке —
    # обучаем без eval_set (теряем early stopping, но не падаем).
    try:
        clf.fit(X_tr, y_tr, sample_weight=sw_tr,
                eval_set=[(X_val, y_val)], verbose=False)
    except Exception as e:
        logger.warning("%s: early-stop val incompatible (%s) — fitting without eval_set",
                       attr_name, str(e)[:100])
        clf_no_es = XGBClassifier(
            **{k: v for k, v in MULTICLASS_PARAMS.items() if k != "early_stopping_rounds"},
            eval_metric="mlogloss",
        )
        clf_no_es.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
        clf = clf_no_es

    # Raw uncalibrated proba — for ECE before calibration
    y_proba_raw = clf.predict_proba(X_test)
    ece_raw, bins_raw = compute_ece(y_test_enc, y_proba_raw)

    final_clf = clf
    ece_cal, bins_cal = None, None
    if do_calibrate and X_calib is not None:
        cal_clf, ok = calibrate(clf, X_calib, y_calib, method=calibration_method)
        if ok:
            final_clf = cal_clf
            y_proba_cal = cal_clf.predict_proba(X_test)
            ece_cal, bins_cal = compute_ece(y_test_enc, y_proba_cal)

    y_proba = final_clf.predict_proba(X_test)
    y_pred_enc = y_proba.argmax(axis=1)
    confidences = y_proba.max(axis=1)
    # Threshold tuned on X_val (internal training split), NOT X_test.
    # Reporting acc/f1 with X_test-tuned threshold on X_test is leakage.
    best_threshold = find_best_threshold(final_clf, X_val, y_val)

    cal_info = f", ECE {ece_raw:.3f}→{ece_cal:.3f}" if ece_cal is not None else f", ECE {ece_raw:.3f} (uncalibrated)"
    best_iter = getattr(clf, "best_iteration", None)
    iter_str = f"best_iter={best_iter}" if best_iter is not None else "no_eval_set"
    logger.info("=== %s (%s, threshold=%.2f%s) ===",
                attr_name, iter_str, best_threshold, cal_info)
    report = classification_report(
        y_test_enc, y_pred_enc,
        labels=np.arange(len(le.classes_)),
        target_names=[str(c) for c in le.classes_],
        zero_division=0,
    )
    for line in report.strip().split("\n"):
        logger.info("  %s", line)

    high_conf = confidences >= best_threshold
    if high_conf.sum() > 0:
        acc = (y_pred_enc[high_conf] == y_test_enc[high_conf]).mean()
        f1_macro = f1_score(y_test_enc[high_conf], y_pred_enc[high_conf],
                            average="macro", zero_division=0)
        f1_weighted = f1_score(y_test_enc[high_conf], y_pred_enc[high_conf],
                               average="weighted", zero_division=0)
        logger.info("  conf >= %.2f: %d/%d products, acc=%.3f, "
                    "macro_f1=%.3f, weighted_f1=%.3f",
                     best_threshold, high_conf.sum(), len(y_test_enc), acc,
                     f1_macro, f1_weighted)
        # Confidence calibration check: how often is ML high-confidence on each class?
        # Reveals "router silently keeps wrong minority predictions" issue.
        per_class_rows = []
        for cls_id in range(len(le.classes_)):
            cls_name = str(le.classes_[cls_id])
            cls_mask_true = y_test_enc == cls_id
            cls_mask_pred_conf = (y_pred_enc == cls_id) & high_conf
            n_true = cls_mask_true.sum()
            n_pred_conf = cls_mask_pred_conf.sum()
            if n_pred_conf > 0:
                pred_acc = (y_test_enc[cls_mask_pred_conf] == cls_id).mean()
            else:
                pred_acc = float('nan')
            recall_conf = ((y_pred_enc == cls_id) & high_conf & cls_mask_true).sum() / max(n_true, 1)
            per_class_rows.append((cls_name, n_true, n_pred_conf, pred_acc, recall_conf))
        # Compact one-line per class report
        logger.info("  per-class @threshold: "
                    + " | ".join(f"{n}: n_true={t}/n_pred_conf={p}/prec={a:.2f}/recall={r:.2f}"
                                 for n, t, p, a, r in per_class_rows))
    logger.info("  Would fallback: %d/%d (%.1f%%)",
                (~high_conf).sum(), len(y_test_enc), (~high_conf).mean() * 100)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, f"{prefix}_{attr_name}_xgb.pkl"), "wb") as f:
        pickle.dump(final_clf, f)
    # Save raw uncalibrated XGB alongside for analysis (notebook ablations)
    if final_clf is not clf:
        with open(os.path.join(MODELS_DIR, f"{prefix}_{attr_name}_xgb_raw.pkl"), "wb") as f:
            pickle.dump(clf, f)
    with open(os.path.join(MODELS_DIR, f"{prefix}_{attr_name}_le.pkl"), "wb") as f:
        pickle.dump(le, f)
    _save_calibration_data(prefix, attr_name, ece_raw, ece_cal, bins_raw, bins_cal)

    return final_clf, le, best_threshold


def train_binary(X_train, X_test, y_train, y_test, attr_name: str, prefix: str,
                 do_calibrate: bool = True, calibration_method: str = "sigmoid"):
    """Train binary classifier with regularization, early stopping, and calibration."""
    if do_calibrate:
        X_fit, X_calib, y_fit, y_calib = _stratified_split_safe(
            X_train, y_train,
            test_size=0.10, random_state=RANDOM_STATE, stratify=y_train,
        )
    else:
        X_fit, y_fit = X_train, y_train
        X_calib, y_calib = None, None

    X_tr, X_val, y_tr, y_val = _stratified_split_safe(
        X_fit, y_fit,
        test_size=0.15, random_state=RANDOM_STATE, stratify=y_fit,
    )

    clf = XGBClassifier(
        **BINARY_PARAMS,
        eval_metric="logloss",
        scale_pos_weight=(y_fit == 0).sum() / max((y_fit == 1).sum(), 1),
    )
    try:
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    except Exception as e:
        logger.warning("%s: early-stop val incompatible (%s) — fitting without eval_set",
                       attr_name, str(e)[:100])
        clf_no_es = XGBClassifier(
            **{k: v for k, v in BINARY_PARAMS.items() if k != "early_stopping_rounds"},
            eval_metric="logloss",
            scale_pos_weight=(y_fit == 0).sum() / max((y_fit == 1).sum(), 1),
        )
        clf_no_es.fit(X_tr, y_tr, verbose=False)
        clf = clf_no_es

    y_proba_raw = clf.predict_proba(X_test)
    ece_raw, bins_raw = compute_ece(y_test.values, y_proba_raw)

    final_clf = clf
    ece_cal, bins_cal = None, None
    if do_calibrate and X_calib is not None:
        cal_clf, ok = calibrate(clf, X_calib, y_calib, method=calibration_method)
        if ok:
            final_clf = cal_clf
            y_proba_cal = cal_clf.predict_proba(X_test)
            ece_cal, bins_cal = compute_ece(y_test.values, y_proba_cal)

    y_proba_pos = final_clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba_pos >= 0.5).astype(int)
    confidences = np.maximum(y_proba_pos, 1 - y_proba_pos)
    # Threshold tuned on X_val (internal training split), NOT X_test.
    best_threshold = find_best_threshold(final_clf, X_val, y_val)

    cal_info = f", ECE {ece_raw:.3f}→{ece_cal:.3f}" if ece_cal is not None else f", ECE {ece_raw:.3f} (uncalibrated)"
    best_iter = getattr(clf, "best_iteration", None)
    iter_str = f"best_iter={best_iter}" if best_iter is not None else "no_eval_set"
    logger.info("=== %s (%s, threshold=%.2f%s) ===",
                attr_name, iter_str, best_threshold, cal_info)
    report = classification_report(y_test, y_pred)
    for line in report.strip().split("\n"):
        logger.info("  %s", line)

    high_conf = confidences >= best_threshold
    if high_conf.sum() > 0:
        acc = (y_pred[high_conf] == y_test.values[high_conf]).mean()
        f1 = f1_score(y_test.values[high_conf], y_pred[high_conf], average="macro", zero_division=0)
        logger.info("  conf >= %.2f: %d/%d products, acc=%.3f, macro_f1=%.3f",
                     best_threshold, high_conf.sum(), len(y_test), acc, f1)
    logger.info("  Would fallback: %d/%d (%.1f%%)",
                (~high_conf).sum(), len(y_test), (~high_conf).mean() * 100)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, f"{prefix}_{attr_name}_xgb.pkl"), "wb") as f:
        pickle.dump(final_clf, f)
    if final_clf is not clf:
        with open(os.path.join(MODELS_DIR, f"{prefix}_{attr_name}_xgb_raw.pkl"), "wb") as f:
            pickle.dump(clf, f)
    _save_calibration_data(prefix, attr_name, ece_raw, ece_cal, bins_raw, bins_cal)

    return final_clf, best_threshold


def recompute_calibration_only(args):
    """Regenerate calibration JSONs without touching XGBoost pkls.

    Loads existing models/{prefix}_{attr}_xgb.pkl + _le.pkl, recomputes
    predict_proba on the same test split that the original training used,
    saves updated {prefix}_{attr}_calibration.json (ECE + bins).

    The split is reproduced via train_test_split(np.arange(len(df)), ...,
    random_state=RANDOM_STATE), matching main()'s global split. Calibration is
    therefore strictly comparable across runs only when df length and ordering
    are unchanged from the original training.

    Note: this only recomputes ECE/bins for the already-saved classifier
    (which may itself be a CalibratedClassifierCV wrapper). Re-running isotonic
    calibration on top is out of scope; per-attr decision per pre-registration
    rule is applied later via src/eval/compare_ece.py.
    """
    setup_logging()
    gold_infix = "_gold" if args.use_gold else ""
    prefix = args.category + gold_infix + args.model_suffix

    config = CATEGORY_CONFIG[args.category]
    ss_filename = config["silver_standard"]
    if args.use_extended:
        ss_filename = ss_filename.replace("_silver_standard", "_silver_extended")
    ss_path = os.path.join(PROCESSED_DIR, ss_filename)
    cache_path = os.path.join(PROCESSED_DIR, config["embeddings_cache"])

    if not os.path.exists(ss_path):
        logger.error("Silver standard not found: %s", ss_path)
        return

    df = pd.read_parquet(ss_path)
    logger.info("[calibration-only] Loaded %d products", len(df))

    if not os.path.exists(cache_path):
        logger.info("[calibration-only] embeddings cache missing — recomputing")
        texts = build_text(df)
        X_all = get_embeddings(texts, cache_path)
    else:
        X_all = np.load(cache_path)
        if X_all.shape[0] != len(df):
            logger.warning("[calibration-only] embeddings shape %s != df len %d — "
                           "recomputing without cache", X_all.shape, len(df))
            texts = build_text(df)
            X_all = get_embeddings(texts, cache_path=None)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    X_test_all = X_all[test_idx]

    regenerated = 0
    for attr_name, attr_config in config["classifiers"].items():
        xgb_path = os.path.join(MODELS_DIR, f"{prefix}_{attr_name}_xgb.pkl")
        if not os.path.exists(xgb_path):
            logger.warning("[calibration-only] %s: no xgb pkl at %s — skipping",
                           attr_name, xgb_path)
            continue
        if attr_name not in df.columns:
            logger.warning("[calibration-only] %s: not in silver — skipping", attr_name)
            continue

        with open(xgb_path, "rb") as f:
            clf = pickle.load(f)

        if attr_config["type"] == "multiclass":
            le_path = os.path.join(MODELS_DIR, f"{prefix}_{attr_name}_le.pkl")
            if not os.path.exists(le_path):
                logger.warning("[calibration-only] %s: no LE — skipping", attr_name)
                continue
            with open(le_path, "rb") as f:
                le = pickle.load(f)

            y_train = train_df[attr_name].copy()
            y_test = test_df[attr_name].copy()

            train_mask = y_train.notna()
            counts = y_train[train_mask].value_counts()
            valid = counts[counts >= attr_config["min_samples"]].index
            test_mask = y_test.notna() & y_test.isin(valid)

            # Restrict test to classes known to the LabelEncoder
            known = set(le.classes_)
            test_mask = test_mask & y_test.isin(known)

            if test_mask.sum() < 5:
                logger.warning("[calibration-only] %s: test set too small (%d) — skipping",
                               attr_name, int(test_mask.sum()))
                continue

            X_test = X_test_all[test_mask.values]
            y_test_enc = le.transform(y_test[test_mask])
            y_proba = clf.predict_proba(X_test)

            ece, bins = compute_ece(y_test_enc, y_proba)

        elif attr_config["type"] == "binary":
            y_test = test_df[attr_name].astype(bool).astype(int)
            X_test = X_test_all
            y_proba = clf.predict_proba(X_test)
            ece, bins = compute_ece(y_test.values, y_proba)

        else:
            continue

        # The classifier loaded from pkl is the final (possibly calibrated) one,
        # so the ECE we just computed corresponds to "ece_calibrated" semantically.
        # However, original code semantics save BOTH ece_raw (pre-cal) and ece_calibrated
        # (post-cal). After silver-fix we no longer have the raw pre-cal proba, so we
        # write the recomputed ECE into both slots and rely on prev calibrated value
        # in the baseline snapshot for delta comparison.
        prev_path = os.path.join(MODELS_DIR, f"{prefix}_{attr_name}_calibration.json")
        out = {
            "attr": attr_name,
            "ece_raw": ece,
            "ece_calibrated": ece,
            "bins_raw": bins,
            "bins_calibrated": bins,
            "regenerated_calibration_only": True,
        }
        with open(prev_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("[calibration-only] %s: ECE=%.4f (n_test=%d) → %s",
                    attr_name, ece, len(y_proba), os.path.basename(prev_path))
        regenerated += 1

    logger.info("[calibration-only] done: regenerated %d calibration JSONs under prefix %s",
                regenerated, prefix)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()))
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip CalibratedClassifierCV wrapping (saves raw XGB only)")
    parser.add_argument("--calibration-method", choices=["sigmoid", "isotonic"],
                        default="sigmoid",
                        help="Sigmoid (Platt, default) is robust on small N; "
                             "isotonic is non-parametric but needs ≥500 calibration samples")
    parser.add_argument("--use-extended", action="store_true",
                        help="Read silver_extended_*.parquet вместо silver_standard "
                             "(coverage-extended labels from src/eval/coverage_extension.py)")
    parser.add_argument("--model-suffix", default="",
                        help="Suffix добавляемый к prefix моделей (e.g. '_extended'). "
                             "Позволяет обучить параллельный set models без overwrite оригиналов.")
    parser.add_argument("--use-gold", action="store_true",
                        help="Use verified gold (consensus_gold_v1_emulated.parquet) for silver_strong attrs; "
                             "split is brand-disjoint from {cat}_gold_split.parquet. Implies "
                             "limiting train to gold-tier silver_strong attrs only.")
    parser.add_argument("--gold-attrs", nargs="*", default=None,
                        help="Restrict to subset of silver_strong attrs (cat-relative). "
                             "If unset and --use-gold — picks all silver_strong for the category.")
    parser.add_argument("--calibration-only", action="store_true",
                        help="Skip XGBoost retraining; only regenerate calibration JSONs (ECE + isotonic) "
                             "from existing models/{cat}_stratified_{attr}_xgb.pkl on current silver/embeddings.")
    parser.add_argument("--with-tfidf", action="store_true",
                        help="Расширить признаковое пространство XGBoost разреженными TF-IDF n-граммами "
                             "(1, 2; top-5000) поверх плотных SBERT-эмбеддингов. Векторайзер сохраняется "
                             "как {prefix}_tfidf.pkl и загружается в infer.py при наличии файла. "
                             "Рекомендуется вместе с --model-suffix _hybrid, чтобы не перезаписать "
                             "базовые SBERT-only модели.")
    args = parser.parse_args()
    do_calibrate = not args.no_calibrate
    calibration_method = args.calibration_method

    if args.calibration_only:
        recompute_calibration_only(args)
        return
    # --use-gold добавляет "_gold" к suffix, чтобы не перетереть оригинальные silver-модели
    gold_infix = "_gold" if args.use_gold else ""
    MODEL_PREFIX = args.category + gold_infix + args.model_suffix

    config = CATEGORY_CONFIG[args.category]
    ss_filename = config["silver_standard"]
    if args.use_extended:
        ss_filename = ss_filename.replace("_silver_standard", "_silver_extended")
    ss_path = os.path.join(PROCESSED_DIR, ss_filename)
    cache_path = os.path.join(PROCESSED_DIR, config["embeddings_cache"])

    if not os.path.exists(ss_path):
        logger.error("Silver standard not found: %s", ss_path)
        return

    df = pd.read_parquet(ss_path)
    logger.info("Loaded %d products from silver standard", len(df))
    # Сохраняем оригинальный positional index — embeddings кэш индексирован
    # по этим позициям; при --use-gold отфильтрованный df ужмётся и нам нужно
    # будет переиндексировать X_all по сохранённым позициям.
    df["_orig_idx"] = np.arange(len(df))

    # === --use-gold logic (router-rescue, 2026-05-13) ===
    # Перебиваем silver labels значениями из consensus_gold_v1_emulated.parquet для silver_strong
    # атрибутов и режем выборку до train-сплита из {cat}_gold_split.parquet
    # (brand-disjoint). Обучение ограничивается silver_strong атрибутами.
    attrs_override = None
    if args.use_gold:
        from src.eval.validation_sources import VALIDATION_SOURCE, get_tier
        cat_clean = args.category.replace("_stratified", "")
        consensus_path = os.path.join(PROCESSED_DIR, "consensus_gold_v1_emulated.parquet")
        split_path = os.path.join(PROCESSED_DIR, f"{cat_clean}_gold_split.parquet")
        if not os.path.exists(consensus_path):
            raise FileNotFoundError(
                f"--use-gold requires {consensus_path}. Run Phase A first "
                "(python -m src.eval.build_consensus_gold)."
            )
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"--use-gold requires {split_path}. Run Phase B (B.3) first."
            )
        gold = pd.read_parquet(consensus_path)
        gold = gold[(gold.category == cat_clean) & (gold.gt_consensus.notna())]
        split_df = pd.read_parquet(split_path)
        split_df["code"] = split_df["code"].astype(str)
        gold["code"] = gold["code"].astype(str)

        # silver_strong атрибуты для этой категории
        ss_attrs = [a for (c, a) in VALIDATION_SOURCE
                    if c == cat_clean and get_tier(c, a).value == "silver_strong"]
        if args.gold_attrs:
            ss_attrs = [a for a in ss_attrs if a in args.gold_attrs]
        logger.info("[--use-gold] %s silver_strong attrs: %s", cat_clean, ss_attrs)

        # Long format gold → wide; overlay поверх silver
        gold_wide = gold[gold.attr.isin(ss_attrs)].pivot_table(
            index="code", columns="attr", values="gt_consensus", aggfunc="first"
        ).reset_index()

        df["code"] = df["code"].astype(str)
        df = df.merge(split_df, on="code", how="inner")
        for attr in ss_attrs:
            if attr in gold_wide.columns:
                df = df.merge(
                    gold_wide[["code", attr]].rename(columns={attr: f"{attr}_gold"}),
                    on="code", how="left",
                )
                # gold перебивает silver там, где gold не-null
                if attr in df.columns:
                    df[attr] = df[f"{attr}_gold"].combine_first(df[attr])
                else:
                    df[attr] = df[f"{attr}_gold"]
                df = df.drop(columns=[f"{attr}_gold"])
                # Normalize case: gt_consensus в consensus_gold lowercase,
                # silver сохраняет исходный case → combine_first даёт mix
                # ("Wheat"/"wheat" → две разные метки для LabelEncoder).
                df[attr] = df[attr].where(df[attr].isna(),
                                           df[attr].astype(str).str.lower())

        # Train slice = brand-disjoint train split
        df = df[df["split"] == "train"].copy().reset_index(drop=True)
        logger.info("[--use-gold] training on %d products from train split", len(df))

        # Ограничить атрибуты для обучения
        attrs_override = ss_attrs

    # Global train/test split — aligned with run_experiments.py.
    # Note: random split (without stratify=) chosen for backward-compat alignment.
    # ML diagnostics показывает: для multiclass с >5 классами random split
    # может пропускать редкие классы из train (см. notebooks/07 §2). Если будете
    # переучивать с нуля — добавьте stratify=df['main_attr'].fillna('_').
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    logger.info("Global split: %d train, %d test", len(train_df), len(test_df))

    # Diagnostic: check class-balance drift между train/test для каждого attr
    for attr_name in config["classifiers"]:
        if attr_name not in df.columns:
            continue
        tr_dist = train_df[attr_name].value_counts(normalize=True, dropna=False).sort_index()
        te_dist = test_df[attr_name].value_counts(normalize=True, dropna=False).sort_index()
        common = tr_dist.index.intersection(te_dist.index)
        l1 = (tr_dist.reindex(common).fillna(0) - te_dist.reindex(common).fillna(0)).abs().sum()
        train_only = set(tr_dist.index) - set(te_dist.index)
        test_only = set(te_dist.index) - set(tr_dist.index)
        if l1 > 0.10 or train_only or test_only:
            logger.warning(
                "  [%s] class-balance drift: L1=%.3f, train_only=%s, test_only=%s "
                "(consider stratify= for this attr)",
                attr_name, float(l1), train_only or "—", test_only or "—",
            )

    # Embeddings — закэшированы по позициям ОРИГИНАЛЬНОГО df. Если --use-gold
    # отфильтровал df, выбираем embeddings по сохранённым _orig_idx, иначе
    # пересчитываем для отфильтрованного подмножества (без записи в кэш).
    if args.use_gold and os.path.exists(cache_path):
        X_all_full = np.load(cache_path)
        logger.info("Loaded cached embeddings %s, reindexing to gold subset", X_all_full.shape)
        X_all = X_all_full[df["_orig_idx"].values]
    elif args.use_gold:
        texts = build_text(df)
        X_all = get_embeddings(texts, cache_path=None)
    else:
        texts = build_text(df)
        X_all = get_embeddings(texts, cache_path)
    logger.info("Embedding shape: %s", X_all.shape)

    if args.with_tfidf:
        # TF-IDF, сжатый TruncatedSVD до 128-dim → плотная numpy-матрица.
        # Старый hstack(csr(SBERT_dense), tfidf_sparse) был патологически медленным:
        # 768-dim dense зашитый в csr давал миллионы non-zero, XGB итерировал каждый.
        # Через SVD получаем 896-dim plain dense (SBERT 768 + svd128), XGB быстр.
        from sklearn.decomposition import TruncatedSVD
        texts_all = build_text(df)
        train_texts = [texts_all[i] for i in train_idx]
        vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), lowercase=True,
        )
        X_tfidf_train_sparse = vectorizer.fit_transform(train_texts)
        X_tfidf_test_sparse = vectorizer.transform([texts_all[i] for i in test_idx])
        logger.info("TF-IDF raw: train=%s, test=%s, vocab=%d",
                    X_tfidf_train_sparse.shape, X_tfidf_test_sparse.shape,
                    len(vectorizer.vocabulary_))
        n_components = min(128, X_tfidf_train_sparse.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_tfidf_train = svd.fit_transform(X_tfidf_train_sparse).astype(np.float32)
        X_tfidf_test = svd.transform(X_tfidf_test_sparse).astype(np.float32)
        ev = float(svd.explained_variance_ratio_.sum())
        logger.info("TF-IDF SVD-%d: explained variance ratio=%.3f, dense shape train=%s",
                    n_components, ev, X_tfidf_train.shape)
        vec_path = os.path.join(MODELS_DIR, f"{MODEL_PREFIX}_tfidf.pkl")
        svd_path = os.path.join(MODELS_DIR, f"{MODEL_PREFIX}_tfidf_svd.pkl")
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(vec_path, "wb") as f:
            pickle.dump(vectorizer, f)
        with open(svd_path, "wb") as f:
            pickle.dump(svd, f)
        logger.info("Saved TF-IDF vectorizer+SVD to %s, %s", vec_path, svd_path)
        X_train_all = np.hstack([X_all[train_idx], X_tfidf_train])
        X_test_all = np.hstack([X_all[test_idx], X_tfidf_test])
        logger.info("Hybrid features (dense): train=%s, test=%s",
                    X_train_all.shape, X_test_all.shape)
    else:
        X_train_all = X_all[train_idx]
        X_test_all = X_all[test_idx]

    thresholds = {}
    for attr_name, attr_config in config["classifiers"].items():
        if attrs_override is not None and attr_name not in attrs_override:
            continue
        if attr_name not in df.columns:
            logger.warning("Skipping %s: not in silver standard", attr_name)
            continue

        if attr_config["type"] == "multiclass":
            y_train = train_df[attr_name].copy()
            y_test = test_df[attr_name].copy()

            train_mask = y_train.notna()
            counts = y_train[train_mask].value_counts()
            valid = counts[counts >= attr_config["min_samples"]].index
            if "exclude_classes" in attr_config:
                valid = valid.difference(attr_config["exclude_classes"])
            train_mask = train_mask & y_train.isin(valid)
            test_mask = y_test.notna() & y_test.isin(valid)

            if y_train[train_mask].nunique() < 2:
                logger.warning("Skipping %s: less than 2 classes", attr_name)
                continue

            # Один failed attr не должен убивать весь pipeline. Чаще всего падает
            # при разреженных классах: внутренний split (calibration/early-stop val)
            # может оставить <2 классов в subsetе → XGBoost mlogloss падает.
            try:
                _, _, t = train_multiclass(
                    X_train_all[train_mask.values], X_test_all[test_mask.values],
                    y_train[train_mask], y_test[test_mask],
                    attr_name, MODEL_PREFIX,
                    do_calibrate=do_calibrate, calibration_method=calibration_method,
                )
                thresholds[attr_name] = t
            except Exception as e:
                logger.error("Failed training %s: %s — skipping", attr_name, e)

        elif attr_config["type"] == "binary":
            y_train = train_df[attr_name].astype(bool).astype(int)
            y_test = test_df[attr_name].astype(bool).astype(int)

            if y_train.sum() < attr_config["min_positive"]:
                logger.warning("Skipping %s: only %d positive samples", attr_name, y_train.sum())
                continue

            try:
                _, t = train_binary(
                    X_train_all, X_test_all,
                    y_train, y_test,
                    attr_name, MODEL_PREFIX,
                    do_calibrate=do_calibrate, calibration_method=calibration_method,
                )
                thresholds[attr_name] = t
            except Exception as e:
                logger.error("Failed training %s: %s — skipping", attr_name, e)

    thresh_path = os.path.join(MODELS_DIR, f"{MODEL_PREFIX}_thresholds.pkl")
    with open(thresh_path, "wb") as f:
        pickle.dump(thresholds, f)
    logger.info("Optimal thresholds: %s", thresholds)
    logger.info("Done. Models saved to models/")


if __name__ == "__main__":
    main()
