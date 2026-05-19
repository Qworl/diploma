"""Layer 0 v3 — LightGBM on TF-IDF n-grams.

Per the EXP5b finding (LightGBM TF-IDF beats XGB+embeddings by +2.66pp on
Layer 1.5), apply the same architecture to Layer 0 category router.
Targeted at fixing the catastrophic LOCO recall (chocolate=9%, pasta=21%)
where embeddings don't generalize to unseen brands.

Outputs (saved alongside v1 artifacts, _v3 suffix):
  models/category_router_v3_lgbm.pkl
  models/category_router_v3_vec.pkl
  models/category_router_v3_le.pkl
  models/category_router_v3_threshold.json
  models/category_router_v3_meta.json
  models/category_router_v3_loco.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from datetime import datetime, timezone
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.common import MODELS_DIR, PROCESSED_DIR, RAW_DIR
from src.pipeline.category_router.constants import (
    ROUTER_CLASSES,
    ROUTER_INPUT_FIELDS,
)
from src.pipeline.category_router.data import load_positive
from src.pipeline.category_router.ood_sampler import sample_ood
from src.pipeline.category_router.split import brand_disjoint_split

logger = logging.getLogger(__name__)


def _build_text(row) -> str:
    return " ".join(str(row[k] or "") for k in ROUTER_INPUT_FIELDS).strip() or " "


def _calibrate_threshold(proba_known: np.ndarray, target_fpr: float) -> tuple[float, float]:
    max_p = proba_known.max(axis=1)
    t = float(np.quantile(max_p, target_fpr))
    fpr = float(np.mean(max_p < t))
    return t, fpr


def _evaluate_loco(
    full_df: pd.DataFrame, target_fpr: float, seed: int, models_dir: str,
) -> list[dict]:
    """Leave-one-category-out: train on 6 cats, test if model says OOD for the held-out 7th."""
    rows: list[dict] = []
    for held_out in ROUTER_CLASSES:
        train_part = full_df[full_df["category_label"] != held_out].copy()
        loco_part = full_df[full_df["category_label"] == held_out].copy()
        if len(loco_part) == 0:
            continue

        train_texts = [_build_text(r) for _, r in train_part.iterrows()]
        loco_texts = [_build_text(r) for _, r in loco_part.iterrows()]

        sub_classes = [c for c in ROUTER_CLASSES if c != held_out]
        le_sub = LabelEncoder().fit(sub_classes)
        y_train = le_sub.transform(train_part["category_label"])

        vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=20_000, sublinear_tf=True,
        )
        X_train = vec.fit_transform(train_texts)
        X_loco = vec.transform(loco_texts)

        n = len(sub_classes)
        clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05, num_leaves=31,
            min_child_samples=5, verbose=-1,
            objective="multiclass", num_class=n,
        )
        clf.fit(X_train, y_train)
        proba_loco = clf.predict_proba(X_loco)

        # Use held-out category's own threshold per a fresh calibration on a sub-train val
        # For simplicity reuse global threshold from primary trained model loaded later
        # — record max proba; recall = fraction of LOCO with max_proba < threshold
        loco_max = proba_loco.max(axis=1)

        # Compute a self-threshold per LOCO model (5% FPR on its own validation split of train)
        X_tr_inner, X_val_inner = train_test_split(
            X_train, test_size=0.1, random_state=seed,
        )
        proba_val = clf.predict_proba(X_val_inner)
        thr, _ = _calibrate_threshold(proba_val, target_fpr)

        recall = float(np.mean(loco_max < thr))
        rows.append({
            "leave_out_category": held_out,
            "ood_recall": recall,
            "mean_confidence_on_loco": float(loco_max.mean()),
            "n_loco_examples": int(len(loco_part)),
            "loco_threshold": thr,
        })
        logger.info("LOCO[%s]: ood_recall=%.3f n=%d threshold=%.3f",
                    held_out, recall, len(loco_part), thr)

    return rows


def train_router_v3(
    *,
    processed_dir: str = PROCESSED_DIR,
    off_parquet: str = os.path.join(RAW_DIR, "en.openfoodfacts.org.products.parquet"),
    n_per_class: int = 5000,
    ood_fraction: float = 1.0,
    seed: int = 42,
    models_dir: str = MODELS_DIR,
    target_fpr_known: float = 0.05,
    skip_loco: bool = False,
) -> dict[str, Any]:
    logger.info("Loading positive samples (n_per_class=%d)", n_per_class)
    pos = load_positive(processed_dir=processed_dir, n_per_class=n_per_class, seed=seed)
    n_ood = int(ood_fraction * n_per_class)
    logger.info("Sampling OOD (n=%d)", n_ood)
    ood = sample_ood(off_parquet=off_parquet, n=n_ood, seed=seed)
    ood = ood[list(ROUTER_INPUT_FIELDS) + ["category_label", "brand"]]

    pos_train, pos_test = brand_disjoint_split(pos, test_size=0.2, seed=seed)
    ood_train, ood_test = brand_disjoint_split(ood, test_size=0.2, seed=seed)
    train_df = pd.concat([pos_train, ood_train], ignore_index=True)
    test_df = pd.concat([pos_test, ood_test], ignore_index=True)
    train_df["is_ood"] = (train_df["category_label"] == "unknown").astype(bool)
    test_df["is_ood"] = (test_df["category_label"] == "unknown").astype(bool)

    # Build texts
    logger.info("Building texts (train=%d, test=%d)", len(train_df), len(test_df))
    train_texts = [_build_text(r) for _, r in train_df.iterrows()]
    test_texts = [_build_text(r) for _, r in test_df.iterrows()]

    known_mask_train = (train_df["category_label"] != "unknown").to_numpy()
    train_texts_known = [t for t, m in zip(train_texts, known_mask_train) if m]
    y_train_known_str = train_df.loc[known_mask_train, "category_label"].to_numpy()

    le = LabelEncoder().fit(list(ROUTER_CLASSES))
    y_train_known = le.transform(y_train_known_str)

    logger.info("Fitting TF-IDF vectorizer on %d known-train texts", len(train_texts_known))
    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=20_000, sublinear_tf=True,
    )
    X_train_known = vec.fit_transform(train_texts_known)

    # Inner train/val split for early stopping + threshold calibration
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_known, y_train_known,
        test_size=0.1, random_state=seed, stratify=y_train_known,
    )

    logger.info("Training LightGBM (X_tr shape=%s, n_classes=%d)", X_tr.shape, len(ROUTER_CLASSES))
    clf = lgb.LGBMClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05, num_leaves=63,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, verbose=-1,
        objective="multiclass", num_class=len(ROUTER_CLASSES),
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    proba_val = clf.predict_proba(X_val)
    threshold, fpr_on_known = _calibrate_threshold(proba_val, target_fpr_known)

    # Save artifacts
    suffix = "v3"
    with open(os.path.join(models_dir, f"category_router_{suffix}_lgbm.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(models_dir, f"category_router_{suffix}_vec.pkl"), "wb") as f:
        pickle.dump(vec, f)
    with open(os.path.join(models_dir, f"category_router_{suffix}_le.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(models_dir, f"category_router_{suffix}_threshold.json"), "w") as f:
        json.dump({
            "threshold": threshold,
            "fpr_on_known": fpr_on_known,
            "calibrated_on": "validation_split",
            "n_validation": int(X_val.shape[0]),
            "target_fpr": target_fpr_known,
        }, f, indent=2)

    # Test eval
    X_test = vec.transform(test_texts)
    known_mask_test = (test_df["category_label"] != "unknown").to_numpy()
    y_test_known_str = test_df.loc[known_mask_test, "category_label"].to_numpy()
    y_test_known = le.transform(y_test_known_str)

    proba_test_full = clf.predict_proba(X_test)
    proba_test_known = proba_test_full[known_mask_test]
    y_pred_known = proba_test_known.argmax(axis=1)
    test_acc = float(accuracy_score(y_test_known, y_pred_known))
    test_f1 = float(f1_score(y_test_known, y_pred_known, average="macro"))
    per_class_p, per_class_r, per_class_f, _ = precision_recall_fscore_support(
        y_test_known, y_pred_known,
        labels=list(range(len(ROUTER_CLASSES))), zero_division=0,
    )
    per_class_f1 = {
        ROUTER_CLASSES[i]: float(per_class_f[i]) for i in range(len(ROUTER_CLASSES))
    }

    ood_scores = 1.0 - proba_test_full.max(axis=1)
    y_test_ood = (~known_mask_test).astype(int)
    if 0 < y_test_ood.sum() < len(y_test_ood):
        ood_auroc = float(roc_auc_score(y_test_ood, ood_scores))
    else:
        ood_auroc = float("nan")

    logger.info(
        "Test: acc=%.4f f1_macro=%.4f ood_auroc=%.4f",
        test_acc, test_f1, ood_auroc,
    )
    logger.info("Per-class F1: %s", per_class_f1)

    # LOCO eval
    loco_rows: list[dict] = []
    if not skip_loco:
        logger.info("Running LOCO eval...")
        full_pos_with_loco = pd.concat([pos_train, pos_test], ignore_index=True)
        loco_rows = _evaluate_loco(full_pos_with_loco, target_fpr_known, seed, models_dir)
        loco_df = pd.DataFrame(loco_rows)
        loco_df.to_parquet(os.path.join(models_dir, f"category_router_{suffix}_loco.parquet"))

    meta: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "lightgbm_tfidf_ngrams_1_2",
        "n_train_known": int(known_mask_train.sum()),
        "n_train_ood": int((~known_mask_train).sum()),
        "n_test_known": int(known_mask_test.sum()),
        "n_test_ood": int((~known_mask_test).sum()),
        "n_per_class": n_per_class,
        "ood_fraction": ood_fraction,
        "seed": seed,
        "tfidf_max_features": 20_000,
        "tfidf_ngram_range": [1, 2],
        "threshold": threshold,
        "fpr_on_known": fpr_on_known,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "test_per_class_f1": per_class_f1,
        "ood_auroc": ood_auroc,
        "loco_rows": loco_rows,
    }
    with open(os.path.join(models_dir, f"category_router_{suffix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=5000)
    p.add_argument("--ood-fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--skip-loco", action="store_true")
    args = p.parse_args()
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout, force=True,
    )
    meta = train_router_v3(
        n_per_class=args.n_per_class,
        ood_fraction=args.ood_fraction,
        seed=args.seed,
        target_fpr_known=args.target_fpr,
        skip_loco=args.skip_loco,
    )
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    _cli()
