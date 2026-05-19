"""CLI: python -m src.pipeline.category_router.train

Trains Layer 0 category router.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.common import MODELS_DIR, PROCESSED_DIR, RAW_DIR
from src.pipeline.category_router.constants import (
    EMBEDDINGS_NPY,
    ROUTER_CLASSES,
    ROUTER_INPUT_FIELDS,
    TEST_PARQUET,
    TRAIN_PARQUET,
)
from src.pipeline.category_router.data import load_positive
from src.pipeline.category_router.ood_sampler import sample_ood
from src.pipeline.category_router.split import brand_disjoint_split

logger = logging.getLogger(__name__)


def _build_text(row) -> str:
    return " ".join(str(row[k] or "") for k in ROUTER_INPUT_FIELDS).strip() or " "


def _embed_texts(embedder, df: pd.DataFrame) -> np.ndarray:
    texts = [_build_text(row) for _, row in df.iterrows()]
    return np.asarray(embedder.encode(texts, show_progress_bar=False))


def _calibrate_threshold(
    proba_known: np.ndarray, target_fpr: float
) -> tuple[float, float]:
    """Pick threshold so that fraction of known with max_softmax < t == target_fpr."""
    max_p = proba_known.max(axis=1)
    t = float(np.quantile(max_p, target_fpr))
    fpr = float(np.mean(max_p < t))
    return t, fpr


def train_router(
    *,
    processed_dir: str = PROCESSED_DIR,
    off_parquet: str = os.path.join(RAW_DIR, "en.openfoodfacts.org.products.parquet"),
    embedder=None,
    n_per_class: int = 5000,
    ood_fraction: float = 1.0,
    seed: int = 42,
    models_dir: str = MODELS_DIR,
    train_parquet: str = TRAIN_PARQUET,
    test_parquet: str = TEST_PARQUET,
    embeddings_npy: str = EMBEDDINGS_NPY,
    target_fpr_known: float = 0.05,
) -> dict[str, Any]:
    """Train category router end-to-end. Returns meta dict."""
    if embedder is None:
        from sentence_transformers import SentenceTransformer

        from src.common import EMBEDDING_MODEL
        embedder = SentenceTransformer(EMBEDDING_MODEL)

    pos = load_positive(processed_dir=processed_dir,
                        n_per_class=n_per_class, seed=seed)
    n_ood = int(ood_fraction * n_per_class)
    ood = sample_ood(off_parquet=off_parquet, n=n_ood, seed=seed)
    ood = ood[list(ROUTER_INPUT_FIELDS) + ["category_label", "brand"]]

    pos_train, pos_test = brand_disjoint_split(pos, test_size=0.2, seed=seed)
    ood_train, ood_test = brand_disjoint_split(ood, test_size=0.2, seed=seed)
    train_df = pd.concat([pos_train, ood_train], ignore_index=True)
    test_df = pd.concat([pos_test, ood_test], ignore_index=True)
    train_df["is_ood"] = (train_df["category_label"] == "unknown").astype(bool)
    test_df["is_ood"] = (test_df["category_label"] == "unknown").astype(bool)
    train_df.to_parquet(train_parquet)
    test_df.to_parquet(test_parquet)

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    embeddings = _embed_texts(embedder, full_df)
    np.save(embeddings_npy, embeddings)
    n_train = len(train_df)
    X_train_all = embeddings[:n_train]
    X_test_all = embeddings[n_train:]

    known_mask_train = (train_df["category_label"] != "unknown").to_numpy()
    X_train_known = X_train_all[known_mask_train]
    y_train_known_str = train_df.loc[known_mask_train, "category_label"].to_numpy()

    X_tr, X_val, y_tr_str, y_val_str = train_test_split(
        X_train_known, y_train_known_str,
        test_size=0.1, random_state=seed, stratify=y_train_known_str,
    )
    le = LabelEncoder().fit(list(ROUTER_CLASSES))
    y_tr = le.transform(y_tr_str)
    y_val = le.transform(y_val_str)

    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, gamma=0.1,
        objective="multi:softprob", eval_metric="mlogloss",
        early_stopping_rounds=20, random_state=seed, n_jobs=1,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    proba_val = clf.predict_proba(X_val)
    threshold, fpr_on_known = _calibrate_threshold(proba_val, target_fpr_known)
    with open(os.path.join(models_dir, "category_router_threshold.json"), "w") as f:
        json.dump({
            "threshold": threshold,
            "fpr_on_known": fpr_on_known,
            "calibrated_on": "validation_split",
            "n_validation": int(len(X_val)),
            "target_fpr": target_fpr_known,
        }, f, indent=2)

    with open(os.path.join(models_dir, "category_router_xgb.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(models_dir, "category_router_le.pkl"), "wb") as f:
        pickle.dump(le, f)

    known_mask_test = (test_df["category_label"] != "unknown").to_numpy()
    X_test_known = X_test_all[known_mask_test]
    y_test_known = le.transform(
        test_df.loc[known_mask_test, "category_label"].to_numpy()
    )
    proba_test_known = clf.predict_proba(X_test_known)
    y_pred_known = proba_test_known.argmax(axis=1)
    test_acc = float(accuracy_score(y_test_known, y_pred_known))
    test_f1 = float(f1_score(y_test_known, y_pred_known, average="macro"))
    per_class_p, per_class_r, per_class_f, _ = precision_recall_fscore_support(
        y_test_known, y_pred_known, labels=list(range(len(ROUTER_CLASSES))),
        zero_division=0,
    )
    per_class_f1 = {
        ROUTER_CLASSES[i]: float(per_class_f[i]) for i in range(len(ROUTER_CLASSES))
    }

    proba_test_full = clf.predict_proba(X_test_all)
    ood_scores = 1.0 - proba_test_full.max(axis=1)
    y_test_ood = (~known_mask_test).astype(int)
    if y_test_ood.sum() > 0 and y_test_ood.sum() < len(y_test_ood):
        ood_auroc = float(roc_auc_score(y_test_ood, ood_scores))
    else:
        ood_auroc = float("nan")

    meta: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_known": int(known_mask_train.sum()),
        "n_train_ood": int((~known_mask_train).sum()),
        "n_test_known": int(known_mask_test.sum()),
        "n_test_ood": int((~known_mask_test).sum()),
        "n_per_class": n_per_class,
        "ood_fraction": ood_fraction,
        "seed": seed,
        "threshold": threshold,
        "fpr_on_known": fpr_on_known,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "test_per_class_f1": per_class_f1,
        "ood_auroc": ood_auroc,
    }
    with open(os.path.join(models_dir, "category_router_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=5000)
    p.add_argument("--ood-fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-fpr", type=float, default=0.05)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    meta = train_router(
        n_per_class=args.n_per_class,
        ood_fraction=args.ood_fraction,
        seed=args.seed,
        target_fpr_known=args.target_fpr,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    _cli()
