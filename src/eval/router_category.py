"""CLI: python -m src.eval.router_category --mode {standard,loco,mahalanobis,all}

Standard metrics already computed in train_router; this module adds LOCO stress
and a Mahalanobis-distance OOD comparison.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.common import MODELS_DIR, PROCESSED_DIR, RAW_DIR
from src.pipeline.category_router.constants import (
    ARTIFACT_LOCO, ROUTER_CLASSES, ROUTER_INPUT_FIELDS,
    EMBEDDINGS_NPY, TRAIN_PARQUET, TEST_PARQUET,
)
from src.pipeline.category_router.data import load_positive
from src.pipeline.category_router.mahalanobis_ood import (
    fit_mahalanobis, distance_to_nearest_centroid, loco_recall,
)
from src.pipeline.category_router.split import brand_disjoint_split

logger = logging.getLogger(__name__)


def _embed(embedder, df: pd.DataFrame) -> np.ndarray:
    texts = [
        " ".join(str(row[k] or "") for k in ROUTER_INPUT_FIELDS).strip() or " "
        for _, row in df.iterrows()
    ]
    return np.asarray(embedder.encode(texts, show_progress_bar=False))


def run_loco(
    *,
    processed_dir: str,
    off_parquet: str,
    embedder,
    n_per_class: int,
    seed: int,
    output_path: str,
    target_fpr_known: float,
) -> pd.DataFrame:
    """For each known class: train without it, treat held-out as OOD-positive."""
    pos = load_positive(processed_dir=processed_dir,
                        n_per_class=n_per_class, seed=seed)
    rows: list[dict] = []
    for leave_out in ROUTER_CLASSES:
        train_pos = pos[pos["category_label"] != leave_out].copy()
        loco_test = pos[pos["category_label"] == leave_out].copy()

        train_part, _ = brand_disjoint_split(train_pos, test_size=0.2, seed=seed)
        X_train = _embed(embedder, train_part)
        y_train_str = train_part["category_label"].to_numpy()

        present = [c for c in ROUTER_CLASSES if c != leave_out]
        le = LabelEncoder().fit(present)
        y_train = le.transform(y_train_str)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=seed,
            stratify=y_train,
        )
        clf = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, gamma=0.1,
            objective="multi:softprob", eval_metric="mlogloss",
            early_stopping_rounds=20, random_state=seed, n_jobs=1,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        proba_val = clf.predict_proba(X_val)
        max_p_val = proba_val.max(axis=1)
        threshold = float(np.quantile(max_p_val, target_fpr_known))

        X_loco = _embed(embedder, loco_test)
        proba_loco = clf.predict_proba(X_loco)
        max_p_loco = proba_loco.max(axis=1)
        ood_recall = float(np.mean(max_p_loco < threshold))
        mean_conf = float(np.mean(max_p_loco))
        rows.append({
            "leave_out_category": leave_out,
            "ood_recall": ood_recall,
            "mean_confidence_on_loco": mean_conf,
            "n_loco_examples": int(len(loco_test)),
            "loco_threshold": threshold,
        })
        logger.info("LOCO %s: ood_recall=%.3f mean_conf=%.3f n=%d",
                    leave_out, ood_recall, mean_conf, len(loco_test))

    df = pd.DataFrame(rows)
    df.to_parquet(output_path)
    return df


ARTIFACT_MAHALANOBIS_LOCO = os.path.join(MODELS_DIR, "category_router_mahalanobis_loco.parquet")


def run_mahalanobis_loco(
    *,
    embeddings_npy: str = EMBEDDINGS_NPY,
    train_parquet: str = TRAIN_PARQUET,
    test_parquet: str = TEST_PARQUET,
    target_fpr_known: float = 0.05,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Compute Mahalanobis-distance LOCO recall@FPR=target for all 7 known classes.

    Uses persisted embeddings and parquets (do not retrain XGBoost; this is a
    pure embedding-space analysis). For each leave_out class:
        - Fit per-class centroids + pooled covariance on the OTHER 6 known classes (train).
        - Calibrate threshold on known test rows of the same 6 (quantile @ target_fpr).
        - Compute recall on train rows of leave_out as held-out OOD positives.

    Returns DataFrame with columns:
        leave_out_category, n_loco, mahalanobis_recall, mahalanobis_threshold,
        mean_d_known, mean_d_loco, fpr_on_known
    """
    # Load persisted embeddings and labels.
    # Embeddings are saved as concat([train, test]) with positional alignment:
    #   embeddings[:n_train]  ↔  train_df (row-order)
    #   embeddings[n_train:]  ↔  test_df  (row-order)
    embeddings = np.load(embeddings_npy)
    train_df = pd.read_parquet(train_parquet).reset_index(drop=True)
    test_df = pd.read_parquet(test_parquet).reset_index(drop=True)

    n_train = len(train_df)
    X_train_all = embeddings[:n_train]   # shape (n_train, D)
    X_test_all = embeddings[n_train:]    # shape (n_test, D)

    rows: list[dict] = []
    for leave_out in ROUTER_CLASSES:
        # Train: all known classes except leave_out (positional selection)
        train_pos_mask = (
            (train_df["category_label"] != leave_out) &
            (train_df["category_label"] != "unknown")
        ).to_numpy()
        X_train = X_train_all[train_pos_mask]
        y_train = train_df.loc[train_pos_mask, "category_label"].to_numpy()

        # Test known: same 6 classes (for FPR calibration)
        test_known_pos_mask = (
            (test_df["category_label"] != leave_out) &
            (test_df["category_label"] != "unknown")
        ).to_numpy()
        # Test loco: the held-out class (ground-truth OOD positives)
        test_loco_pos_mask = (test_df["category_label"] == leave_out).to_numpy()

        X_test_known = X_test_all[test_known_pos_mask]
        X_test_loco = X_test_all[test_loco_pos_mask]

        fit = fit_mahalanobis(X_train, y_train)
        result = loco_recall(
            fit,
            X_known_test=X_test_known,
            X_loco_test=X_test_loco,
            target_fpr=target_fpr_known,
        )

        rows.append({
            "leave_out_category": leave_out,
            "n_loco": result["n_loco"],
            "mahalanobis_recall": result["ood_recall_loco"],
            "mahalanobis_threshold": result["threshold"],
            "mean_d_known": result["mean_d_known"],
            "mean_d_loco": result["mean_d_loco"],
            "fpr_on_known": result["fpr_on_known"],
        })
        logger.info(
            "Mahalanobis LOCO %s: recall=%.3f threshold=%.3f n=%d",
            leave_out, result["ood_recall_loco"], result["threshold"], result["n_loco"],
        )

    df = pd.DataFrame(rows)
    if output_path is not None:
        df.to_parquet(output_path)
    return df


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["standard", "loco", "mahalanobis", "all"], default="all")
    p.add_argument("--n-per-class", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-fpr", type=float, default=0.05)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.mode in {"standard", "all"}:
        meta_path = os.path.join(MODELS_DIR, "category_router_meta.json")
        if os.path.exists(meta_path):
            print(json.dumps(json.load(open(meta_path)), indent=2))
        else:
            print("standard metrics not found; run train first")

    if args.mode in {"loco", "all"}:
        from sentence_transformers import SentenceTransformer
        from src.common import EMBEDDING_MODEL
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        run_loco(
            processed_dir=PROCESSED_DIR,
            off_parquet=os.path.join(RAW_DIR, "en.openfoodfacts.org.products.parquet"),
            embedder=embedder,
            n_per_class=args.n_per_class,
            seed=args.seed,
            output_path=ARTIFACT_LOCO,
            target_fpr_known=args.target_fpr,
        )

    if args.mode in {"mahalanobis", "all"}:
        mahal_df = run_mahalanobis_loco(
            target_fpr_known=args.target_fpr,
            output_path=ARTIFACT_MAHALANOBIS_LOCO,
        )
        # Side-by-side comparison vs existing softmax LOCO
        print("\n=== Mahalanobis LOCO results ===")
        if os.path.exists(ARTIFACT_LOCO):
            softmax_df = pd.read_parquet(ARTIFACT_LOCO)[
                ["leave_out_category", "ood_recall"]
            ].rename(columns={"ood_recall": "softmax_recall"})
            cmp = mahal_df.merge(softmax_df, on="leave_out_category", how="left")
            cmp["delta"] = cmp["mahalanobis_recall"] - cmp["softmax_recall"]
            print(cmp[["leave_out_category", "n_loco",
                        "softmax_recall", "mahalanobis_recall", "delta"]].to_string(
                index=False, float_format=lambda x: f"{x:.3f}"))
            print(f"\nMacro mean Δ = {cmp['delta'].mean():.3f}")
        else:
            print(mahal_df.to_string(index=False))


if __name__ == "__main__":
    _cli()
