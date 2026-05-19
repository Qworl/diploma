"""CLI: python -m src.pipeline.category_router.fit_mahalanobis

Fits a Mahalanobis-distance semantic OOD detector on top of the existing
router-training embeddings. Produces an artifact that the demo cascade loads
as Layer 0.5 (semantic OOD) — runs between the cheap input-validator
(Layer −1) and the softmax-threshold router (Layer 0).

Why a second OOD signal: the router's max-softmax threshold is calibrated on
real but off-distribution products (supplements, household). Adversarial
gibberish such as "ssad" embeds into a region close to some class centroid
and gets confident predictions from XGBoost — softmax cannot say "none of
the above" by construction (probabilities sum to 1). A distance-based
detector (Lee et al., NeurIPS 2018) does not share that closed-set
limitation: large distance to every centroid ⇒ "unfamiliar".

Outputs to ``models/category_router_mahalanobis.npz`` (plus a tiny
``..._meta.json`` with metrics and the calibration recipe).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.common import MODELS_DIR
from src.pipeline.category_router.constants import (
    EMBEDDINGS_NPY,
    TEST_PARQUET,
    TRAIN_PARQUET,
)
from src.pipeline.category_router.mahalanobis_ood import (
    distance_to_nearest_centroid,
    fit_mahalanobis,
)

logger = logging.getLogger(__name__)

ARTIFACT_NPZ = os.path.join(MODELS_DIR, "category_router_mahalanobis.npz")
ARTIFACT_META_JSON = os.path.join(MODELS_DIR, "category_router_mahalanobis_meta.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.05,
        help="желаемый FPR на in-distribution тесте (по умолчанию 0.05)",
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=1e-3,
        help="диагональная регуляризация ковариации",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not (os.path.exists(EMBEDDINGS_NPY) and os.path.exists(TRAIN_PARQUET)):
        raise FileNotFoundError(
            "router train artefacts not found — сначала "
            "`python -m src.pipeline.category_router.train`"
        )

    emb = np.load(EMBEDDINGS_NPY)
    tr = pd.read_parquet(TRAIN_PARQUET)
    te = pd.read_parquet(TEST_PARQUET)
    if len(emb) != len(tr) + len(te):
        raise RuntimeError(
            f"embeddings shape {emb.shape} != train+test={len(tr) + len(te)}"
        )
    emb_tr = emb[: len(tr)]
    emb_te = emb[len(tr) :]

    # 1) Подгонка по эмбеддингам только известных 7 классов (is_ood=False)
    in_tr_mask = ~tr["is_ood"].to_numpy()
    X_known_tr = emb_tr[in_tr_mask]
    y_known_tr = tr.loc[in_tr_mask, "category_label"].to_numpy()
    logger.info("fit on %d in-distribution train samples", len(X_known_tr))
    fit = fit_mahalanobis(X_known_tr, y_known_tr, reg=args.reg)

    # 2) Калибровка порога на known-test: квантиль (1−target_fpr)
    in_te_mask = ~te["is_ood"].to_numpy()
    X_known_te = emb_te[in_te_mask]
    X_unknown_te = emb_te[~in_te_mask]
    d_known = distance_to_nearest_centroid(fit, X_known_te)
    d_unknown = distance_to_nearest_centroid(fit, X_unknown_te)
    threshold = float(np.quantile(d_known, 1.0 - args.target_fpr))

    fpr_on_known = float((d_known > threshold).mean())
    recall_on_unknown = float((d_unknown > threshold).mean())

    logger.info("threshold = %.4f (target FPR=%.2f)", threshold, args.target_fpr)
    logger.info(
        "in-distribution known test: n=%d, FPR=%.3f, mean d=%.3f",
        len(d_known), fpr_on_known, float(d_known.mean()),
    )
    logger.info(
        "out-of-distribution (router 'unknown'): n=%d, recall=%.3f, mean d=%.3f",
        len(d_unknown), recall_on_unknown, float(d_unknown.mean()),
    )

    # 3) Сохранить артефакт в одном npz: ordered classes, centroids matrix, inv_cov, threshold
    classes_arr = np.array(list(fit.classes))
    centroids_mat = np.stack([fit.centroids[c] for c in fit.classes], axis=0)
    np.savez(
        ARTIFACT_NPZ,
        classes=classes_arr,
        centroids=centroids_mat,
        inv_cov=fit.inv_cov,
        threshold=np.array(threshold),
    )
    logger.info("saved artefact: %s", ARTIFACT_NPZ)

    meta = {
        "artifact": os.path.basename(ARTIFACT_NPZ),
        "classes": list(fit.classes),
        "n_train_in_distribution": int(in_tr_mask.sum()),
        "n_test_in_distribution": int(in_te_mask.sum()),
        "n_test_out_of_distribution": int((~in_te_mask).sum()),
        "target_fpr": args.target_fpr,
        "reg": args.reg,
        "threshold": threshold,
        "fpr_on_known_test": fpr_on_known,
        "recall_on_unknown_test": recall_on_unknown,
        "mean_distance_known": float(d_known.mean()),
        "mean_distance_unknown": float(d_unknown.mean()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "Mahalanobis (Lee et al., NeurIPS 2018), pooled within-class covariance",
    }
    with open(ARTIFACT_META_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("saved meta: %s", ARTIFACT_META_JSON)


if __name__ == "__main__":
    main()
