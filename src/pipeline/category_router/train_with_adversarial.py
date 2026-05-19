"""CLI: python -m src.pipeline.category_router.train_with_adversarial

Plan C: retrain the softmax router with an 8-th explicit class ``garbage``
fed by synthetic adversarial inputs (see :mod:`adversarial_sampler`).

Why an 8-th class instead of recalibrating the existing threshold:
adversarial inputs like ``ssad`` get embedded close enough to ``cosmetics``
that XGBoost outputs ``max_softmax ≈ 0.79``. The current OOD threshold is
0.51 — below 0.79, so the input is *not* flagged. Lowering the threshold
to catch 0.79 would push FPR on real known products well above the 5%
target. The principled fix is to expose the model to garbage during
training so it learns to *reserve* probability mass for "none of the
above": an explicit ``garbage`` class gives the model a target to map
unfamiliar embeddings onto.

Inference policy (the demo cascade enforces):

* predicted class == ``garbage``                    → OOD
* max softmax over real 7 classes < threshold       → OOD
* else                                              → use predicted class

Artefacts (do **not** overwrite v1):
    models/category_router_adv_xgb.pkl
    models/category_router_adv_le.pkl
    models/category_router_adv_threshold.json
    models/category_router_adv_meta.json

Reuses :file:`category_router_train.parquet` and
:file:`category_router_embeddings.npy` from v1 — only the adversarial
slice is freshly generated and embedded.
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.common import EMBEDDING_MODEL, MODELS_DIR, PROCESSED_DIR
from src.pipeline.category_router.adversarial_sampler import sample_adversarial
from src.pipeline.category_router.constants import (
    EMBEDDINGS_NPY,
    ROUTER_CLASSES,
    ROUTER_INPUT_FIELDS,
    TEST_PARQUET,
    TRAIN_PARQUET,
)

logger = logging.getLogger(__name__)

GARBAGE_LABEL = "garbage"

ARTIFACT_XGB = os.path.join(MODELS_DIR, "category_router_adv_xgb.pkl")
ARTIFACT_LE = os.path.join(MODELS_DIR, "category_router_adv_le.pkl")
ARTIFACT_THRESHOLD = os.path.join(MODELS_DIR, "category_router_adv_threshold.json")
ARTIFACT_META = os.path.join(MODELS_DIR, "category_router_adv_meta.json")


def _build_text(row: pd.Series) -> str:
    return " ".join(str(row[k] or "") for k in ROUTER_INPUT_FIELDS).strip() or " "


def _embed_texts(embedder, df: pd.DataFrame) -> np.ndarray:
    texts = [_build_text(r) for _, r in df.iterrows()]
    return np.asarray(embedder.encode(texts, show_progress_bar=False))


def _calibrate_threshold_on_real(
    proba_known: np.ndarray, real_class_indices: list[int], target_fpr: float
) -> tuple[float, float]:
    """Pick threshold on max softmax across *real* 7 classes only.

    The 8-th column (garbage) is excluded from max_softmax so a real product
    that gets some mass leaked into ``garbage`` is not falsely OOD'd just
    because its top-2 real-class confidence dropped a bit.
    """
    max_p = proba_known[:, real_class_indices].max(axis=1)
    t = float(np.quantile(max_p, target_fpr))
    fpr = float(np.mean(max_p < t))
    return t, fpr


def train(
    n_adversarial: int = 2000,
    seed: int = 42,
    target_fpr_known: float = 0.05,
) -> dict[str, Any]:
    if not (
        os.path.exists(TRAIN_PARQUET)
        and os.path.exists(TEST_PARQUET)
        and os.path.exists(EMBEDDINGS_NPY)
    ):
        raise FileNotFoundError(
            "v1 router artefacts missing — run "
            "`python -m src.pipeline.category_router.train` first"
        )

    tr = pd.read_parquet(TRAIN_PARQUET)
    te = pd.read_parquet(TEST_PARQUET)
    emb_all = np.load(EMBEDDINGS_NPY)
    if len(emb_all) != len(tr) + len(te):
        raise RuntimeError(
            f"embeddings ({emb_all.shape}) != train+test={len(tr) + len(te)}"
        )
    emb_tr = emb_all[: len(tr)]
    emb_te = emb_all[len(tr) :]

    # 1) Generate adversarial corpus and embed it freshly through SBERT
    logger.info("generating %d adversarial samples", n_adversarial)
    adv = sample_adversarial(n_adversarial, seed=seed)
    adv_train, adv_test = train_test_split(
        adv, test_size=0.2, random_state=seed
    )
    adv_train = adv_train.reset_index(drop=True)
    adv_test = adv_test.reset_index(drop=True)

    from sentence_transformers import SentenceTransformer
    logger.info("loading SBERT %s", EMBEDDING_MODEL)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("embedding adversarial train (%d)", len(adv_train))
    adv_emb_tr = _embed_texts(embedder, adv_train)
    logger.info("embedding adversarial test (%d)", len(adv_test))
    adv_emb_te = _embed_texts(embedder, adv_test)

    # 2) Build extended train / test tables for v4
    tr_real = tr.copy()
    te_real = te.copy()
    tr_real["is_ood"] = (tr_real["category_label"] == "unknown").astype(bool)
    te_real["is_ood"] = (te_real["category_label"] == "unknown").astype(bool)
    adv_train["is_ood"] = True
    adv_test["is_ood"] = True

    full_train = pd.concat([tr_real, adv_train], ignore_index=True)
    full_test = pd.concat([te_real, adv_test], ignore_index=True)
    X_train_full = np.vstack([emb_tr, adv_emb_tr])
    X_test_full = np.vstack([emb_te, adv_emb_te])
    assert len(X_train_full) == len(full_train), (len(X_train_full), len(full_train))
    assert len(X_test_full) == len(full_test), (len(X_test_full), len(full_test))

    # 3) Train an 8-class XGBoost. unknown rows (real OFF off-class) are
    # excluded so that 8 = {7 real} ∪ {garbage}; "unknown" stays an OOD
    # evaluation signal, not a class.
    classes_8 = tuple(list(ROUTER_CLASSES) + [GARBAGE_LABEL])
    le = LabelEncoder().fit(list(classes_8))

    train_mask = full_train["category_label"].isin(classes_8).to_numpy()
    X_train = X_train_full[train_mask]
    y_train = le.transform(full_train.loc[train_mask, "category_label"].to_numpy())

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
    )

    logger.info(
        "training 8-class XGBoost: train=%d (val=%d), classes=%s",
        len(X_tr), len(X_val), classes_8,
    )
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, gamma=0.1,
        objective="multi:softprob", eval_metric="mlogloss",
        early_stopping_rounds=20, random_state=seed, n_jobs=1,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # 4) Calibrate threshold on max softmax over the 7 real classes,
    # evaluated on a held-out validation slice of *real* products only.
    real_class_idx = [int(le.transform([c])[0]) for c in ROUTER_CLASSES]
    proba_val = clf.predict_proba(X_val)
    real_only_val_mask = np.isin(y_val, real_class_idx)
    threshold, fpr_on_known = _calibrate_threshold_on_real(
        proba_val[real_only_val_mask], real_class_idx, target_fpr_known
    )

    # 5) Test-set evaluation: real-known accuracy + OOD detection vs both
    # "unknown" (real off-class) and "garbage" (adversarial).
    known_te_mask = full_test["category_label"].isin(ROUTER_CLASSES).to_numpy()
    unknown_te_mask = (full_test["category_label"] == "unknown").to_numpy()
    garbage_te_mask = (full_test["category_label"] == GARBAGE_LABEL).to_numpy()

    proba_te = clf.predict_proba(X_test_full)
    max_real = proba_te[:, real_class_idx].max(axis=1)
    pred_idx = proba_te.argmax(axis=1)
    pred_label = le.inverse_transform(pred_idx)
    pred_garbage_mask = pred_label == GARBAGE_LABEL

    is_ood_flag = (max_real < threshold) | pred_garbage_mask

    real_known_acc = float(
        accuracy_score(
            le.transform(full_test.loc[known_te_mask, "category_label"]),
            pred_idx[known_te_mask],
        )
    )
    real_known_f1 = float(
        f1_score(
            le.transform(full_test.loc[known_te_mask, "category_label"]),
            pred_idx[known_te_mask],
            average="macro",
        )
    )
    fpr_on_known_test = float(is_ood_flag[known_te_mask].mean())
    recall_on_unknown_test = float(is_ood_flag[unknown_te_mask].mean())
    recall_on_adversarial_test = float(is_ood_flag[garbage_te_mask].mean())
    ood_auroc = float(
        roc_auc_score(
            (~known_te_mask).astype(int),
            1.0 - max_real,
        )
    )

    logger.info("threshold=%.4f  FPR(real known)=%.3f", threshold, fpr_on_known_test)
    logger.info("real-known test: acc=%.3f  f1_macro=%.3f", real_known_acc, real_known_f1)
    logger.info("OOD recall: unknown=%.3f  adversarial=%.3f",
                recall_on_unknown_test, recall_on_adversarial_test)
    logger.info("OOD AUROC (1 − max_softmax over real classes): %.3f", ood_auroc)

    # 6) Save artefacts
    with open(ARTIFACT_XGB, "wb") as f:
        pickle.dump(clf, f)
    with open(ARTIFACT_LE, "wb") as f:
        pickle.dump(le, f)
    with open(ARTIFACT_THRESHOLD, "w") as f:
        json.dump({
            "threshold": threshold,
            "calibrated_on": "real_class_validation_split",
            "target_fpr": target_fpr_known,
            "garbage_label": GARBAGE_LABEL,
            "real_classes": list(ROUTER_CLASSES),
        }, f, indent=2)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_real_known": int(
            full_train["category_label"].isin(ROUTER_CLASSES).sum()
        ),
        "n_train_garbage": int(len(adv_train)),
        "n_test_real_known": int(known_te_mask.sum()),
        "n_test_unknown": int(unknown_te_mask.sum()),
        "n_test_garbage": int(garbage_te_mask.sum()),
        "seed": seed,
        "threshold": threshold,
        "fpr_on_known_test": fpr_on_known_test,
        "recall_on_unknown_test": recall_on_unknown_test,
        "recall_on_adversarial_test": recall_on_adversarial_test,
        "real_known_test_accuracy": real_known_acc,
        "real_known_test_f1_macro": real_known_f1,
        "ood_auroc": ood_auroc,
        "classes": list(classes_8),
        "garbage_label": GARBAGE_LABEL,
    }
    with open(ARTIFACT_META, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-adversarial", type=int, default=2000)
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train(
        n_adversarial=args.n_adversarial,
        target_fpr_known=args.target_fpr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
