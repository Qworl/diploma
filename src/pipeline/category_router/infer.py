"""Runtime category-router inference."""
from __future__ import annotations

import json
import logging
import os
import pickle
from typing import TypedDict

import numpy as np

from src.pipeline.category_router.constants import ROUTER_INPUT_FIELDS

logger = logging.getLogger(__name__)


class RouterPrediction(TypedDict):
    predicted: str
    confidence: float
    alternatives: list[tuple[str, float]]
    is_ood: bool


def _build_text(product: dict) -> str:
    return " ".join(
        str(product.get(k, "") or "") for k in ROUTER_INPUT_FIELDS
    ).strip() or " "


class CategoryRouter:
    def __init__(
        self,
        clf,
        label_encoder,
        threshold: float,
        embedder,
        garbage_label: str | None = None,
    ):
        self._clf = clf
        self._le = label_encoder
        self._threshold = float(threshold)
        self._embedder = embedder
        self._garbage_label = garbage_label
        all_labels = [str(x) for x in label_encoder.classes_]
        self._real_class_indices = [
            i for i, lbl in enumerate(all_labels)
            if lbl != garbage_label
        ]

    @classmethod
    def load(cls, models_dir: str, embedder, variant: str = "v1") -> "CategoryRouter":
        """Load a router artefact.

        ``variant`` selects which trained model to use:

        * ``"v1"`` (default) — legacy 7-class softmax. Calibrated max-softmax
          threshold catches "real but off-class" OOD reasonably; misses
          adversarial gibberish, which embeds close to some class centroid.
        * ``"adv"`` — 8-class with explicit ``garbage`` class trained on
          synthetic adversarial inputs (см. train_with_adversarial). Catches
          adversarial at 100% recall but introduces FPR on short bare-name
          real products (e.g. ``Lindt``, ``Roquefort AOP`` без контекста).
          Deployed offline для главы 3; в demo не подключён по умолчанию,
          потому что Layer 0.5 (Mahalanobis) уже закрывает adversarial без
          этой регрессии.

        File-naming convention: v1 → ``category_router_*.pkl``, adv →
        ``category_router_adv_*.pkl``.
        """
        suffix = "_adv" if variant == "adv" else ""
        candidates = [(
            os.path.join(models_dir, f"category_router{suffix}_xgb.pkl"),
            os.path.join(models_dir, f"category_router{suffix}_le.pkl"),
            os.path.join(models_dir, f"category_router{suffix}_threshold.json"),
        )]
        for xgb_path, le_path, thr_path in candidates:
            if all(os.path.exists(p) for p in (xgb_path, le_path, thr_path)):
                with open(xgb_path, "rb") as f:
                    clf = pickle.load(f)
                with open(le_path, "rb") as f:
                    le = pickle.load(f)
                with open(thr_path, "r") as f:
                    thr_meta = json.load(f)
                logger.info(
                    "router artefact: %s (garbage_label=%s)",
                    os.path.basename(xgb_path), thr_meta.get("garbage_label"),
                )
                return cls(
                    clf=clf,
                    label_encoder=le,
                    threshold=thr_meta["threshold"],
                    embedder=embedder,
                    garbage_label=thr_meta.get("garbage_label"),
                )
        raise FileNotFoundError(
            "no router artefact found in %s — train with "
            "`python -m src.pipeline.category_router.train` (v1) or "
            "`python -m src.pipeline.category_router.train_with_adversarial` (v4)"
            % models_dir
        )

    def predict(self, product: dict) -> RouterPrediction:
        text = _build_text(product)
        vec = np.asarray(
            self._embedder.encode([text], show_progress_bar=False)
        )
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        proba = self._clf.predict_proba(vec)[0]
        # OOD = max softmax over the *real* classes only (excludes garbage
        # if it's a class). The classifier's argmax can still point at
        # garbage — that's caught explicitly below.
        real_proba = proba[self._real_class_indices]
        real_order_local = np.argsort(real_proba)[::-1]
        top3 = [
            (
                str(self._le.inverse_transform(
                    [self._real_class_indices[int(j)]]
                )[0]),
                float(real_proba[int(j)]),
            )
            for j in real_order_local[:3]
        ]
        max_real_p = float(real_proba[real_order_local[0]])
        argmax_full = int(np.argmax(proba))
        argmax_label = str(self._le.inverse_transform([argmax_full])[0])
        is_garbage_pred = (
            self._garbage_label is not None
            and argmax_label == self._garbage_label
        )
        is_ood = bool((max_real_p < self._threshold) or is_garbage_pred)
        predicted = "unknown" if is_ood else top3[0][0]
        return RouterPrediction(
            predicted=predicted,
            confidence=max_real_p,
            alternatives=top3,
            is_ood=is_ood,
        )
