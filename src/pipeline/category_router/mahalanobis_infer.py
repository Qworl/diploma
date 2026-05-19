"""Runtime loader for the Mahalanobis OOD detector (Layer 0.5).

Loads the npz artefact produced by
``python -m src.pipeline.category_router.fit_mahalanobis`` and exposes a
single :func:`score` method that takes an SBERT embedding and returns the
Mahalanobis distance + OOD flag + nearest-centroid class name.

Why a separate detector: the softmax router (Layer 0) calibrates its OOD
threshold on `max(softmax)` — by construction confident on any input,
including adversarial gibberish. A distance-based score in embedding space
is not normalised across classes and stays large for inputs that do not
resemble any known centroid.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from src.common import MODELS_DIR

ARTIFACT_NPZ = os.path.join(MODELS_DIR, "category_router_mahalanobis.npz")


class MahalanobisScore(TypedDict):
    distance: float
    threshold: float
    is_ood: bool
    nearest_class: str


@dataclass
class MahalanobisOOD:
    classes: tuple[str, ...]
    centroids: np.ndarray   # (C, D)
    inv_cov: np.ndarray     # (D, D)
    threshold: float

    @classmethod
    def load(cls, path: str = ARTIFACT_NPZ) -> "MahalanobisOOD":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing Mahalanobis artefact: {path} — train with "
                "`python -m src.pipeline.category_router.fit_mahalanobis`"
            )
        m = np.load(path)
        return cls(
            classes=tuple(str(c) for c in m["classes"].tolist()),
            centroids=np.asarray(m["centroids"], dtype=np.float64),
            inv_cov=np.asarray(m["inv_cov"], dtype=np.float64),
            threshold=float(m["threshold"]),
        )

    def score(self, vec: np.ndarray) -> MahalanobisScore:
        """Return distance to nearest centroid and OOD verdict for one vector."""
        x = np.asarray(vec, dtype=np.float64).reshape(-1)
        diffs = x[None, :] - self.centroids                       # (C, D)
        d = np.sqrt(np.einsum("ij,jk,ik->i", diffs, self.inv_cov, diffs))
        i = int(d.argmin())
        return MahalanobisScore(
            distance=float(d[i]),
            threshold=self.threshold,
            is_ood=bool(d[i] > self.threshold),
            nearest_class=self.classes[i],
        )
