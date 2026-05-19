"""Per-attribute Mahalanobis-distance validator for the pasta cascade.

For each attribute (grain_type, pasta_shape, ...) fits a MahalanobisFit
using the ML-layer training embeddings restricted to rows where the
silver label is known. Score on a test embedding = distance to nearest
class centroid for that attr (large => unfamiliar).
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from src.pipeline.category_router.mahalanobis_ood import (
    MahalanobisFit,
    distance_to_nearest_centroid,
    fit_mahalanobis,
)


def _clean_labels(y: np.ndarray) -> np.ndarray:
    """Convert None / NaN / 'nan' to a sentinel for filtering."""
    out: list = []
    for v in y:
        if v is None:
            out.append(None)
            continue
        if isinstance(v, float) and np.isnan(v):
            out.append(None)
            continue
        s = str(v).strip()
        if s.lower() in {"", "nan", "none", "null"}:
            out.append(None)
            continue
        out.append(s)
    return np.array(out, dtype=object)


def fit_per_attr_mahalanobis(
    X: np.ndarray,
    labels: Mapping[str, np.ndarray],
    attrs: Iterable[str],
    reg: float = 1e-3,
) -> dict[str, MahalanobisFit]:
    """Fit one Mahalanobis model per attribute.

    Parameters
    ----------
    X      : (N, D) training embeddings (e.g. pasta_stratified_embeddings.npy).
    labels : mapping attr -> (N,) silver labels (may contain None / NaN).
    attrs  : attributes to fit. Attributes with <2 classes or fewer than
             3 examples per class are skipped (cannot estimate covariance).
    reg    : diagonal regularisation passed to fit_mahalanobis.
    """
    fits: dict[str, MahalanobisFit] = {}
    for attr in attrs:
        if attr not in labels:
            continue
        y_raw = _clean_labels(np.asarray(labels[attr]))
        mask = np.array([v is not None for v in y_raw])
        if mask.sum() < 6:
            continue
        y = y_raw[mask]
        Xa = X[mask]
        classes = set(y.tolist())
        if len(classes) < 2:
            continue
        # Need at least 2 examples per class for within-class scatter
        if any((y == c).sum() < 2 for c in classes):
            # Drop too-small classes
            keep = np.array([(y == v).sum() >= 2 for v in y])
            y = y[keep]
            Xa = Xa[keep]
            classes = set(y.tolist())
            if len(classes) < 2:
                continue
        fits[attr] = fit_mahalanobis(Xa, y, reg=reg)
    return fits


def score_per_attr_mahalanobis(
    X: np.ndarray,
    fits: Mapping[str, MahalanobisFit],
) -> dict[str, np.ndarray]:
    """Score X (N, D) against each fitted attribute.

    Returns mapping attr -> (N,) array of nearest-centroid distances.
    """
    return {attr: distance_to_nearest_centroid(fit, X) for attr, fit in fits.items()}
