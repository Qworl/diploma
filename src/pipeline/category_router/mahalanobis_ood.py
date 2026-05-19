"""Mahalanobis-distance OOD detector for the category router (Layer 0).

Implements the distance-based OOD baseline from:
    Lee et al. "A Simple Unified Framework for Detecting Out-of-Distribution
    Samples and Adversarial Attacks." NeurIPS 2018.

Algorithm
---------
1. Fit: compute per-class centroids (μ_c) and a *pooled* within-class
   covariance Σ = Σ_c (X_c − μ_c)ᵀ(X_c − μ_c) / (N − C).
   Regularize: Σ ← Σ + reg·I, then invert once.

2. Score: for a new sample x, its OOD score is
       min_c  sqrt( (x − μ_c)ᵀ Σ⁻¹ (x − μ_c) ).
   Large score ⟹ far from every known class ⟹ likely OOD.

3. Threshold calibration (loco_recall): given known-class test examples,
   pick the quantile threshold such that FPR ≈ target_fpr.  Then evaluate
   recall on a held-out / leave-one-class-out set.

This module is intentionally self-contained (numpy only) so it can be
imported in the thesis notebook without loading the full pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MahalanobisFit:
    """Fitted parameters for Mahalanobis-distance OOD detection."""

    classes: tuple[str, ...]            # ordered class names
    centroids: dict[str, np.ndarray]    # per-class mean vector (D,)
    inv_cov: np.ndarray                 # pooled inverse covariance (D, D)


def fit_mahalanobis(
    X: np.ndarray,
    y: np.ndarray,
    reg: float = 1e-3,
) -> MahalanobisFit:
    """Fit per-class centroids + pooled within-class covariance.

    Parameters
    ----------
    X   : (N, D) float array of embeddings.
    y   : (N,) array of string class labels.
    reg : diagonal regularisation added to covariance before inversion.

    Returns
    -------
    MahalanobisFit with centroids and a single shared inv_cov.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = tuple(sorted(set(y.tolist())))
    N, D = X.shape
    C = len(classes)

    centroids: dict[str, np.ndarray] = {}
    for c in classes:
        mask = y == c
        centroids[c] = X[mask].mean(axis=0)

    # Pooled within-class scatter (unbiased: divide by N − C)
    scatter = np.zeros((D, D), dtype=np.float64)
    for c in classes:
        mask = y == c
        diff = X[mask] - centroids[c]          # (n_c, D)
        scatter += diff.T @ diff

    denom = max(N - C, 1)
    cov = scatter / denom + reg * np.eye(D, dtype=np.float64)
    inv_cov = np.linalg.pinv(cov)

    return MahalanobisFit(classes=classes, centroids=centroids, inv_cov=inv_cov)


def distance_to_nearest_centroid(
    fit: MahalanobisFit,
    X: np.ndarray,
) -> np.ndarray:
    """Return Mahalanobis distance to the nearest class centroid.

    For each row x in X:
        score(x) = min_c  sqrt( (x − μ_c) Σ⁻¹ (x − μ_c)ᵀ )

    Parameters
    ----------
    fit : fitted MahalanobisFit.
    X   : (N, D) float array.

    Returns
    -------
    (N,) array of non-negative distances.
    """
    X = np.asarray(X, dtype=np.float64)
    # Stack per-class distances: shape (C, N)
    per_class = np.stack(
        [
            np.sqrt(
                np.einsum(
                    "ij,jk,ik->i",
                    X - mu,
                    fit.inv_cov,
                    X - mu,
                )
            )
            for mu in (fit.centroids[c] for c in fit.classes)
        ],
        axis=0,
    )   # (C, N)
    return per_class.min(axis=0)   # (N,)


def loco_recall(
    fit: MahalanobisFit,
    X_known_test: np.ndarray,
    X_loco_test: np.ndarray,
    target_fpr: float = 0.05,
) -> dict[str, float]:
    """Calibrate threshold on known examples, then measure OOD recall on LOCO set.

    Threshold is the (1 − target_fpr) quantile of distances on X_known_test.
    A sample is flagged as OOD when its distance *exceeds* the threshold.

    Parameters
    ----------
    fit           : fitted MahalanobisFit.
    X_known_test  : (Nk, D) — in-distribution hold-out for FPR calibration.
    X_loco_test   : (Nl, D) — held-out class examples (ground-truth OOD).
    target_fpr    : desired false-positive rate on X_known_test.

    Returns
    -------
    dict with keys:
        threshold, fpr_on_known, ood_recall_loco,
        mean_d_known, mean_d_loco, n_known, n_loco.
    """
    X_known_test = np.asarray(X_known_test, dtype=np.float64)
    X_loco_test = np.asarray(X_loco_test, dtype=np.float64)

    d_known = distance_to_nearest_centroid(fit, X_known_test)
    d_loco = distance_to_nearest_centroid(fit, X_loco_test)

    # Threshold: the quantile at which (1 − target_fpr) of known samples
    # fall *below* the threshold → FPR ≈ target_fpr on known.
    threshold = float(np.quantile(d_known, 1.0 - target_fpr))

    fpr_on_known = float((d_known > threshold).mean())
    ood_recall_loco = float((d_loco > threshold).mean())

    return {
        "threshold": threshold,
        "fpr_on_known": fpr_on_known,
        "ood_recall_loco": ood_recall_loco,
        "mean_d_known": float(d_known.mean()),
        "mean_d_loco": float(d_loco.mean()),
        "n_known": len(d_known),
        "n_loco": len(d_loco),
    }
