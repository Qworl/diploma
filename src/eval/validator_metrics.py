"""Metrics for Trek A1 validator comparison."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc_for_validator(
    scores: pd.Series | np.ndarray,
    is_error: pd.Series | np.ndarray,
) -> float | None:
    """ROC AUC for a single validator. None if undefined."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(is_error, dtype=bool)
    mask = ~np.isnan(s)
    s, y = s[mask], y[mask]
    if len(np.unique(y)) < 2:
        return None
    if len(s) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return None


def precision_recall_at_k(
    scores: pd.Series | np.ndarray,
    is_error: pd.Series | np.ndarray,
    k: float,
) -> tuple[float | None, float | None, int]:
    """Top-k routing budget.

    Sort by ``scores`` descending; take the top ``ceil(k * n)`` rows; report
    precision (errors caught / routed) and recall (errors caught / total errors).
    Rows with NaN scores are placed *last* (treated as low priority).
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(is_error, dtype=bool)
    if len(s) == 0:
        return None, None, 0
    # NaN → -inf so they sink to the bottom
    s_clean = np.where(np.isnan(s), -np.inf, s)
    n = len(s_clean)
    k_n = max(1, int(np.ceil(k * n)))
    top_idx = np.argsort(-s_clean)[:k_n]
    routed_y = y[top_idx]
    n_errors_total = int(y.sum())
    n_caught = int(routed_y.sum())
    n_routed = int(k_n)
    precision = float(n_caught / n_routed) if n_routed else None
    recall = float(n_caught / n_errors_total) if n_errors_total else None
    return precision, recall, n_routed


def random_baseline(
    is_error: pd.Series | np.ndarray,
    k: float,
) -> tuple[float, float]:
    """Expectation for uniform-random routing at budget k."""
    y = np.asarray(is_error, dtype=bool)
    base_rate = float(y.mean()) if len(y) else 0.0
    return base_rate, float(k)


def static_policy_baseline(
    df: pd.DataFrame,
    attrs_to_route: Iterable[str],
) -> dict[str, float]:
    """§6.14.7 static-policy: always route the listed attrs, never the rest.

    Requires columns ``attr`` and ``is_error``.
    """
    attrs_set = set(attrs_to_route)
    n_total = len(df)
    if n_total == 0:
        return {"routing_rate": 0.0, "precision": None, "recall": None, "n_routed": 0}
    mask = df["attr"].isin(attrs_set).values
    n_routed = int(mask.sum())
    errors_total = int(df["is_error"].sum())
    errors_caught = int(df.loc[mask, "is_error"].sum())
    precision = float(errors_caught / n_routed) if n_routed else None
    recall = float(errors_caught / errors_total) if errors_total else None
    return {
        "routing_rate": n_routed / n_total,
        "precision": precision,
        "recall": recall,
        "n_routed": n_routed,
    }
