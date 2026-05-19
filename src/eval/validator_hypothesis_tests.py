"""Pre-registered hypothesis tests for Trek A1."""
from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.eval.validator_metrics import (
    auc_for_validator,
    precision_recall_at_k,
    static_policy_baseline,
)

# Three pre-registered hypotheses (H1, H2, H3); Bonferroni-corrected.
BONFERRONI_ALPHA = 0.05 / 3


def _auc_safe(y: np.ndarray, s: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return None


def paired_bootstrap_auc_diff(
    score_a: np.ndarray,
    score_b: np.ndarray,
    is_error: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alternative: Literal["greater", "two-sided"] = "greater",
) -> dict:
    """Paired bootstrap test for AUC(a) - AUC(b).

    Resamples cells with replacement; on each resample computes AUC(a), AUC(b)
    and their difference. p-value = fraction of bootstrap diffs <= 0 (for
    "greater" alternative).
    """
    s_a = np.asarray(score_a, dtype=float)
    s_b = np.asarray(score_b, dtype=float)
    y = np.asarray(is_error, dtype=bool)
    mask = ~np.isnan(s_a) & ~np.isnan(s_b)
    s_a, s_b, y = s_a[mask], s_b[mask], y[mask]
    n = len(y)
    if n < 2 or len(np.unique(y)) < 2:
        return {"auc_a": None, "auc_b": None, "diff": None,
                "p_value": float("nan"), "n": n}

    auc_a = _auc_safe(y, s_a)
    auc_b = _auc_safe(y, s_b)
    diff = (auc_a - auc_b) if (auc_a is not None and auc_b is not None) else None

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    diffs.fill(np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            diffs[i] = 0.0
            continue
        try:
            a = roc_auc_score(y_b, s_a[idx])
            b = roc_auc_score(y_b, s_b[idx])
            diffs[i] = a - b
        except ValueError:
            diffs[i] = 0.0
    if alternative == "greater":
        p = float((diffs <= 0).mean())
    else:
        p = float((np.abs(diffs) >= abs(diff or 0.0)).mean())
    return {
        "auc_a": auc_a, "auc_b": auc_b, "diff": diff,
        "p_value": p, "n": n, "n_boot": n_boot,
    }


def pareto_dominance_vs_static(
    df: pd.DataFrame,
    validator_cols: Iterable[str],
    static_attrs: set[str],
) -> dict[str, dict]:
    """For each validator, check whether it Pareto-beats the static policy.

    A validator "beats" the static policy if it can match the static-policy
    recall at a strictly lower routing rate.
    """
    base = static_policy_baseline(df, attrs_to_route=static_attrs)
    static_recall = base["recall"]
    static_rate = base["routing_rate"]

    out: dict[str, dict] = {}
    for col in validator_cols:
        scores = df[col].values
        y = df["is_error"].values
        if static_recall is None or np.isnan(scores.astype(float)).all():
            out[col] = {
                "beats_static": False,
                "static_routing_rate": static_rate,
                "static_recall": static_recall,
                "routing_rate_at_match": None,
            }
            continue
        # Sweep routing budgets from 1% to 100% in 1% steps;
        # find the smallest budget where recall >= static_recall.
        match_rate: float | None = None
        for k_pct in range(1, 101):
            k = k_pct / 100.0
            _, recall, _ = precision_recall_at_k(scores, y, k=k)
            if recall is not None and recall >= static_recall:
                match_rate = k
                break
        beats = (match_rate is not None) and (match_rate < static_rate)
        out[col] = {
            "beats_static": bool(beats),
            "static_routing_rate": static_rate,
            "static_recall": static_recall,
            "routing_rate_at_match": match_rate,
        }
    return out
