"""Coverage gain, recovery accuracy and CIs for Trek A2."""
from __future__ import annotations

import math
from typing import Tuple

import pandas as pd


def _norm(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"", "nan", "none", "null"}:
            return None
        if s in {"true", "yes"}:
            return True
        if s in {"false", "no"}:
            return False
        return s
    if isinstance(v, bool):
        return v
    return str(v).strip().lower()


def _values_equal(a, b) -> bool:
    na, nb = _norm(a), _norm(b)
    if na is None or nb is None:
        return False
    return na == nb


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval. Returns (lo, hi) in [0, 1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    # 1-alpha/2 quantile of N(0,1); for alpha=0.05 z=1.959964.
    # alpha=0.017 (Bonferroni) → z=2.387.
    from scipy.stats import norm  # local import — scipy is already an indirect dep
    z = norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def aggregate_metrics(
    mask_log: pd.DataFrame,
    cascade_log: pd.DataFrame,
    *,
    n_products: int,
    alpha: float = 0.05,
) -> dict:
    """Aggregate (mask_log, cascade_log) → metric dict.

    Both DataFrames must have columns (code, attr). mask_log adds
    (masked, original_value); cascade_log adds (cascade_pred, cascade_layer,
    llm_called, cost_tokens).
    """
    joined = mask_log.merge(cascade_log, on=["code", "attr"], how="left")

    n_cells = len(joined)
    # Originally filled (= what partner supplies before cascade)
    baseline_filled = joined["original_value"].apply(lambda v: _norm(v) is not None)
    n_baseline = int(baseline_filled.sum())

    # Cascade output filled (regardless of layer)
    cascade_filled = joined["cascade_pred"].apply(lambda v: _norm(v) is not None)
    # Post-cascade coverage = originally filled OR (masked AND cascade filled).
    # We don't double-count cells already filled by the partner.
    masked_col = joined["masked"].astype(bool)
    post_filled = baseline_filled | (masked_col & cascade_filled)
    n_post = int(post_filled.sum())

    # Coverage gain: cells that were masked in the simulation AND cascade filled them.
    # This measures how many artificially-emptied slots the cascade recovered.
    coverage_gain_cells = int((masked_col & cascade_filled).sum())
    coverage_gain_frac = coverage_gain_cells / n_cells if n_cells else float("nan")
    gain_lo, gain_hi = wilson_ci(coverage_gain_cells, n_cells, alpha=alpha)

    # Recovery accuracy: among cells that were masked AND cascade filled, how
    # many match the held-out original value? (We use `original_value` as
    # ground-truth proxy when the runner is silver-based, or audited gold
    # when the runner pre-injected audited values into `original_value`.)
    recov_universe = joined[joined["masked"] & cascade_filled].copy()
    if len(recov_universe) > 0:
        recov_universe["correct"] = recov_universe.apply(
            lambda r: _values_equal(r["cascade_pred"], r["original_value"]),
            axis=1,
        )
        n_rec_correct = int(recov_universe["correct"].sum())
        n_rec = int(len(recov_universe))
        recovery_accuracy = n_rec_correct / n_rec
        rec_lo, rec_hi = wilson_ci(n_rec_correct, n_rec, alpha=alpha)
    else:
        recovery_accuracy = float("nan")
        n_rec_correct = 0
        n_rec = 0
        rec_lo, rec_hi = float("nan"), float("nan")

    # LLM cost: count one call per (code) with any llm_called=True row.
    per_product_llm = (
        cascade_log.groupby("code")["llm_called"].any().astype(int)
    )
    llm_calls = int(per_product_llm.sum())
    llm_per_1000 = llm_calls / n_products * 1000 if n_products else float("nan")

    return {
        "n_cells": n_cells,
        "n_products": n_products,
        "baseline_coverage": n_baseline / n_cells if n_cells else float("nan"),
        "post_cascade_coverage": n_post / n_cells if n_cells else float("nan"),
        "coverage_gain_pp": coverage_gain_frac * 100,
        "coverage_gain_ci_pp": (gain_lo * 100, gain_hi * 100),
        "coverage_gain_cells": coverage_gain_cells,
        "recovery_accuracy": recovery_accuracy,
        "recovery_accuracy_ci": (rec_lo, rec_hi),
        "recovery_n": n_rec,
        "recovery_n_correct": n_rec_correct,
        "llm_calls": llm_calls,
        "llm_calls_per_1000_products": llm_per_1000,
        "per_layer": cascade_log["cascade_layer"].value_counts().to_dict(),
    }
