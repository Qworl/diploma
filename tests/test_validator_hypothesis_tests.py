"""Unit tests for the hypothesis-testing layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.validator_hypothesis_tests import (
    BONFERRONI_ALPHA,
    paired_bootstrap_auc_diff,
    pareto_dominance_vs_static,
)


def test_bonferroni_alpha_is_three_way():
    assert BONFERRONI_ALPHA == pytest.approx(0.05 / 3)


def test_paired_bootstrap_detects_separation():
    """Synthetic: score_a perfectly separates, score_b is random → reject null."""
    rng = np.random.default_rng(0)
    n = 200
    y = np.array([False] * 100 + [True] * 100)
    score_a = y.astype(float) + rng.normal(0, 0.05, size=n)  # near-perfect
    score_b = rng.normal(size=n)                              # noise
    result = paired_bootstrap_auc_diff(
        score_a, score_b, y, n_boot=200, seed=42, alternative="greater",
    )
    assert result["auc_a"] > 0.9
    assert result["auc_b"] < 0.7
    assert result["p_value"] < 0.05


def test_paired_bootstrap_returns_p_one_when_equal_scores():
    rng = np.random.default_rng(0)
    n = 100
    y = (rng.uniform(size=n) > 0.5)
    s = rng.normal(size=n)
    result = paired_bootstrap_auc_diff(s, s, y, n_boot=100, seed=1, alternative="greater")
    assert result["p_value"] >= 0.4  # paired diff identically zero


def test_pareto_dominance_simple():
    """One validator strictly Pareto-dominates static-policy."""
    df = pd.DataFrame({
        "attr": ["a"] * 5 + ["b"] * 5,
        "is_error": [True, False, False, False, False,  True, True, False, False, False],
        "validator_x": [0.9, 0.1, 0.1, 0.1, 0.1,  0.9, 0.9, 0.1, 0.1, 0.1],
    })
    # Static policy: route only attr 'b' (5 cells, 2 errors)
    # Validator x at top-3 scores: catches all 3 errors with 3 routed → strictly better
    out = pareto_dominance_vs_static(
        df=df,
        validator_cols=["validator_x"],
        static_attrs={"b"},
    )
    v = out["validator_x"]
    assert v["beats_static"] is True
    assert v["routing_rate_at_match"] < v["static_routing_rate"]
