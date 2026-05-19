import numpy as np
import pandas as pd
import pytest

from src.eval.router_stats import (
    mcnemar_test,
    paired_bootstrap_delta,
    routing_to_pred,
)


def test_mcnemar_b_eq_c_no_difference():
    a_correct = np.array([1, 1, 0, 0])
    b_correct = np.array([0, 0, 1, 1])
    p = mcnemar_test(a_correct, b_correct)
    assert 0.0 < p <= 1.0


def test_mcnemar_a_dominates():
    a_correct = np.array([1] * 50 + [1] * 50)
    b_correct = np.array([0] * 50 + [1] * 50)
    p = mcnemar_test(a_correct, b_correct)
    assert p < 0.001


def test_paired_bootstrap_returns_ci():
    a_correct = np.random.binomial(1, 0.9, size=200)
    b_correct = np.random.binomial(1, 0.8, size=200)
    delta, ci_lo, ci_hi = paired_bootstrap_delta(a_correct, b_correct,
                                                   n_iter=200, seed=42)
    assert ci_lo <= delta <= ci_hi


def test_routing_to_pred_picks_correct_branch():
    df = pd.DataFrame({
        "cascade_pred": ["A", "B"],
        "llm_pred": ["X", "Y"],
        "silver_gt": ["A", "Y"],
    })
    decisions = np.array([False, True])
    correct = routing_to_pred(df, decisions)
    assert correct.tolist() == [1, 1]
