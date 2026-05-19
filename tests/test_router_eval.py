import numpy as np
import pandas as pd
import pytest

from src.eval.router_eval import (
    compute_routed_accuracy,
    pareto_curve_router,
    pareto_curve_static,
    pareto_curve_random,
)


def _toy_df():
    """5 rows: cascade right on all (in this toy), LLM right on different rows."""
    return pd.DataFrame({
        "category": ["a"] * 5,
        "attr": ["x"] * 5,
        "cascade_pred": ["A", "B", "A", "B", "A"],
        "llm_pred":     ["B", "B", "B", "A", "A"],
        "silver_gt":    ["A", "B", "A", "B", "A"],
        "cascade_conf": [0.99, 0.7, 0.5, 0.3, 0.1],
    })


def test_compute_routed_accuracy_pure_cascade():
    df = _toy_df()
    decisions = np.array([False] * 5)
    acc, cost = compute_routed_accuracy(df, decisions)
    assert acc == 1.0
    assert cost == 0.0


def test_compute_routed_accuracy_pure_llm():
    df = _toy_df()
    decisions = np.array([True] * 5)
    acc, cost = compute_routed_accuracy(df, decisions)
    # llm_pred: B B B A A; silver: A B A B A → 2 correct (idx 1, 4) = 0.4
    assert acc == 0.4
    assert cost == 1.0


def test_compute_routed_accuracy_mixed():
    df = _toy_df()
    decisions = np.array([False, False, False, True, True])
    acc, cost = compute_routed_accuracy(df, decisions)
    # First 3 cascade: A B A vs A B A → 3 correct
    # Last 2 llm: A A vs B A → 1 correct (idx 4)
    # Total: 4/5 = 0.8
    assert acc == 0.8
    assert cost == 0.4


def test_pareto_curve_static_monotonic_cost():
    df = _toy_df()
    pareto = pareto_curve_static(df, thresholds=np.linspace(0, 1, 11))
    costs = pareto["cost"].values
    assert all(costs[i] <= costs[i + 1] for i in range(len(costs) - 1))


def test_pareto_curve_router_returns_required_columns():
    df = _toy_df()
    proba = np.array([0.95, 0.5, 0.4, 0.3, 0.1])
    pareto = pareto_curve_router(df, proba, thresholds=np.linspace(0, 1, 11))
    assert {"strategy", "threshold", "cost", "accuracy"}.issubset(pareto.columns)
    assert (pareto["strategy"] == "router").all()
