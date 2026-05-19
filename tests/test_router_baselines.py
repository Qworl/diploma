import numpy as np
import pandas as pd
import pytest

from src.pipeline.router.baselines import (
    static_confidence_threshold,
    per_attr_static_table,
    random_router,
    build_per_attr_table,
)


def test_static_confidence_threshold_sends_low_conf_to_llm():
    df = pd.DataFrame({
        "cascade_conf": [0.1, 0.5, 0.95, 0.99, 0.3],
    })
    decisions = static_confidence_threshold(df, threshold=0.4)
    assert decisions.tolist() == [True, False, False, False, True]


def test_static_confidence_threshold_zero_threshold():
    """τ=0 → nothing goes to LLM."""
    df = pd.DataFrame({"cascade_conf": [0.0, 0.5, 0.99]})
    decisions = static_confidence_threshold(df, threshold=0.0)
    assert decisions.tolist() == [False, False, False]


def test_static_confidence_threshold_full_threshold():
    """τ=1 → everything goes to LLM."""
    df = pd.DataFrame({"cascade_conf": [0.0, 0.5, 0.99]})
    decisions = static_confidence_threshold(df, threshold=1.0)
    assert decisions.tolist() == [True, True, True]


def test_build_per_attr_table_picks_better_path_per_pair():
    train = pd.DataFrame({
        "category": ["a", "a", "a", "a", "b", "b"],
        "attr": ["x", "x", "x", "x", "y", "y"],
        "cascade_correct": [1, 1, 1, 1, 0, 0],
        "llm_correct": [0, 0, 0, 0, 1, 1],
    })
    table = build_per_attr_table(train)
    assert table[("a", "x")] == False
    assert table[("b", "y")] == True


def test_per_attr_static_table_uses_table_decisions():
    table = {("a", "x"): False, ("b", "y"): True}
    df = pd.DataFrame({
        "category": ["a", "a", "b", "b"],
        "attr": ["x", "x", "y", "y"],
    })
    decisions = per_attr_static_table(df, table)
    assert decisions.tolist() == [False, False, True, True]


def test_random_router_respects_budget():
    df = pd.DataFrame({"cascade_conf": np.random.uniform(size=1000)})
    decisions = random_router(df, llm_budget=0.3, seed=42)
    rate = decisions.mean()
    assert 0.25 < rate < 0.35
