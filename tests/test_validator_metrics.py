"""Unit tests for validator_metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.validator_metrics import (
    auc_for_validator,
    precision_recall_at_k,
    random_baseline,
    static_policy_baseline,
)


def _df(scores, errors, attrs=None):
    n = len(scores)
    return pd.DataFrame({
        "score": scores,
        "is_error": errors,
        "attr": attrs if attrs is not None else ["x"] * n,
    })


def test_auc_perfect_separation():
    df = _df([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
    assert auc_for_validator(df["score"], df["is_error"]) == pytest.approx(1.0)


def test_auc_single_class_returns_none():
    df = _df([0.1, 0.5, 0.8], [False, False, False])
    assert auc_for_validator(df["score"], df["is_error"]) is None


def test_auc_with_nans_drops_them():
    df = _df([0.1, np.nan, 0.8, 0.9], [False, True, True, True])
    # Only 3 rows survive; AUC well-defined
    val = auc_for_validator(df["score"], df["is_error"])
    assert val is not None and 0.0 <= val <= 1.0


def test_precision_recall_at_top_50_pct():
    # 4 rows, 2 errors at top-2 scores → precision=1.0, recall=1.0
    df = _df([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
    p, r, n = precision_recall_at_k(df["score"], df["is_error"], k=0.5)
    assert n == 2
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(1.0)


def test_precision_recall_at_top_25_pct_imperfect():
    df = _df([0.1, 0.5, 0.8, 0.9], [True, False, False, True])
    # Top-1 (k=0.25) selects score=0.9, which is an error
    p, r, n = precision_recall_at_k(df["score"], df["is_error"], k=0.25)
    assert n == 1
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(0.5)  # 1 of 2 errors caught


def test_random_baseline_expectation():
    df = _df([0.0] * 10, [True] * 3 + [False] * 7)
    p, r = random_baseline(df["is_error"], k=0.2)
    # Expected precision = base rate = 0.3; recall = k = 0.2
    assert p == pytest.approx(0.3)
    assert r == pytest.approx(0.2)


def test_static_policy_baseline():
    df = pd.DataFrame({
        "attr":     ["pasta_shape"] * 4 + ["is_filled"] * 4,
        "is_error": [True, True, False, False, False, False, True, False],
    })
    # Always route pasta_shape (4 cells, 2 errors), never is_filled
    result = static_policy_baseline(df, attrs_to_route={"pasta_shape"})
    assert result["routing_rate"] == pytest.approx(4 / 8)
    assert result["precision"] == pytest.approx(2 / 4)
    assert result["recall"] == pytest.approx(2 / 3)  # 2 of 3 total errors caught
    assert result["n_routed"] == 4
