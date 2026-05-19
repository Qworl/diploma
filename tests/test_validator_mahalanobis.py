"""Unit tests for src.eval.validator_mahalanobis."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.validator_mahalanobis import (
    fit_per_attr_mahalanobis,
    score_per_attr_mahalanobis,
)


def test_fit_returns_one_fit_per_attr():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    df = {
        "grain_type": np.array(["wheat"] * 25 + ["spelt"] * 25),
        "is_filled":  np.array(["True"]  * 30 + ["False"] * 20),
    }
    fits = fit_per_attr_mahalanobis(X, df, attrs=("grain_type", "is_filled"))
    assert set(fits.keys()) == {"grain_type", "is_filled"}
    assert set(fits["grain_type"].classes) == {"wheat", "spelt"}
    assert set(fits["is_filled"].classes) == {"True", "False"}


def test_score_returns_one_distance_per_row_per_attr():
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(40, 8))
    labels = {"grain_type": np.array(["wheat"] * 20 + ["spelt"] * 20)}
    fits = fit_per_attr_mahalanobis(X_train, labels, attrs=("grain_type",))
    X_test = rng.normal(size=(5, 8))
    scores = score_per_attr_mahalanobis(X_test, fits)
    assert set(scores.keys()) == {"grain_type"}
    assert scores["grain_type"].shape == (5,)
    assert (scores["grain_type"] >= 0).all()


def test_skips_attr_with_only_one_class():
    X = np.zeros((10, 4))
    labels = {"grain_type": np.array(["wheat"] * 10)}  # single class
    fits = fit_per_attr_mahalanobis(X, labels, attrs=("grain_type",))
    assert "grain_type" not in fits  # cannot define a within-class scatter


def test_skips_attr_with_missing_labels():
    X = np.zeros((10, 4))
    labels = {"grain_type": np.array([None] * 10)}
    fits = fit_per_attr_mahalanobis(X, labels, attrs=("grain_type",))
    assert "grain_type" not in fits
