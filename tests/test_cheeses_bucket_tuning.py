"""Tests for src.eval.cheeses_bucket_tuning.

Validates:
1. _apply_boundaries — deterministic label assignment
2. _score_boundaries — accuracy computation
3. _grid_search_boundaries — deterministic boundary derivation
4. apply_bucket_boundaries — identical round-trip via saved spec
"""
from __future__ import annotations

import pytest

from src.eval.cheeses_bucket_tuning import (
    _apply_boundaries,
    _grid_search_boundaries,
    _score_boundaries,
    apply_bucket_boundaries,
)


# ---------------------------------------------------------------------------
# _apply_boundaries
# ---------------------------------------------------------------------------

class TestApplyBoundaries:
    BOUNDARIES = [1.0, 10.0, 20.0]  # 4 classes
    LABELS = ["0", "low", "med", "high"]

    def test_below_first_boundary(self):
        assert _apply_boundaries(0.5, self.BOUNDARIES, self.LABELS) == "0"

    def test_at_first_boundary_goes_to_second_class(self):
        # value == boundary → lands in NEXT class (left-open convention)
        assert _apply_boundaries(1.0, self.BOUNDARIES, self.LABELS) == "low"

    def test_between_boundaries(self):
        assert _apply_boundaries(5.0, self.BOUNDARIES, self.LABELS) == "low"

    def test_at_second_boundary(self):
        assert _apply_boundaries(10.0, self.BOUNDARIES, self.LABELS) == "med"

    def test_above_all_boundaries(self):
        assert _apply_boundaries(25.0, self.BOUNDARIES, self.LABELS) == "high"

    def test_none_value_returns_none(self):
        assert _apply_boundaries(None, self.BOUNDARIES, self.LABELS) is None

    def test_nan_returns_none(self):
        import math
        assert _apply_boundaries(float("nan"), self.BOUNDARIES, self.LABELS) is None

    def test_string_numeric(self):
        # Should coerce to float
        assert _apply_boundaries("5.0", self.BOUNDARIES, self.LABELS) == "low"

    def test_zero_value(self):
        assert _apply_boundaries(0.0, self.BOUNDARIES, self.LABELS) == "0"


# ---------------------------------------------------------------------------
# _score_boundaries
# ---------------------------------------------------------------------------

class TestScoreBoundaries:
    def test_perfect_score(self):
        # boundaries at 1.0, 10.0, 20.0
        values = [0.5, 5.0, 15.0, 25.0]
        gold = ["0", "low", "med", "high"]
        acc = _score_boundaries(values, gold, [1.0, 10.0, 20.0], ["0", "low", "med", "high"])
        assert acc == pytest.approx(1.0)

    def test_zero_score(self):
        values = [25.0, 25.0, 25.0, 25.0]
        gold = ["0", "low", "med", "med"]
        acc = _score_boundaries(values, gold, [1.0, 10.0, 20.0], ["0", "low", "med", "high"])
        assert acc == pytest.approx(0.0)

    def test_partial_score(self):
        values = [0.5, 5.0, 5.0, 25.0]
        gold = ["0", "low", "high", "high"]  # 3rd is wrong
        acc = _score_boundaries(values, gold, [1.0, 10.0, 20.0], ["0", "low", "med", "high"])
        assert acc == pytest.approx(3 / 4)

    def test_none_values_excluded_from_denominator(self):
        values = [None, 5.0, None, 25.0]
        gold = [None, "low", None, "high"]
        acc = _score_boundaries(values, gold, [1.0, 10.0, 20.0], ["0", "low", "med", "high"])
        assert acc == pytest.approx(1.0)

    def test_all_none_returns_zero(self):
        acc = _score_boundaries([None, None], [None, None], [1.0, 10.0, 20.0], ["0", "low", "med", "high"])
        assert acc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _grid_search_boundaries — determinism
# ---------------------------------------------------------------------------

class TestGridSearchBoundaries:
    def _make_clean_data(self):
        """Synthetic data with known clean boundaries at 2.0 and 12.0."""
        values = (
            [0.5] * 5    # → "low"
            + [5.0] * 20  # → "med"
            + [20.0] * 5  # → "high"
        )
        gold = ["low"] * 5 + ["med"] * 20 + ["high"] * 5
        return values, gold

    def test_determinism_same_call(self):
        values, gold = self._make_clean_data()
        b1, a1 = _grid_search_boundaries(values, gold, ["low", "med", "high"])
        b2, a2 = _grid_search_boundaries(values, gold, ["low", "med", "high"])
        assert b1 == b2
        assert a1 == pytest.approx(a2)

    def test_determinism_with_fixed_grid(self):
        values, gold = self._make_clean_data()
        grid = [10, 50, 90]
        b1, a1 = _grid_search_boundaries(values, gold, ["low", "med", "high"], percentile_grid=grid)
        b2, a2 = _grid_search_boundaries(values, gold, ["low", "med", "high"], percentile_grid=grid)
        assert b1 == b2

    def test_accuracy_positive(self):
        values, gold = self._make_clean_data()
        _, acc = _grid_search_boundaries(values, gold, ["low", "med", "high"])
        assert acc > 0.5

    def test_four_class_scheme(self):
        """4-class scheme requires 3 boundaries; result must have length 3."""
        values = [0.0] * 5 + [2.0] * 20 + [12.0] * 10 + [22.0] * 5
        gold = ["0"] * 5 + ["low"] * 20 + ["med"] * 10 + ["high"] * 5
        boundaries, _ = _grid_search_boundaries(values, gold, ["0", "low", "med", "high"])
        assert len(boundaries) == 3

    def test_raises_if_no_valid_values(self):
        with pytest.raises(ValueError, match="No valid numeric values"):
            _grid_search_boundaries([None, None], ["low", "high"], ["low", "high"])


# ---------------------------------------------------------------------------
# apply_bucket_boundaries — round-trip consistency
# ---------------------------------------------------------------------------

class TestApplyBucketBoundaries:
    SPEC: dict = {
        "protein_class": {
            "feature": "proteins_100g",
            "boundaries": [1.0, 10.0, 20.0],
            "labels": ["0", "low", "med", "high"],
            "n_train": 100,
            "best_accuracy": 0.85,
        }
    }

    def test_applies_boundaries_correctly(self):
        assert apply_bucket_boundaries(5.0, "protein_class", self.SPEC) == "low"

    def test_none_value_returns_none(self):
        assert apply_bucket_boundaries(None, "protein_class", self.SPEC) is None

    def test_unknown_attr_returns_none(self):
        assert apply_bucket_boundaries(5.0, "unknown_attr", self.SPEC) is None

    def test_consistent_with_direct_call(self):
        """apply_bucket_boundaries must produce same result as _apply_boundaries."""
        from src.eval.cheeses_bucket_tuning import _apply_boundaries
        for v in [0.5, 5.0, 15.0, 25.0]:
            assert (
                apply_bucket_boundaries(v, "protein_class", self.SPEC)
                == _apply_boundaries(v, [1.0, 10.0, 20.0], ["0", "low", "med", "high"])
            )
