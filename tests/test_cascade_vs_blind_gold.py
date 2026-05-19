"""Tests for src.eval.cascade_vs_blind_gold.

Validates:
1. _apply_fixed_boundaries — cheeses fat_class boundaries
2. _compute_v2_accuracy — null exclusion and accuracy calculation
3. _get_v1_accuracy — audited-only filtering
4. Post-processing dispatched correctly per category
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.cascade_vs_blind_gold import (
    CHEESES_FAT_CLASS_BOUNDARIES,
    CHEESES_FAT_CLASS_LABELS,
    _apply_fixed_boundaries,
    _compute_v2_accuracy,
    _get_v1_accuracy,
    _postprocess_fat_class,
    _postprocess_protein_class,
)


# ---------------------------------------------------------------------------
# _apply_fixed_boundaries (cheeses fat_class: <15/15-25/25-32/>32)
# ---------------------------------------------------------------------------

class TestApplyFixedBoundaries:
    B = CHEESES_FAT_CLASS_BOUNDARIES
    L = CHEESES_FAT_CLASS_LABELS

    def test_very_low(self):
        assert _apply_fixed_boundaries(10.0, self.B, self.L) == "low"

    def test_at_boundary_15_goes_to_medium(self):
        assert _apply_fixed_boundaries(15.0, self.B, self.L) == "medium"

    def test_between_15_25(self):
        assert _apply_fixed_boundaries(20.0, self.B, self.L) == "medium"

    def test_at_boundary_25_goes_to_high(self):
        assert _apply_fixed_boundaries(25.0, self.B, self.L) == "high"

    def test_between_25_32(self):
        assert _apply_fixed_boundaries(28.0, self.B, self.L) == "high"

    def test_at_boundary_32_goes_to_very_high(self):
        assert _apply_fixed_boundaries(32.0, self.B, self.L) == "very_high"

    def test_above_32(self):
        assert _apply_fixed_boundaries(40.0, self.B, self.L) == "very_high"

    def test_none_returns_none(self):
        assert _apply_fixed_boundaries(None, self.B, self.L) is None

    def test_nan_returns_none(self):
        assert _apply_fixed_boundaries(float("nan"), self.B, self.L) is None


# ---------------------------------------------------------------------------
# _compute_v2_accuracy
# ---------------------------------------------------------------------------

class TestComputeV2Accuracy:
    def _make_preds(self, codes, attrs, predicted):
        rows = []
        for code, attr, pred in zip(codes, attrs, predicted):
            rows.append({"code": code, "attr": attr, "predicted": pred,
                         "confidence": 1.0, "layer": "ml"})
        return pd.DataFrame(rows)

    def _make_gold(self, codes, attrs, gold_values, cat="pasta"):
        rows = []
        for code, attr, val in zip(codes, attrs, gold_values):
            rows.append({
                "category": cat,
                "code": code,
                "attr": attr,
                "gold_value": val if val is not None else "",
                "gold_is_null": val is None,
                "signal_type": "text_derived",
                "opus_reasoning": None,
            })
        return pd.DataFrame(rows)

    def test_perfect_accuracy(self):
        preds = self._make_preds(["A", "B"], ["grain_type", "grain_type"], ["wheat", "rice"])
        gold = self._make_gold(["A", "B"], ["grain_type", "grain_type"], ["wheat", "rice"])
        result = _compute_v2_accuracy(preds, gold, "pasta")
        assert result["grain_type"]["accuracy"] == pytest.approx(1.0)

    def test_partial_accuracy(self):
        preds = self._make_preds(["A", "B"], ["grain_type", "grain_type"], ["wheat", "corn"])
        gold = self._make_gold(["A", "B"], ["grain_type", "grain_type"], ["wheat", "rice"])
        result = _compute_v2_accuracy(preds, gold, "pasta")
        assert result["grain_type"]["accuracy"] == pytest.approx(0.5)

    def test_null_gold_excluded_from_denominator(self):
        """Rows with gold_is_null=True must not count toward n or n_correct."""
        preds = self._make_preds(
            ["A", "B", "C"], ["grain_type"] * 3, ["wheat", "rice", "corn"]
        )
        gold = self._make_gold(
            ["A", "B", "C"], ["grain_type"] * 3, ["wheat", None, None]
        )
        result = _compute_v2_accuracy(preds, gold, "pasta")
        gt = result["grain_type"]
        assert gt["n"] == 1
        assert gt["n_null_gold"] == 2
        assert gt["accuracy"] == pytest.approx(1.0)

    def test_all_null_gives_nan_accuracy(self):
        preds = self._make_preds(["A"], ["grain_type"], ["wheat"])
        gold = self._make_gold(["A"], ["grain_type"], [None])
        result = _compute_v2_accuracy(preds, gold, "pasta")
        assert np.isnan(result["grain_type"]["accuracy"])

    def test_multiple_attrs(self):
        preds = self._make_preds(
            ["A", "A"], ["grain_type", "is_organic"], ["wheat", "True"]
        )
        gold = self._make_gold(
            ["A", "A"], ["grain_type", "is_organic"], ["wheat", "False"]
        )
        result = _compute_v2_accuracy(preds, gold, "pasta")
        assert result["grain_type"]["accuracy"] == pytest.approx(1.0)
        assert result["is_organic"]["accuracy"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _get_v1_accuracy
# ---------------------------------------------------------------------------

class TestGetV1Accuracy:
    def _make_v1(self, rows_spec):
        """rows_spec: list of (attr, status, mode, manual_value, cascade_pred)"""
        return pd.DataFrame([
            {
                "attr": r[0], "status": r[1], "mode": r[2],
                "manual_value": r[3], "cascade_pred": r[4],
                "code": "X", "cascade_conf": 1.0, "cascade_layer": "ml",
                "silver_value": r[3],
            }
            for r in rows_spec
        ])

    def test_perfect_blind_confirmed(self):
        v1 = self._make_v1([
            ("grain_type", "confirmed", "blind", "wheat", "wheat"),
            ("grain_type", "confirmed", "blind", "rice", "rice"),
        ])
        result = _get_v1_accuracy(v1)
        assert result["grain_type"] == pytest.approx(1.0)

    def test_partial_llm_confirmed(self):
        v1 = self._make_v1([
            ("grain_type", "confirmed", "llm", "wheat", "wheat"),
            ("grain_type", "confirmed", "llm", "rice", "corn"),
        ])
        result = _get_v1_accuracy(v1)
        assert result["grain_type"] == pytest.approx(0.5)

    def test_prefill_auto_excluded(self):
        """Rows with mode='prefill' and status='auto' must be excluded."""
        v1 = self._make_v1([
            ("grain_type", "confirmed", "llm", "wheat", "wheat"),
            ("grain_type", "auto", "prefill", "rice", "corn"),  # excluded
        ])
        result = _get_v1_accuracy(v1)
        assert result["grain_type"] == pytest.approx(1.0)

    def test_override_included(self):
        v1 = self._make_v1([
            ("grain_type", "override", "blind", "wheat", "wheat"),
        ])
        result = _get_v1_accuracy(v1)
        assert result["grain_type"] == pytest.approx(1.0)

    def test_manual_only_included(self):
        v1 = self._make_v1([
            ("grain_type", "manual_only", "llm", "wheat", "corn"),
        ])
        result = _get_v1_accuracy(v1)
        assert result["grain_type"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _postprocess_fat_class (smoke test)
# ---------------------------------------------------------------------------

class TestPostprocessFatClass:
    def test_no_fat_class_attr_unchanged(self):
        preds = pd.DataFrame([
            {"code": "A", "attr": "milk_source", "predicted": "cow", "confidence": 0.9, "layer": "ml"},
        ])
        result = _postprocess_fat_class(preds, ["A"])
        assert result.equals(preds)

    def test_returns_dataframe(self):
        preds = pd.DataFrame([
            {"code": "A", "attr": "fat_class", "predicted": "high", "confidence": 0.9, "layer": "ml"},
        ])
        result = _postprocess_fat_class(preds, ["A"])
        assert isinstance(result, pd.DataFrame)
        assert result["layer"].iloc[0] == "bucket_rule"


# ---------------------------------------------------------------------------
# _postprocess_protein_class (smoke test)
# ---------------------------------------------------------------------------

class TestPostprocessProteinClass:
    def test_no_protein_class_attr_unchanged(self):
        preds = pd.DataFrame([
            {"code": "A", "attr": "is_organic", "predicted": "True", "confidence": 0.9, "layer": "ml"},
        ])
        bucket_spec = {
            "protein_class": {
                "feature": "proteins_100g",
                "boundaries": [1.0, 10.0, 20.0],
                "labels": ["0", "low", "med", "high"],
                "n_train": 10,
                "best_accuracy": 0.8,
            }
        }
        result = _postprocess_protein_class(preds, bucket_spec, ["A"])
        assert result.equals(preds)

    def test_layer_updated_for_protein_class(self):
        preds = pd.DataFrame([
            {"code": "A", "attr": "protein_class", "predicted": "low", "confidence": 0.7, "layer": "ml"},
        ])
        bucket_spec = {
            "protein_class": {
                "feature": "proteins_100g",
                "boundaries": [1.0, 10.0, 20.0],
                "labels": ["0", "low", "med", "high"],
                "n_train": 10,
                "best_accuracy": 0.8,
            }
        }
        result = _postprocess_protein_class(preds, bucket_spec, ["A"])
        # Layer should always be updated to bucket_rule
        assert result["layer"].iloc[0] == "bucket_rule"
