"""Tests for src.eval.blind_vs_prefill_analysis."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.eval.blind_vs_prefill_analysis import (
    compute_agreement,
    compute_per_attr_metrics,
    compute_flip_direction,
)


_PREFILL = {
    "code1": {"attr_a": {"value": "X"}, "attr_b": {"value": "Y"}},
    "code2": {"attr_a": {"value": "X"}, "attr_b": {"value": "Z"}},
    "code3": {"attr_a": {"value": "X"}, "attr_b": {"value": "Y"}},
    "code4": {"attr_a": {"value": "X"}, "attr_b": {"value": "Z"}},
}

_BLIND = {
    "code1": {"attr_a": {"value": "X"}, "attr_b": {"value": "Y"}},  # full agree
    "code2": {"attr_a": {"value": "X"}, "attr_b": {"value": "W"}},  # b disagrees
    "code3": {"attr_a": {"value": "X"}, "attr_b": {"value": None}}, # b refuses
    "code4": {"attr_a": {"value": "Q"}, "attr_b": {"value": "Z"}},  # a disagrees
}

_SILVER = pd.DataFrame([
    {"code": "code1", "attr_a": "X", "attr_b": "Y"},
    {"code": "code2", "attr_a": "X", "attr_b": "W"},  # silver matches blind
    {"code": "code3", "attr_a": "X", "attr_b": "Y"},
    {"code": "code4", "attr_a": "X", "attr_b": "Z"},
])


def test_compute_agreement_overall_excludes_nulls():
    """Cells where blind is null are excluded from agreement denominator."""
    result = compute_agreement(_PREFILL, _BLIND)
    # Per cell:
    # code1: a=X==X, b=Y==Y → 2/2
    # code2: a=X==X, b=Z!=W → 1/2
    # code3: a=X==X, b excluded (null) → 1/1
    # code4: a=X!=Q, b=Z==Z → 1/2
    # Total non-null cells = 7. Agreements = 5. → 5/7
    assert result["overall_agreement"] == pytest.approx(5 / 7)
    assert result["n_non_null_cells"] == 7
    assert result["n_total_cells"] == 8


def test_compute_per_attr_metrics():
    df = compute_per_attr_metrics(_PREFILL, _BLIND)
    # attr_a: blind never null. agreements: code1, code2, code3 (X==X), code4 X!=Q
    a = df[df["attr"] == "attr_a"].iloc[0]
    assert a["n_total"] == 4
    assert a["n_non_null"] == 4
    assert a["agreement"] == pytest.approx(3 / 4)
    assert a["refusal_rate"] == 0.0

    # attr_b: code3 is null. non-null cells = 3. Agreements: code1 (Y==Y), code4 (Z==Z), not code2. = 2/3
    b = df[df["attr"] == "attr_b"].iloc[0]
    assert b["n_total"] == 4
    assert b["n_non_null"] == 3
    assert b["agreement"] == pytest.approx(2 / 3)
    assert b["refusal_rate"] == pytest.approx(1 / 4)


def test_compute_flip_direction_with_silver():
    """When blind disagrees with prefill, count which side silver matches."""
    df = compute_flip_direction(_PREFILL, _BLIND, _SILVER)
    # Disagreements (excluding nulls):
    # code2 attr_b: prefill=Z, blind=W, silver=W → blind matches silver (flip toward silver)
    # code4 attr_a: prefill=X, blind=Q, silver=X → prefill matches silver (flip away from silver)
    by_attr = df.set_index("attr")
    assert by_attr.loc["attr_b", "flip_to_silver"] == 1
    assert by_attr.loc["attr_b", "flip_away_silver"] == 0
    assert by_attr.loc["attr_a", "flip_to_silver"] == 0
    assert by_attr.loc["attr_a", "flip_away_silver"] == 1


def test_compute_agreement_handles_missing_codes():
    """Codes in prefill but not blind (or vice versa) are skipped."""
    blind_partial = {"code1": _BLIND["code1"]}
    result = compute_agreement(_PREFILL, blind_partial)
    # Only code1 considered, 2 cells, 2 agreements
    assert result["overall_agreement"] == 1.0
    assert result["n_non_null_cells"] == 2


def test_compute_per_attr_includes_cohen_kappa():
    df = compute_per_attr_metrics(_PREFILL, _BLIND)
    # Just check column present and not NaN for attrs with variance
    assert "cohen_kappa" in df.columns
