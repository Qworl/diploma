"""Headline v2 must compute refusal-aware accuracy + Wilson CI + gold_coverage."""
import pandas as pd
import pytest

from src.eval.headline_v2 import compute_headline


def _make_inputs():
    # 5 cells: 3 non-null gold (1 correct, 1 wrong, 1 cascade-abstain),
    # 2 gold=null (excluded). cascade-abstain → counts as miss per §4.2.
    gold = pd.DataFrame([
        {"category": "pasta", "code": "1", "attr": "x",
         "gold_value": "A", "gold_is_null": False, "signal_type": "text_derived"},
        {"category": "pasta", "code": "2", "attr": "x",
         "gold_value": "B", "gold_is_null": False, "signal_type": "text_derived"},
        {"category": "pasta", "code": "3", "attr": "x",
         "gold_value": "A", "gold_is_null": False, "signal_type": "text_derived"},
        {"category": "pasta", "code": "4", "attr": "x",
         "gold_value": None, "gold_is_null": True, "signal_type": "text_derived"},
        {"category": "pasta", "code": "5", "attr": "x",
         "gold_value": None, "gold_is_null": True, "signal_type": "text_derived"},
    ])
    cascade = pd.DataFrame([
        {"code": "1", "attr": "x", "predicted": "A", "confidence": 0.9, "layer": "ml"},
        {"code": "2", "attr": "x", "predicted": "C", "confidence": 0.8, "layer": "ml"},
        {"code": "3", "attr": "x", "predicted": None, "confidence": None, "layer": "abstain"},
        {"code": "4", "attr": "x", "predicted": "Z", "confidence": 0.6, "layer": "ml"},
        {"code": "5", "attr": "x", "predicted": "Z", "confidence": 0.6, "layer": "ml"},
    ])
    return gold, cascade


def test_compute_headline_aggregates_per_cat_attr():
    gold, cascade = _make_inputs()
    out = compute_headline(gold, cascade, category="pasta")
    assert len(out) == 1  # one (cat, attr) row
    row = out.iloc[0]
    assert row["n_non_null_gold"] == 3
    assert row["n_correct"] == 1  # only code 1
    assert abs(row["accuracy"] - 1/3) < 1e-9
    assert row["gold_coverage_rate"] == 3 / 5
    assert 0 < row["wilson_lower"] < row["accuracy"] < row["wilson_upper"] < 1
    assert row["signal_type"] == "text_derived"
