"""Silver-only headline uses silver as ground truth on brand-disjoint test split."""
import pandas as pd
from src.eval.headline_silver_only import compute_silver_headline


def test_silver_accuracy_excludes_silver_nulls():
    silver = pd.DataFrame([
        {"code": "1", "attr": "x", "silver_value": "A"},
        {"code": "2", "attr": "x", "silver_value": "B"},
        {"code": "3", "attr": "x", "silver_value": None},  # excluded
    ])
    cascade = pd.DataFrame([
        {"code": "1", "attr": "x", "predicted": "A", "layer": "ml"},
        {"code": "2", "attr": "x", "predicted": "X", "layer": "ml"},
        {"code": "3", "attr": "x", "predicted": "A", "layer": "ml"},
    ])
    out = compute_silver_headline(silver, cascade, category="beverages")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_non_null_silver"] == 2
    assert row["n_correct"] == 1
    assert abs(row["accuracy"] - 0.5) < 1e-9
    assert row["category"] == "beverages"
    assert row["attr"] == "x"
    assert 0 < row["wilson_lower"] < row["accuracy"] < row["wilson_upper"] < 1
    assert row["eval_set"] == "silver_brand_disjoint_test"
