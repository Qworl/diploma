"""Masking determinism, idempotency, and rate adherence."""
import pandas as pd

from src.eval.catalog_completion.masking import (
    mask_row,
    mask_dataframe,
    sentinel_for,
)


PROFILE = {
    "n_rows": 1000,
    "partner_attrs": {"product_name": 0.05, "brands": 0.20},
    "target_attrs": {"grain_type": 0.50, "is_organic": 0.10},
}


def test_mask_row_deterministic_per_code():
    row = pd.Series({"code": "abc", "product_name": "X", "brands": "Y",
                     "grain_type": "wheat", "is_organic": True})
    a = mask_row(row, PROFILE, global_seed=42)
    b = mask_row(row, PROFILE, global_seed=42)
    assert a.to_dict() == b.to_dict()


def test_mask_row_changes_with_seed():
    row = pd.Series({"code": "abc", "product_name": "X", "brands": "Y",
                     "grain_type": "wheat", "is_organic": True})
    samples = {tuple(mask_row(row, PROFILE, global_seed=s).fillna("__N__"))
               for s in range(50)}
    # Different seeds should produce more than one outcome.
    assert len(samples) > 1


def test_mask_dataframe_rate_matches_profile_within_tolerance():
    n = 5000
    df = pd.DataFrame({
        "code": [f"c{i}" for i in range(n)],
        "product_name": ["X"] * n,
        "brands": ["Y"] * n,
        "grain_type": ["wheat"] * n,
        "is_organic": [True] * n,
    })
    out, log = mask_dataframe(df, PROFILE, global_seed=0)
    # Target rate 0.5 +/- 0.03
    masked_rate = log[log["attr"] == "grain_type"]["masked"].mean()
    assert 0.46 < masked_rate < 0.54


def test_log_records_originals_for_recovery_check():
    df = pd.DataFrame({
        "code": ["a"], "product_name": ["X"], "brands": ["Y"],
        "grain_type": ["wheat"], "is_organic": [True],
    })
    _, log = mask_dataframe(df, PROFILE, global_seed=0)
    # Every (code, attr) should be in log with the original value.
    cells = log.set_index(["code", "attr"])
    assert cells.loc[("a", "grain_type"), "original_value"] == "wheat"


def test_sentinel_is_none_for_object_columns():
    assert sentinel_for("object") is None
    assert sentinel_for("bool") is None
