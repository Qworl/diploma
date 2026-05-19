"""Tests for sample_pasta_gold.py sampling logic."""
import pandas as pd
import pytest

from src.manual_label.sample_pasta_gold import (
    build_sample,
    SamplingError,
)


def _make_silver_extended() -> pd.DataFrame:
    """Minimal stand-in for pasta_stratified_silver_extended.parquet."""
    return pd.DataFrame({
        "code": [f"c{i}" for i in range(60)],
        "product_name": [f"Pasta {i}" for i in range(60)],
        "brands": [f"Brand{i % 10}" for i in range(60)],
        "ingredients_text": ["wheat"] * 60,
        "quantity": ["500g"] * 60,
        "lang": ["en"] * 60,
        "silver_grain_type": ["wheat"] * 60,
        "silver_pasta_shape": ["spaghetti"] * 60,
        "silver_is_filled": ["False"] * 60,
        "silver_is_organic": ["False"] * 60,
        "silver_is_gluten_free": ["False"] * 60,
        "silver_is_vegan": ["True"] * 60,
        "silver_nutri_score_grade": ["A"] * 30 + [None] * 30,
        "silver_protein_class": ["med"] * 30 + [None] * 30,
    })


def _make_split() -> pd.DataFrame:
    return pd.DataFrame({
        "code": [f"c{i}" for i in range(60)],
        "split": ["test"] * 20 + ["train"] * 30 + ["val"] * 10,
    })


def _make_disagreement() -> pd.DataFrame:
    # 10 disagreement codes OUTSIDE the test fold (in train range c20..c29)
    # so that priority A > B > C leaves all 10 disagreement codes for pool B.
    return pd.DataFrame({
        "code": [f"c{i}" for i in range(20, 30)],
        "attr": ["grain_type"] * 10,
        "cascade_pred": ["wheat"] * 10,
        "llm_pred": ["rice"] * 10,
    })


def test_priority_order_respected():
    df = build_sample(
        silver_extended=_make_silver_extended(),
        split=_make_split(),
        disagreement=_make_disagreement(),
        n_total=30, n_test=15, n_disagreement=10, n_control=5,
        seed=42,
    )
    sources = df["source"].value_counts()
    # test pool exhausted first (20 available but only 15 requested)
    assert sources.get("brand_disjoint_test", 0) == 15
    assert sources.get("disagreement", 0) == 10
    assert sources.get("gold_tier_control", 0) == 5
    assert len(df) == 30


def test_no_duplicate_codes():
    df = build_sample(
        silver_extended=_make_silver_extended(),
        split=_make_split(),
        disagreement=_make_disagreement(),
        n_total=30, n_test=15, n_disagreement=10, n_control=5,
        seed=42,
    )
    assert df["code"].is_unique


def test_manual_columns_initialised_empty():
    df = build_sample(
        silver_extended=_make_silver_extended(),
        split=_make_split(),
        disagreement=_make_disagreement(),
        n_total=10, n_test=5, n_disagreement=3, n_control=2,
        seed=42,
    )
    for attr in ("grain_type", "pasta_shape", "is_organic"):
        assert (df[f"manual_{attr}"] == "").all()
        assert (df[f"manual_{attr}_status"] == "empty").all()


def test_raises_when_pool_too_small():
    with pytest.raises(SamplingError):
        build_sample(
            silver_extended=_make_silver_extended(),
            split=_make_split(),
            disagreement=_make_disagreement(),
            n_total=500, n_test=400, n_disagreement=50, n_control=50,
            seed=42,
        )
