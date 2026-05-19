"""Regression test for is_organic=False being lost during build_sample.

The original `s.get(..., "") or ""` in build_sample treated bool False as falsy
and silently dropped it to "". Regenerating the CSV after the fix surfaced that
the on-disk artifact contained NaN where False should have been.
"""
import pandas as pd
import pytest

from src.manual_label.sample_pasta_gold import build_sample


def _make_silver_with_falses() -> pd.DataFrame:
    return pd.DataFrame({
        "code": [f"c{i}" for i in range(60)],
        "product_name": [f"Pasta {i}" for i in range(60)],
        "brands": [f"Brand{i % 10}" for i in range(60)],
        "ingredients_text": ["wheat"] * 60,
        "quantity": ["500g"] * 60,
        "lang": ["en"] * 60,
        "silver_grain_type": ["wheat"] * 60,
        "silver_pasta_shape": ["spaghetti"] * 60,
        "silver_is_filled": [False] * 60,        # all False
        "silver_is_organic": [False] * 40 + [True] * 20,
        "silver_is_gluten_free": [False] * 60,
        "silver_is_vegan": [True] * 60,
        "silver_nutri_score_grade": ["A"] * 30 + [None] * 30,
        "silver_protein_class": ["med"] * 30 + [None] * 30,
    })


def _make_split() -> pd.DataFrame:
    return pd.DataFrame({
        "code": [f"c{i}" for i in range(60)],
        "split": ["test"] * 20 + ["train"] * 30 + ["val"] * 10,
    })


def test_bool_false_survives_build_sample():
    df = build_sample(
        silver_extended=_make_silver_with_falses(),
        split=_make_split(),
        disagreement=pd.DataFrame(columns=["code", "attr", "cascade_pred", "llm_pred"]),
        n_total=20, n_test=15, n_disagreement=0, n_control=5,
        seed=42,
    )
    # is_filled is False for ALL rows — none should be empty/NaN
    is_filled_vals = df["silver_is_filled"].tolist()
    assert all(v is False or v == False for v in is_filled_vals), \
        f"build_sample lost False values: {is_filled_vals}"

    # is_organic has both False and True — sample should preserve some of each
    is_organic_vals = df["silver_is_organic"]
    assert (is_organic_vals == False).sum() > 0, "no False values survived"
