"""Tests for src.eval.attribute_taxonomy."""
from __future__ import annotations

import pandas as pd
import pytest

from src.eval.attribute_taxonomy import (
    SIGNAL_TYPE,
    build_taxonomy_dataframe,
    classify_attribute,
)


def test_classify_known_tag_derived():
    assert classify_attribute("pasta", "is_organic") == "tag_derived"
    assert classify_attribute("chocolate", "is_organic") == "tag_derived"
    assert classify_attribute("pasta", "is_vegan") == "tag_derived"
    assert classify_attribute("pasta", "is_gluten_free") == "tag_derived"


def test_classify_known_text_derived():
    assert classify_attribute("pasta", "pasta_shape") == "text_derived"
    assert classify_attribute("pasta", "grain_type") == "text_derived"
    assert classify_attribute("chocolate", "chocolate_type") == "text_derived"
    assert classify_attribute("cheeses", "milk_source") == "text_derived"


def test_classify_known_nutri_derived():
    assert classify_attribute("pasta", "nutri_score_grade") == "nutri_derived"
    assert classify_attribute("pasta", "protein_class") == "nutri_derived"
    assert classify_attribute("cheeses", "fat_class") == "nutri_derived"


def test_classify_unknown_returns_text_derived_default():
    """Default when not explicitly mapped."""
    assert classify_attribute("pasta", "unknown_made_up_attr") == "text_derived"


def test_build_taxonomy_dataframe_has_all_columns():
    df = build_taxonomy_dataframe()
    for col in ["category", "attr", "signal_type", "primary_path",
                "secondary_path", "multi_source"]:
        assert col in df.columns


def test_build_taxonomy_marks_multi_source_for_is_organic():
    df = build_taxonomy_dataframe()
    is_organic = df[df["attr"] == "is_organic"]
    assert (is_organic["multi_source"] == True).all()
    assert (is_organic["primary_path"] == "labels_tags").all()
    assert (is_organic["secondary_path"].notna()).all()


def test_build_taxonomy_includes_all_six_food_domains():
    df = build_taxonomy_dataframe()
    cats = set(df["category"].unique())
    assert {"pasta", "chocolate", "cheeses", "beverages", "cereals", "cosmetics"}.issubset(cats)


def test_signal_type_constants():
    assert SIGNAL_TYPE.TAG == "tag_derived"
    assert SIGNAL_TYPE.TEXT == "text_derived"
    assert SIGNAL_TYPE.NUTRI == "nutri_derived"
