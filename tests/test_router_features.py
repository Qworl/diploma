import numpy as np
import pandas as pd
import pytest

from src.pipeline.router.features import (
    featurize,
    detect_lang,
    build_brand_set,
    ATTR_TYPE,
    FEATURE_COLUMNS,
)


def test_attr_type_taxonomy_covers_main_attrs():
    """Every main attribute from FOOD_CATS § 1 is classified."""
    must_have = [
        ("pasta", "grain_type"),
        ("pasta", "is_organic"),
        ("pasta", "nutri_score_grade"),
        ("chocolate", "chocolate_type"),
        ("beverages", "beverage_type"),
        ("cosmetics", "product_type"),
    ]
    for key in must_have:
        assert key in ATTR_TYPE, f"Missing taxonomy for {key}"


def test_detect_lang_basic():
    """Heuristic returns one of 5 EU langs."""
    assert detect_lang("Spaghetti integrali con olive") in ("it", "en", "es")
    assert detect_lang("Schokolade mit Vollmilch") in ("de", "en")
    assert detect_lang("") == "unknown"


def test_build_brand_set_lowercases_and_dedupes():
    df = pd.DataFrame({"brands": ["Barilla", "Nestle ,Maggi", "barilla", None]})
    brands = build_brand_set(df)
    assert "barilla" in brands
    assert "nestle" in brands
    assert "maggi" in brands


def test_featurize_outputs_consistent_shape_and_columns():
    """Featurize 5 rows × 2 cats → matrix shape (5, N_features), columns named."""
    df = pd.DataFrame({
        "category": ["pasta", "pasta", "chocolate", "chocolate", "pasta"],
        "code": ["1", "2", "3", "4", "5"],
        "attr": ["grain_type", "is_organic", "chocolate_type", "cocoa_percentage", "pasta_shape"],
        "cascade_layer": ["ml", "ml", "regex", "regex", "ml"],
        "cascade_conf": [0.9, 0.8, 0.95, 0.7, 0.6],
        "product_name": ["Barilla spaghetti", "Bio penne", "Lindt dark", "Milka 70%", "Penne rigate"],
        "brands": ["Barilla", "Carrefour Bio", "Lindt", "Milka", "Barilla"],
    })
    brand_set = {"barilla", "carrefour bio", "lindt", "milka"}
    X, cols = featurize(df, brand_set=brand_set)
    assert X.shape == (5, len(cols))
    assert len(cols) >= 8  # at least the 8 features; one-hot expands to more
    # brand_known should be 1 for all 5 here
    bk_idx = cols.index("brand_known")
    assert (X[:, bk_idx] == 1).all()


def test_featurize_brand_unknown_when_missing():
    df = pd.DataFrame({
        "category": ["pasta"],
        "code": ["x"],
        "attr": ["grain_type"],
        "cascade_layer": ["ml"],
        "cascade_conf": [0.9],
        "product_name": ["NoBrand spaghetti"],
        "brands": ["UnknownBrandXYZ"],
    })
    brand_set = {"barilla"}
    X, cols = featurize(df, brand_set=brand_set)
    bk_idx = cols.index("brand_known")
    assert X[0, bk_idx] == 0
