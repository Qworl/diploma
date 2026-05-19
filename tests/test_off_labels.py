"""Tests for OFF labels pre-step (apply_off_labels function)."""
import pytest

from src.pipeline.off_labels import apply_off_labels
from src.pipeline.off_labels.rules import (
    _type_a_bool,
    _type_b_multiclass,
    _type_c_numeric,
    _type_d_direct,
)
from src.pipeline.schemas import PASTA_SCHEMA


# --- Type A tests ---

def test_type_a_organic_positive():
    """en:organic в labels_tags → True"""
    row = {"labels_tags": "en:organic,en:eu-organic"}
    assert _type_a_bool(row, "is_organic") is True


def test_type_a_organic_absent():
    """no organic tag → False (assume absence = False per spec)"""
    row = {"labels_tags": "en:no-additives"}
    assert _type_a_bool(row, "is_organic") is False


def test_type_a_organic_missing_field():
    """labels_tags отсутствует → None (unknown)"""
    row = {}
    assert _type_a_bool(row, "is_organic") is None


def test_type_a_gluten_free_variants():
    """разные варианты gluten-free тэгов"""
    for tag in ["en:no-gluten", "en:gluten-free", "en:sans-gluten"]:
        row = {"labels_tags": tag}
        assert _type_a_bool(row, "is_gluten_free") is True


def test_type_a_contains_nuts():
    """allergens_tags ∋ en:nuts → True"""
    row = {"allergens_tags": "en:nuts,en:milk"}
    assert _type_a_bool(row, "contains_nuts") is True


def test_type_a_unknown_attribute():
    """неизвестный атрибут → None (а не crash)"""
    row = {"labels_tags": "en:organic"}
    assert _type_a_bool(row, "completely_made_up_attr") is None


def test_type_a_whole_grain_via_category():
    """is_whole_grain detects via categories_tags or labels_tags (hybrid TYPE_AC rule).
    NOTE: is_whole_grain moved to TYPE_AC, but test_off_labels still documents the behavior."""
    from src.pipeline.off_labels.rules import _type_ac_hybrid
    row = {"categories_tags": "en:pastas,en:whole-grain-pastas", "labels_tags": ""}
    assert _type_ac_hybrid(row, "is_whole_grain") is True


def test_type_a_whole_grain_neither_field():
    """When both source fields missing → None"""
    from src.pipeline.off_labels.rules import _type_ac_hybrid
    row = {}
    assert _type_ac_hybrid(row, "is_whole_grain") is None


# --- Type B tests ---

def test_type_b_pasta_shape_spaghetti():
    row = {"categories_tags": "en:pastas,en:dry-pastas,en:spaghetti"}
    assert _type_b_multiclass(row, "pasta_shape") == "spaghetti"


def test_type_b_pasta_shape_unknown():
    """en:dry-pastas без specific shape → None (gap, нужен LLM)"""
    row = {"categories_tags": "en:pastas,en:dry-pastas"}
    assert _type_b_multiclass(row, "pasta_shape") is None


def test_type_b_chocolate_dark():
    row = {"categories_tags": "en:chocolates,en:dark-chocolates"}
    assert _type_b_multiclass(row, "chocolate_type") == "dark"


def test_type_b_beverage_water():
    row = {"categories_tags": "en:beverages,en:waters,en:mineral-waters"}
    assert _type_b_multiclass(row, "beverage_type") == "water"


def test_type_b_pasta_shape_specific_beats_generic_noodles():
    """Trek D audit: en:tagliatelle should beat en:noodles regardless of order."""
    # tagliatelle FIRST in tag list — old behavior would also get tagliatelle
    row1 = {"categories_tags": "en:tagliatelle,en:pastas,en:noodles"}
    assert _type_b_multiclass(row1, "pasta_shape") == "tagliatelle"
    # tagliatelle LAST in tag list — old behavior returned noodles, new returns tagliatelle
    row2 = {"categories_tags": "en:noodles,en:pastas,en:tagliatelle"}
    assert _type_b_multiclass(row2, "pasta_shape") == "tagliatelle"


def test_type_b_pasta_shape_specific_subtypes():
    """Trek D audit: en:egg-tagliatelle / en:rice-vermicelli / en:durum-wheat-macaroni."""
    assert _type_b_multiclass(
        {"categories_tags": "en:egg-tagliatelle,en:noodles"}, "pasta_shape"
    ) == "tagliatelle"
    assert _type_b_multiclass(
        {"categories_tags": "en:rice-vermicelli,en:noodles"}, "pasta_shape"
    ) == "vermicelli"
    assert _type_b_multiclass(
        {"categories_tags": "en:cellophane-noodles,en:noodles"}, "pasta_shape"
    ) == "vermicelli"
    assert _type_b_multiclass(
        {"categories_tags": "en:durum-wheat-macaroni,en:pastas"}, "pasta_shape"
    ) == "macaroni"


def test_type_b_pasta_shape_pappardelle_fettuccine():
    """Ribbon-pasta synonyms map to tagliatelle (closest schema value)."""
    assert _type_b_multiclass(
        {"categories_tags": "en:pappardelle"}, "pasta_shape"
    ) == "tagliatelle"
    assert _type_b_multiclass(
        {"categories_tags": "en:fettuccine"}, "pasta_shape"
    ) == "tagliatelle"


def test_type_b_grain_legume_pastas():
    """Trek D audit: legume pastas → "other" (no legume bucket in schema)."""
    assert _type_b_multiclass(
        {"categories_tags": "en:lentil-pastas,en:pastas"}, "grain_type"
    ) == "other"
    assert _type_b_multiclass(
        {"categories_tags": "en:chickpea-pastas"}, "grain_type"
    ) == "other"
    assert _type_b_multiclass(
        {"categories_tags": "en:konjac-pasta"}, "grain_type"
    ) == "other"


def test_type_b_grain_legume_singular():
    """OFF taxonomy is inconsistent plural/singular — cover both forms."""
    # Surfaced by silver-fix validation: en:lentil-pasta (singular) missed
    # by the original plural-only mapping.
    assert _type_b_multiclass(
        {"categories_tags": "en:lentil-pasta,en:dry-pastas"}, "grain_type"
    ) == "other"
    assert _type_b_multiclass(
        {"categories_tags": "en:konjac"}, "grain_type"
    ) == "other"
    assert _type_b_multiclass(
        {"categories_tags": "it:konjac"}, "grain_type"
    ) == "other"


def test_type_b_pasta_shape_stuffed_to_other():
    """Stuffed-pasta tags map to "other" per schema bucket."""
    assert _type_b_multiclass(
        {"categories_tags": "en:ravioli,en:fresh-pasta,en:stuffed-pastas"}, "pasta_shape"
    ) == "other"
    assert _type_b_multiclass(
        {"categories_tags": "en:tortellini,en:egg-pastas"}, "pasta_shape"
    ) == "other"
    assert _type_b_multiclass(
        {"categories_tags": "en:cappelletti"}, "pasta_shape"
    ) == "other"
    # Specialty short shapes
    assert _type_b_multiclass(
        {"categories_tags": "en:ditalini,en:pastas"}, "pasta_shape"
    ) == "other"
    assert _type_b_multiclass(
        {"categories_tags": "en:gnocchi"}, "pasta_shape"
    ) == "other"


def test_type_b_grain_legume_beats_wheat_fallback():
    """Lentil pasta with en:fresh-pasta tag should NOT fall back to wheat."""
    # Before the iteration-order fix this returned "wheat" because en:fresh-pasta
    # matched first in tag order.
    row = {"categories_tags": "en:pastas,en:fresh-pasta,en:lentil-pastas"}
    assert _type_b_multiclass(row, "grain_type") == "other"


def test_type_b_grain_buckwheat():
    row = {"categories_tags": "en:pastas,en:buckwheat-pastas"}
    assert _type_b_multiclass(row, "grain_type") == "buckwheat"


# --- Type C tests ---

def test_type_c_sugar_zero():
    row = {"sugars_100g": 0.2}
    assert _type_c_numeric(row, "sugar_class") == "0"


def test_type_c_sugar_low():
    row = {"sugars_100g": 3.5}
    assert _type_c_numeric(row, "sugar_class") == "low"


def test_type_c_sugar_high():
    row = {"sugars_100g": 12.0}
    assert _type_c_numeric(row, "sugar_class") == "high"


def test_type_c_alcohol_zero():
    row = {"alcohol_100g": 0}
    assert _type_c_numeric(row, "alcohol_class") == "0"


def test_type_c_alcohol_med():
    row = {"alcohol_100g": 12}
    assert _type_c_numeric(row, "alcohol_class") == "med"


def test_type_c_cocoa_percentage_70():
    """cocoa_percentage from product_name regex.

    Trek E [X, Y) convention: 70% → "70-85" (matches industry usage).
    """
    row = {"product_name": "Lindt Excellence 70% Dark Chocolate"}
    assert _type_c_numeric(row, "cocoa_percentage") == "70-85"


def test_type_c_cocoa_85():
    row = {"product_name": "Dark 85% Cocoa"}
    assert _type_c_numeric(row, "cocoa_percentage") == "85+"


def test_type_c_missing_numeric():
    row = {}
    assert _type_c_numeric(row, "sugar_class") is None


# --- Type D tests ---

def test_type_d_nutri_score():
    row = {"nutriscore_grade": "a"}
    assert _type_d_direct(row, "nutri_score_grade") == "A"


def test_type_d_nutri_score_capital():
    """В OFF может быть и нижний, и верхний регистр"""
    row = {"nutriscore_grade": "C"}
    assert _type_d_direct(row, "nutri_score_grade") == "C"


def test_type_d_nova_group():
    row = {"nova_group": 4}
    assert _type_d_direct(row, "nova_group") == 4


def test_type_d_palm_oil_free():
    row = {"ingredients_analysis_tags": "en:palm-oil-free,en:vegan"}
    assert _type_d_direct(row, "palm_oil_status") == "palm-oil-free"


def test_type_d_palm_oil_contains():
    row = {"ingredients_analysis_tags": "en:contains-palm-oil"}
    assert _type_d_direct(row, "palm_oil_status") == "contains"


# --- apply_off_labels orchestrator tests ---

def test_apply_off_labels_full_pasta_row():
    """Полный pasta row → большая часть атрибутов выводится из OFF"""
    row = {
        "product_name": "Barilla Spaghetti n.5",
        "brands": "Barilla",
        "labels_tags": "en:organic,en:eu-organic",
        "categories_tags": "en:pastas,en:dry-pastas,en:spaghetti,en:wheat-pastas",
        "countries_tags": "en:italy",
        "nutriscore_grade": "b",
        "sugars_100g": 2.5,
    }
    out = apply_off_labels(row, PASTA_SCHEMA)
    assert out["pasta_shape"] == "spaghetti"
    assert out["grain_type"] == "wheat"
    assert out["is_organic"] is True
    # is_gluten_free is in TYPE_A rules, labels_tags present but no gluten-free tag → False
    assert out["is_gluten_free"] is False
    # is_whole_grain not detected (no whole-grain tag in labels or categories) → not in output (skipped for LLM)


def test_apply_off_labels_minimal_row():
    """Только product_name + brands → почти все атрибуты None"""
    row = {"product_name": "Mystery food", "brands": "Generic"}
    out = apply_off_labels(row, PASTA_SCHEMA)
    # Атрибуты у которых нужны OFF поля → None (не присутствуют в out)
    assert out.get("pasta_shape") is None
    assert out.get("grain_type") is None
    # is_organic — labels_tags отсутствует → None
    assert out.get("is_organic") is None


def test_apply_off_labels_returns_only_schema_attrs():
    """apply_off_labels не возвращает атрибуты не из schema"""
    row = {"product_name": "x", "labels_tags": "en:organic"}
    out = apply_off_labels(row, PASTA_SCHEMA)
    for k in out.keys():
        assert k in PASTA_SCHEMA, f"Unexpected attr {k}"
