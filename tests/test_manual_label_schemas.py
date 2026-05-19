"""Tests for src.manual_label.schemas_loader."""
import pytest

from src.manual_label.schemas_loader import load_pasta_attrs, load_domain_attrs


def test_pasta_returns_all_eight():
    attrs = load_pasta_attrs()
    assert set(attrs.keys()) == {
        "grain_type", "pasta_shape", "is_filled",
        "is_organic", "is_gluten_free", "is_vegan",
        "nutri_score_grade", "protein_class",
    }


def test_load_domain_attrs_pasta_matches_legacy_loader():
    assert load_domain_attrs("pasta") == load_pasta_attrs()


def test_load_domain_attrs_chocolate():
    attrs = load_domain_attrs("chocolate")
    assert "chocolate_type" in attrs
    assert attrs["chocolate_type"]["type"] == "enum"
    assert attrs["chocolate_type"]["values"] == ["dark", "milk", "white", "filled", "other"]
    assert attrs["contains_nuts"]["type"] == "bool"
    assert attrs["contains_nuts"]["values"] == ["True", "False"]
    assert attrs["cocoa_percentage"]["nullable"] is True


def test_load_domain_attrs_cheeses():
    attrs = load_domain_attrs("cheeses")
    assert "milk_source" in attrs
    assert attrs["milk_source"]["values"] == ["cow", "goat", "sheep", "buffalo", "mixed", "other"]
    assert attrs["is_pdo"]["type"] == "bool"
    assert attrs["is_pdo"]["values"] == ["True", "False"]


def test_load_domain_attrs_unknown_raises():
    with pytest.raises(KeyError, match="Unknown domain"):
        load_domain_attrs("widgets")


def test_grain_type_enum_values():
    attrs = load_pasta_attrs()
    g = attrs["grain_type"]
    assert g["type"] == "enum"
    # spelt added post-Trek-D pivot (Dinkel pasta detection from Opus audit)
    assert g["values"] == ["wheat", "spelt", "rice", "corn", "buckwheat", "oat", "mixed", "other"]
    assert g.get("nullable") is False


def test_pasta_shape_is_nullable():
    attrs = load_pasta_attrs()
    assert attrs["pasta_shape"]["nullable"] is True


def test_bool_attrs_have_canonical_values():
    attrs = load_pasta_attrs()
    for name in ("is_filled", "is_organic", "is_gluten_free", "is_vegan"):
        a = attrs[name]
        assert a["type"] == "bool"
        assert a["values"] == ["True", "False"]
