"""Tests for src.manual_label.off_field_filter."""
from __future__ import annotations

import pytest

from src.manual_label.off_field_filter import (
    DERIVED_BLACKLIST,
    curate_prompt_fields,
)


_RAW_OFF = {
    "code": "8000139007057",
    "product_name": "Penne ziti lisce",
    "product_name_en": "Penne ziti smooth",
    "product_name_it": "Penne ziti lisce",
    "brands": "Garofalo",
    "quantity": "500 g",
    "ingredients_text": "Semola di GRANO duro.",
    "ingredients_text_en": "Durum wheat semolina.",
    "ingredients_text_it": "Semola di grano duro.",
    "categories_tags": ["en:cereals-and-potatoes", "en:pasta", "en:dry-pasta"],
    "labels_tags": ["en:no-gluten-claim"],
    "allergens_tags": ["en:gluten"],
    "packaging_tags": ["en:cardboard"],
    "image_front_url": "https://images.openfoodfacts.org/.../front.jpg",
    "image_ingredients_url": "https://images.openfoodfacts.org/.../ing.jpg",
    "nutriments": {
        "energy_100g": 1500,
        "fat_100g": 1.5,
        "sugars_100g": 3.5,
        "salt_100g": 0.01,
        "proteins_100g": 13.0,
        "fiber_100g": 3.0,
        "carbohydrates_100g": 70.0,
        # The fields below are OFF-derived classifications (target answers).
        # MUST be filtered.
        "nutriscore_grade": "a",
        "nutriscore_score": -2,
        "nova_group": 1,
    },
    # Top-level derived classifications also must be filtered.
    "nutriscore_grade": "a",
    "nova_group": 1,
    "nova_groups_tags": ["en:1-unprocessed-or-minimally-processed-foods"],
    "ecoscore_grade": "b",
    "ingredients_analysis_tags": [
        "en:vegan",
        "en:vegetarian",
        "en:palm-oil-free",
    ],
}


def test_blacklist_contains_known_derived():
    for k in [
        "nutriscore_grade", "nutriscore_score", "nova_group",
        "nova_groups_tags", "ecoscore_grade", "ingredients_analysis_tags",
    ]:
        assert k in DERIVED_BLACKLIST


def test_curate_drops_top_level_derived():
    out = curate_prompt_fields(_RAW_OFF)
    for k in DERIVED_BLACKLIST:
        assert k not in out, f"{k} should be filtered"


def test_curate_drops_nutriment_derived_keeps_numeric():
    out = curate_prompt_fields(_RAW_OFF)
    nut = out["nutriments"]
    # Numeric per-100g kept
    assert nut["energy_100g"] == 1500
    assert nut["proteins_100g"] == 13.0
    # Derived classifications dropped
    assert "nutriscore_grade" not in nut
    assert "nutriscore_score" not in nut
    assert "nova_group" not in nut


def test_curate_keeps_partner_fields():
    out = curate_prompt_fields(_RAW_OFF)
    for k in ["product_name", "brands", "quantity"]:
        assert k in out


def test_curate_aggregates_multilang_ingredients():
    out = curate_prompt_fields(_RAW_OFF)
    # All ingredients_text_* keys preserved
    assert "ingredients_text" in out
    assert out.get("ingredients_text_en") == "Durum wheat semolina."
    assert out.get("ingredients_text_it") == "Semola di grano duro."


def test_curate_keeps_tags_and_packaging():
    out = curate_prompt_fields(_RAW_OFF)
    assert out["categories_tags"] == sorted(_RAW_OFF["categories_tags"])
    assert out["labels_tags"] == _RAW_OFF["labels_tags"]
    assert out["allergens_tags"] == _RAW_OFF["allergens_tags"]
    assert out["packaging_tags"] == _RAW_OFF["packaging_tags"]


def test_curate_keeps_image_urls():
    out = curate_prompt_fields(_RAW_OFF)
    assert "image_front_url" in out
    assert "image_ingredients_url" in out


def test_curate_handles_missing_nutriments():
    raw = {"product_name": "X", "brands": "Y"}
    out = curate_prompt_fields(raw)
    assert "nutriments" not in out  # absent rather than empty


def test_curate_does_not_mutate_input():
    import copy
    snapshot = copy.deepcopy(_RAW_OFF)
    curate_prompt_fields(_RAW_OFF)
    assert _RAW_OFF == snapshot
