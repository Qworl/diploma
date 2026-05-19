"""Integration test: CascadePipeline.predict() returns the new response shape."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo", "ml_service"))


@pytest.fixture(scope="module")
def pipeline():
    from cascade import CascadePipeline
    return CascadePipeline(lazy_embedder=False)


def test_predict_shape_no_expected(pipeline):
    out = pipeline.predict(
        public_category="chocolate",
        product={
            "product_name": "Lindt Excellence Dark 70%",
            "brands": "Lindt",
            "ingredients_text": "Cocoa mass, sugar, cocoa butter",
            "quantity": "100g",
        },
        validate_mode="warn",
    )
    assert out["category"] == "chocolate"
    assert "predictions" in out and "expected" in out
    assert "validation_summary" in out
    assert "brand_status" in out["validation_summary"]
    # Every prediction either has a validation dict or None.
    for attr, blk in out["predictions"].items():
        assert "validation" in blk


def test_predict_white_chocolate_trap(pipeline):
    """Expected cocoa_percentage=70 on a white-chocolate product is flagged."""
    out = pipeline.predict(
        public_category="chocolate",
        product={
            "product_name": "Lindt Excellence White",
            "brands": "Lindt",
            "ingredients_text": "Sugar, cocoa butter, milk powder",
            "quantity": "100g",
        },
        validate_mode="warn",
        expected={"cocoa_percentage": 70},
    )
    assert "cocoa_percentage" in out["expected"]
    v = out["expected"]["cocoa_percentage"]
    assert v["validation"] is not None
    assert v["validation"]["flagged"] is True
    assert v["bucketized_to"] in {"70-85", "70+", "70-100"}


def test_predict_regex_layer_not_validated(pipeline):
    """Regex предсказания не валидируются — слой детерминированный.

    На «Lindt Excellence White» regex даёт chocolate_type=white (слово в
    названии). Раньше валидатор флаговал это как rare-given-evidence
    (P≈0.034). После фикса предсказания с layer='regex' не валидируются,
    validation = None.
    """
    out = pipeline.predict(
        public_category="chocolate",
        product={
            "product_name": "Lindt Excellence White",
            "brands": "Lindt",
            "ingredients_text": "Sugar, cocoa butter, milk powder",
            "quantity": "100g",
        },
        validate_mode="warn",
    )
    blk = out["predictions"].get("chocolate_type")
    assert blk is not None
    assert blk["value"] == "white"
    assert blk["layer"] == "regex"
    assert blk["validation"] is None


def test_predict_demote_mode_zeros_flagged(pipeline):
    out = pipeline.predict(
        public_category="chocolate",
        product={
            "product_name": "Lindt Excellence White",
            "brands": "Lindt",
            "ingredients_text": "Sugar, cocoa butter, milk powder",
            "quantity": "100g",
        },
        validate_mode="demote",
    )
    # In demote mode, any flagged prediction has value=None and a special layer.
    for attr, blk in out["predictions"].items():
        v = blk.get("validation")
        if v and v.get("flagged"):
            assert blk["value"] is None
            assert blk["layer"] == "rejected_by_validator"
