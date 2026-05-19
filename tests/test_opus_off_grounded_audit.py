"""Tests for src.manual_label.opus_off_grounded_audit."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.manual_label.opus_off_grounded_audit import (
    audit_product,
    run_audit,
)


_OFF_PRODUCT = {
    "code": "8000139007057",
    "product_name": "Penne ziti lisce",
    "brands": "Garofalo",
    "quantity": "500 g",
    "ingredients_text": "Semola di GRANO duro.",
    "categories_tags": ["en:pastas", "en:dry-pasta"],
    "labels_tags": [],
    "allergens_tags": ["en:gluten"],
    "nutriments": {
        "energy_100g": 1500, "fat_100g": 1.5, "sugars_100g": 3.5,
        "salt_100g": 0.01, "proteins_100g": 13.0, "fiber_100g": 3.0,
        "carbohydrates_100g": 70.0,
    },
    "packaging_tags": ["en:cardboard"],
}


def _fake_call_returning(payload: dict):
    def _call(messages, model, api_key, **kwargs):
        return json.dumps(payload)
    return _call


def _fake_fetch(code, *, cache_dir, **kwargs):
    return _OFF_PRODUCT


def test_audit_product_returns_decisions(tmp_path):
    api_response = {
        "pasta_shape": "penne",
        "grain_type": "wheat",
        "is_filled": False,
        "is_organic": False,
        "is_gluten_free": False,
        "is_vegan": True,
        "nutri_score_grade": "A",
        "protein_class": "med",
    }
    decision, usage = audit_product(
        code="8000139007057",
        domain="pasta",
        cache_dir=tmp_path,
        api_key="test-key",
        fetch_fn=_fake_fetch,
        call_fn=_fake_call_returning(api_response),
    )
    assert decision["pasta_shape"]["value"] == "penne"
    assert decision["grain_type"]["value"] == "wheat"
    assert decision["is_vegan"]["value"] == "True"
    assert usage["in_tokens"] > 0


def test_audit_product_handles_null_response(tmp_path):
    api_response = {
        "pasta_shape": None,
        "grain_type": "wheat",
    }
    decision, usage = audit_product(
        code="8000139007057",
        domain="pasta",
        cache_dir=tmp_path,
        api_key="test-key",
        fetch_fn=_fake_fetch,
        call_fn=_fake_call_returning(api_response),
    )
    assert decision["pasta_shape"]["value"] is None  # refusal


def test_audit_product_filters_derived_from_prompt(tmp_path):
    """Derived OFF classifications must NOT appear in the prompt sent to Opus."""
    captured = {}
    def _spy_call(messages, model, api_key, **kwargs):
        captured["messages"] = messages
        return json.dumps({"pasta_shape": "penne"})

    off_with_derived = dict(_OFF_PRODUCT)
    off_with_derived["nutriscore_grade"] = "a"  # would leak nutri_score_grade
    off_with_derived["ingredients_analysis_tags"] = ["en:vegan"]  # would leak is_vegan

    def _fetch(code, *, cache_dir, **kwargs):
        return off_with_derived

    audit_product(
        code="8000139007057",
        domain="pasta",
        cache_dir=tmp_path,
        api_key="test-key",
        fetch_fn=_fetch,
        call_fn=_spy_call,
    )
    prompt_text = captured["messages"][0]["content"]
    assert "nutriscore_grade" not in prompt_text
    assert "ingredients_analysis_tags" not in prompt_text


def test_run_audit_resume_safe(tmp_path):
    """Re-running with same output skips already-audited codes."""
    csv_path = tmp_path / "products.csv"
    out_path = tmp_path / "decisions.json"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code"])
        w.writeheader()
        w.writerow({"code": "8000139007057"})
        w.writerow({"code": "8000139007058"})

    api_response = {"pasta_shape": "penne"}
    call_count = {"n": 0}
    def _counting_call(messages, model, api_key, **kwargs):
        call_count["n"] += 1
        return json.dumps(api_response)

    # First run audits both
    run_audit(
        csv_path=csv_path,
        domain="pasta",
        out_path=out_path,
        cache_dir=tmp_path,
        api_key="test-key",
        fetch_fn=_fake_fetch,
        call_fn=_counting_call,
    )
    assert call_count["n"] == 2

    # Second run: nothing new
    run_audit(
        csv_path=csv_path,
        domain="pasta",
        out_path=out_path,
        cache_dir=tmp_path,
        api_key="test-key",
        fetch_fn=_fake_fetch,
        call_fn=_counting_call,
    )
    assert call_count["n"] == 2  # no extra calls


def test_run_audit_writes_output(tmp_path):
    csv_path = tmp_path / "products.csv"
    out_path = tmp_path / "decisions.json"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code"])
        w.writeheader()
        w.writerow({"code": "8000139007057"})

    api_response = {"pasta_shape": "penne", "grain_type": "wheat"}
    run_audit(
        csv_path=csv_path,
        domain="pasta",
        out_path=out_path,
        cache_dir=tmp_path,
        api_key="test-key",
        fetch_fn=_fake_fetch,
        call_fn=_fake_call_returning(api_response),
    )
    saved = json.loads(out_path.read_text())
    assert "8000139007057" in saved
    assert saved["8000139007057"]["pasta_shape"]["value"] == "penne"


def test_run_audit_cost_cap(tmp_path):
    """Cost cap hit → stop iterating, save what was done."""
    csv_path = tmp_path / "products.csv"
    out_path = tmp_path / "decisions.json"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code"])
        w.writeheader()
        for i in range(10):
            w.writerow({"code": f"8000139007{i:03d}"})

    # Each call returns a giant string to blow the budget fast.
    big_payload = json.dumps({"pasta_shape": "p" * 100_000})
    def _huge_call(messages, model, api_key, **kwargs):
        return big_payload

    summary = run_audit(
        csv_path=csv_path,
        domain="pasta",
        out_path=out_path,
        cache_dir=tmp_path,
        api_key="test-key",
        max_cost_usd=0.10,  # tiny budget
        fetch_fn=_fake_fetch,
        call_fn=_huge_call,
    )
    assert summary["rows_called"] < 10  # stopped early
    assert summary["estimated_cost_usd"] >= 0.10
