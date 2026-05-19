"""Tests for src.manual_label.opus_audit_caller (Trek E).

The real OpenRouter call is mocked. Goals:
  * audit_product turns a mocked JSON response into the decisions schema.
  * run_audit is resume-safe (skips rows already in decisions JSON).
  * run_audit honours --max-cost-usd by stopping early.
  * Boolean values are coerced "True"/"False" to match CSV format.
  * Out-of-schema values become null with a reasoning note.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.manual_label import opus_audit_caller
from src.manual_label.opus_audit_caller import audit_product, run_audit


def _write_chocolate_csv(path: Path, codes: list[str]) -> None:
    attrs = ["chocolate_type", "cocoa_percentage", "contains_nuts",
             "chocolate_extra", "is_organic", "nutri_score_grade", "protein_class"]
    fieldnames = ["code", "product_name", "brands", "ingredients_text", "quantity",
                  "categories_tags", "source"]
    for a in attrs:
        fieldnames += [f"silver_{a}", f"manual_{a}",
                       f"manual_{a}_status", f"manual_{a}_at",
                       f"manual_{a}_mode", f"manual_{a}_note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in codes:
            row = {k: "" for k in fieldnames}
            row["code"] = c
            row["product_name"] = f"Chocolate {c}"
            row["brands"] = "Brand"
            row["ingredients_text"] = "cocoa, sugar"
            row["quantity"] = "100g"
            for a in attrs:
                row[f"manual_{a}_status"] = "empty"
            writer.writerow(row)


def test_audit_product_parses_chocolate_response():
    response_json = json.dumps({
        "chocolate_type": "dark",
        "cocoa_percentage": "70-85",
        "contains_nuts": False,
        "chocolate_extra": "plain",
        "is_organic": True,
        "nutri_score_grade": "E",
        "protein_class": "med",
    })

    def fake_call(**kwargs):
        return response_json

    product = {"product_name": "Lindt 70%", "brands": "Lindt",
               "ingredients_text": "cocoa, sugar", "quantity": "100g"}
    decision, usage = audit_product(
        product, domain="chocolate", api_key="fake",
        model="anthropic/claude-opus-4", call_fn=fake_call,
    )
    assert decision["chocolate_type"]["value"] == "dark"
    assert decision["cocoa_percentage"]["value"] == "70-85"
    assert decision["contains_nuts"]["value"] == "False"
    assert decision["is_organic"]["value"] == "True"
    assert decision["protein_class"]["value"] == "med"
    assert usage["in_tokens"] > 0
    assert usage["out_tokens"] > 0


def test_audit_product_skips_invalid_enum_value():
    response_json = json.dumps({
        "chocolate_type": "rotini",   # invalid for chocolate
        "cocoa_percentage": "70-85",
        "contains_nuts": False,
        "chocolate_extra": "plain",
        "is_organic": False,
        "nutri_score_grade": "C",
        "protein_class": "low",
    })

    def fake_call(**kwargs):
        return response_json

    product = {"product_name": "X", "brands": "Y",
               "ingredients_text": "cocoa", "quantity": "100g"}
    decision, _ = audit_product(
        product, domain="chocolate", api_key="fake", call_fn=fake_call,
    )
    # parse_llm_response drops invalid enum values before _coerce_for_csv sees them,
    # so chocolate_type ends up as None and the rest are filled.
    assert decision["chocolate_type"]["value"] is None
    assert decision["cocoa_percentage"]["value"] == "70-85"
    assert decision["nutri_score_grade"]["value"] == "C"


def test_run_audit_is_resume_safe(tmp_path):
    csv_path = tmp_path / "choc.csv"
    out_path = tmp_path / "decisions.json"
    _write_chocolate_csv(csv_path, ["C001", "C002", "C003"])

    # Pre-populate one decision so run_audit should skip C001.
    out_path.write_text(json.dumps({
        "C001": {"chocolate_type": {"value": "milk", "status_hint": None, "reasoning": None}},
    }))

    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return json.dumps({
            "chocolate_type": "dark",
            "cocoa_percentage": "70-85",
            "contains_nuts": False,
            "chocolate_extra": "plain",
            "is_organic": False,
            "nutri_score_grade": "D",
            "protein_class": "low",
        })

    result = run_audit(
        csv_path, domain="chocolate", out_path=out_path,
        api_key="fake", call_fn=fake_call, progress_every=1,
    )
    assert result["rows_called"] == 2  # C002, C003 — C001 already present
    assert result["rows_skipped_already_audited"] == 1
    # decisions JSON includes all three
    decisions = json.loads(out_path.read_text())
    assert set(decisions.keys()) == {"C001", "C002", "C003"}
    # C001 preserved (not overwritten)
    assert decisions["C001"]["chocolate_type"]["value"] == "milk"


def test_run_audit_limit_truncates(tmp_path):
    csv_path = tmp_path / "choc.csv"
    out_path = tmp_path / "decisions.json"
    _write_chocolate_csv(csv_path, [f"C{i:03d}" for i in range(10)])

    def fake_call(**kwargs):
        return json.dumps({
            "chocolate_type": "dark",
            "cocoa_percentage": "70-85",
            "contains_nuts": False,
            "chocolate_extra": "plain",
            "is_organic": False,
            "nutri_score_grade": "D",
            "protein_class": "low",
        })

    result = run_audit(
        csv_path, domain="chocolate", out_path=out_path,
        api_key="fake", call_fn=fake_call, progress_every=10, limit=3,
    )
    assert result["rows_called"] == 3


def test_run_audit_cost_cap_stops_early(tmp_path, monkeypatch):
    csv_path = tmp_path / "choc.csv"
    out_path = tmp_path / "decisions.json"
    _write_chocolate_csv(csv_path, [f"C{i:03d}" for i in range(20)])

    # Set unrealistically high prices so even one row blows the cap.
    monkeypatch.setattr(opus_audit_caller, "_PRICE_INPUT_PER_MTOK", 1_000_000.0)
    monkeypatch.setattr(opus_audit_caller, "_PRICE_OUTPUT_PER_MTOK", 1_000_000.0)

    def fake_call(**kwargs):
        return json.dumps({"chocolate_type": "dark"})

    result = run_audit(
        csv_path, domain="chocolate", out_path=out_path,
        api_key="fake", call_fn=fake_call, max_cost_usd=1.0,
        progress_every=1,
    )
    # Should stop after the first row since cost cap is hit.
    assert result["rows_called"] == 1


def test_audit_product_unknown_domain_raises():
    with pytest.raises(KeyError):
        audit_product({"product_name": "x"}, domain="widgets", api_key="fake")
