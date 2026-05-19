"""Tests for src.manual_label.llm_audit.apply_llm_decisions."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.manual_label.llm_audit import apply_llm_decisions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal fieldnames covering grain_type + pasta_shape
_FIELDNAMES = [
    "code", "source",
    "silver_grain_type", "manual_grain_type",
    "manual_grain_type_status", "manual_grain_type_at",
    "manual_grain_type_mode", "manual_grain_type_note",
    "silver_pasta_shape", "manual_pasta_shape",
    "manual_pasta_shape_status", "manual_pasta_shape_at",
    "manual_pasta_shape_mode", "manual_pasta_shape_note",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            # Fill missing keys with empty string
            full = {k: "" for k in _FIELDNAMES}
            full.update(row)
            writer.writerow(full)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_decisions(path: Path, decisions: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(decisions, f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_auto_cell_confirmed(tmp_path):
    """auto cell with silver==value → status becomes confirmed, mode=llm."""
    csv_path = tmp_path / "pasta_gold_250.csv"
    _write_csv(csv_path, [
        {"code": "111", "silver_grain_type": "wheat",
         "manual_grain_type": "wheat", "manual_grain_type_status": "auto",
         "manual_grain_type_mode": "prefill"},
    ])
    decisions_path = tmp_path / "decisions.json"
    _write_decisions(decisions_path, {
        "111": {"grain_type": {"value": "wheat", "status_hint": None, "reasoning": ""}}
    })
    apply_llm_decisions(csv_path, decisions_path, attrs=["grain_type"])
    rows = _read_csv(csv_path)
    assert rows[0]["manual_grain_type"] == "wheat"
    assert rows[0]["manual_grain_type_status"] == "confirmed"
    assert rows[0]["manual_grain_type_mode"] == "llm"
    assert rows[0]["manual_grain_type_at"] != ""


def test_auto_cell_overridden(tmp_path):
    """auto cell with silver!=value → status becomes override, mode=llm."""
    csv_path = tmp_path / "pasta_gold_250.csv"
    _write_csv(csv_path, [
        {"code": "222", "silver_grain_type": "wheat",
         "manual_grain_type": "wheat", "manual_grain_type_status": "auto",
         "manual_grain_type_mode": "prefill"},
    ])
    decisions_path = tmp_path / "decisions.json"
    _write_decisions(decisions_path, {
        "222": {"grain_type": {"value": "rice", "status_hint": None, "reasoning": ""}}
    })
    apply_llm_decisions(csv_path, decisions_path, attrs=["grain_type"])
    rows = _read_csv(csv_path)
    assert rows[0]["manual_grain_type"] == "rice"
    assert rows[0]["manual_grain_type_status"] == "override"
    assert rows[0]["manual_grain_type_mode"] == "llm"


def test_empty_cell_filled(tmp_path):
    """empty cell (silver also empty) with a value → status becomes manual_only."""
    csv_path = tmp_path / "pasta_gold_250.csv"
    _write_csv(csv_path, [
        {"code": "333", "silver_pasta_shape": "",
         "manual_pasta_shape": "", "manual_pasta_shape_status": "empty",
         "manual_pasta_shape_mode": ""},
    ])
    decisions_path = tmp_path / "decisions.json"
    _write_decisions(decisions_path, {
        "333": {"pasta_shape": {"value": "fusilli", "status_hint": None, "reasoning": ""}}
    })
    apply_llm_decisions(csv_path, decisions_path, attrs=["pasta_shape"])
    rows = _read_csv(csv_path)
    assert rows[0]["manual_pasta_shape"] == "fusilli"
    assert rows[0]["manual_pasta_shape_status"] == "manual_only"
    assert rows[0]["manual_pasta_shape_mode"] == "llm"


def test_human_audited_cell_protected(tmp_path):
    """Human-confirmed cell (mode=blind) must NOT be overwritten."""
    csv_path = tmp_path / "pasta_gold_250.csv"
    _write_csv(csv_path, [
        {"code": "444", "silver_grain_type": "wheat",
         "manual_grain_type": "wheat", "manual_grain_type_status": "confirmed",
         "manual_grain_type_mode": "blind"},
    ])
    decisions_path = tmp_path / "decisions.json"
    _write_decisions(decisions_path, {
        "444": {"grain_type": {"value": "rice", "status_hint": None, "reasoning": ""}}
    })
    result = apply_llm_decisions(csv_path, decisions_path, attrs=["grain_type"])
    rows = _read_csv(csv_path)
    # Cell must be unchanged
    assert rows[0]["manual_grain_type"] == "wheat"
    assert rows[0]["manual_grain_type_status"] == "confirmed"
    assert rows[0]["manual_grain_type_mode"] == "blind"
    # Summary must record human_protected
    assert any(k[0] == "human_protected" for k in result["summary"])


def test_unsure_hint(tmp_path):
    """status_hint=unsure forces status=unsure regardless of value."""
    csv_path = tmp_path / "pasta_gold_250.csv"
    _write_csv(csv_path, [
        {"code": "555", "silver_grain_type": "wheat",
         "manual_grain_type": "wheat", "manual_grain_type_status": "auto",
         "manual_grain_type_mode": "prefill"},
    ])
    decisions_path = tmp_path / "decisions.json"
    _write_decisions(decisions_path, {
        "555": {"grain_type": {"value": "wheat", "status_hint": "unsure", "reasoning": ""}}
    })
    apply_llm_decisions(csv_path, decisions_path, attrs=["grain_type"])
    rows = _read_csv(csv_path)
    assert rows[0]["manual_grain_type_status"] == "unsure"
    assert rows[0]["manual_grain_type_mode"] == "llm"


def test_null_value_skipped(tmp_path):
    """value=null → cell is skipped; summary records llm_skipped."""
    csv_path = tmp_path / "pasta_gold_250.csv"
    _write_csv(csv_path, [
        {"code": "666", "silver_grain_type": "wheat",
         "manual_grain_type": "wheat", "manual_grain_type_status": "auto",
         "manual_grain_type_mode": "prefill"},
    ])
    decisions_path = tmp_path / "decisions.json"
    _write_decisions(decisions_path, {
        "666": {"grain_type": {"value": None, "status_hint": None, "reasoning": ""}}
    })
    result = apply_llm_decisions(csv_path, decisions_path, attrs=["grain_type"])
    rows = _read_csv(csv_path)
    # Cell unchanged
    assert rows[0]["manual_grain_type"] == "wheat"
    assert rows[0]["manual_grain_type_status"] == "auto"
    assert rows[0]["manual_grain_type_mode"] == "prefill"
    assert any(k[0] == "llm_skipped" for k in result["summary"])


def test_invalid_value_skipped(tmp_path):
    """Value not in schema → cell is skipped; summary records invalid_value."""
    csv_path = tmp_path / "pasta_gold_250.csv"
    _write_csv(csv_path, [
        {"code": "777", "silver_pasta_shape": "spaghetti",
         "manual_pasta_shape": "spaghetti", "manual_pasta_shape_status": "auto",
         "manual_pasta_shape_mode": "prefill"},
    ])
    decisions_path = tmp_path / "decisions.json"
    _write_decisions(decisions_path, {
        # "rotini" is not in pasta_shape schema values
        "777": {"pasta_shape": {"value": "rotini", "status_hint": None, "reasoning": ""}}
    })
    result = apply_llm_decisions(csv_path, decisions_path, attrs=["pasta_shape"])
    rows = _read_csv(csv_path)
    # Cell unchanged
    assert rows[0]["manual_pasta_shape"] == "spaghetti"
    assert rows[0]["manual_pasta_shape_status"] == "auto"
    assert any(k[0] == "invalid_value" for k in result["summary"])


def test_atomic_backup(tmp_path):
    """After apply, a .bak file exists with the original content."""
    csv_path = tmp_path / "pasta_gold_250.csv"
    _write_csv(csv_path, [
        {"code": "888", "silver_grain_type": "wheat",
         "manual_grain_type": "wheat", "manual_grain_type_status": "auto",
         "manual_grain_type_mode": "prefill"},
    ])
    original_content = csv_path.read_text(encoding="utf-8")
    decisions_path = tmp_path / "decisions.json"
    _write_decisions(decisions_path, {
        "888": {"grain_type": {"value": "wheat", "status_hint": None, "reasoning": ""}}
    })
    apply_llm_decisions(csv_path, decisions_path, attrs=["grain_type"])
    bak_path = csv_path.with_suffix(csv_path.suffix + ".bak")
    assert bak_path.exists(), ".bak file was not created"
    assert bak_path.read_text(encoding="utf-8") == original_content


def test_chocolate_domain_validates_against_chocolate_schema(tmp_path):
    """With domain=chocolate, a valid chocolate_type value is accepted,
    and an invalid one (e.g. a pasta value) is rejected."""
    fieldnames = [
        "code", "source",
        "silver_chocolate_type", "manual_chocolate_type",
        "manual_chocolate_type_status", "manual_chocolate_type_at",
        "manual_chocolate_type_mode", "manual_chocolate_type_note",
    ]
    csv_path = tmp_path / "chocolate_gold.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Row 1: valid chocolate enum value
        writer.writerow({k: "" for k in fieldnames} | {
            "code": "C1", "silver_chocolate_type": "dark",
            "manual_chocolate_type": "dark",
            "manual_chocolate_type_status": "auto",
            "manual_chocolate_type_mode": "prefill",
        })
        # Row 2: same, will receive invalid value
        writer.writerow({k: "" for k in fieldnames} | {
            "code": "C2", "silver_chocolate_type": "dark",
            "manual_chocolate_type": "dark",
            "manual_chocolate_type_status": "auto",
            "manual_chocolate_type_mode": "prefill",
        })

    decisions_path = tmp_path / "decisions.json"
    _write_decisions(decisions_path, {
        "C1": {"chocolate_type": {"value": "milk", "status_hint": None, "reasoning": ""}},
        # "spaghetti" is a pasta_shape value, not a chocolate_type value
        "C2": {"chocolate_type": {"value": "spaghetti", "status_hint": None, "reasoning": ""}},
    })
    result = apply_llm_decisions(
        csv_path, decisions_path, attrs=["chocolate_type"], domain="chocolate",
    )
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    # C1 accepted
    assert rows[0]["manual_chocolate_type"] == "milk"
    assert rows[0]["manual_chocolate_type_status"] == "override"
    assert rows[0]["manual_chocolate_type_mode"] == "llm"
    # C2 rejected (invalid value) — cell unchanged
    assert rows[1]["manual_chocolate_type"] == "dark"
    assert rows[1]["manual_chocolate_type_status"] == "auto"
    assert any(k[0] == "invalid_value" for k in result["summary"])


def test_at_timestamp_set(tmp_path):
    """After apply on an auto cell, _at is a non-empty ISO timestamp."""
    csv_path = tmp_path / "pasta_gold_250.csv"
    _write_csv(csv_path, [
        {"code": "999", "silver_grain_type": "wheat",
         "manual_grain_type": "wheat", "manual_grain_type_status": "auto",
         "manual_grain_type_mode": "prefill",
         "manual_grain_type_at": ""},
    ])
    decisions_path = tmp_path / "decisions.json"
    _write_decisions(decisions_path, {
        "999": {"grain_type": {"value": "wheat", "status_hint": None, "reasoning": ""}}
    })
    apply_llm_decisions(csv_path, decisions_path, attrs=["grain_type"])
    rows = _read_csv(csv_path)
    ts = rows[0]["manual_grain_type_at"]
    assert ts != "", "_at should be set after llm apply"
    # Should be parseable as ISO datetime
    import datetime
    parsed = datetime.datetime.fromisoformat(ts)
    assert parsed is not None
