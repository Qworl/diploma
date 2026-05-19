"""Migration: confident->confirmed/override, empty+silver->auto+prefill, blind tagging."""
import csv
from pathlib import Path

import pytest

from src.manual_label.migrate_to_prefill import migrate_csv


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return r.fieldnames or [], list(r)


def _base_fields() -> list[str]:
    return [
        "code", "source",
        "silver_grain_type", "manual_grain_type",
        "manual_grain_type_status", "manual_grain_type_at", "manual_grain_type_note",
        "silver_pasta_shape", "manual_pasta_shape",
        "manual_pasta_shape_status", "manual_pasta_shape_at", "manual_pasta_shape_note",
    ]


def test_confident_match_becomes_confirmed_blind(tmp_path):
    path = tmp_path / "g.csv"
    _write_csv(path, _base_fields(), [{
        "code": "c1", "source": "brand_disjoint_test",
        "silver_grain_type": "wheat", "manual_grain_type": "wheat",
        "manual_grain_type_status": "confident",
        "manual_grain_type_at": "2026-05-14T10:00:00+00:00",
        "manual_grain_type_note": "",
        "silver_pasta_shape": "spaghetti", "manual_pasta_shape": "",
        "manual_pasta_shape_status": "empty",
        "manual_pasta_shape_at": "", "manual_pasta_shape_note": "",
    }])
    migrate_csv(path, attrs=["grain_type", "pasta_shape"])
    fieldnames, rows = _read_csv(path)
    assert "manual_grain_type_mode" in fieldnames
    assert "manual_pasta_shape_mode" in fieldnames
    r = rows[0]
    assert r["manual_grain_type_status"] == "confirmed"
    assert r["manual_grain_type_mode"] == "blind"
    # empty + silver set -> pre-fill -> auto/prefill
    assert r["manual_pasta_shape"] == "spaghetti"
    assert r["manual_pasta_shape_status"] == "auto"
    assert r["manual_pasta_shape_mode"] == "prefill"


def test_confident_mismatch_becomes_override_blind(tmp_path):
    path = tmp_path / "g.csv"
    _write_csv(path, _base_fields(), [{
        "code": "c1", "source": "disagreement",
        "silver_grain_type": "wheat", "manual_grain_type": "rice",
        "manual_grain_type_status": "confident",
        "manual_grain_type_at": "2026-05-14T10:00:00+00:00",
        "manual_grain_type_note": "",
        "silver_pasta_shape": "", "manual_pasta_shape": "",
        "manual_pasta_shape_status": "empty",
        "manual_pasta_shape_at": "", "manual_pasta_shape_note": "",
    }])
    migrate_csv(path, attrs=["grain_type", "pasta_shape"])
    _, rows = _read_csv(path)
    r = rows[0]
    assert r["manual_grain_type_status"] == "override"
    assert r["manual_grain_type_mode"] == "blind"
    # silver empty + manual empty -> untouched
    assert r["manual_pasta_shape_status"] == "empty"
    assert r["manual_pasta_shape_mode"] == ""


def test_confident_silver_empty_becomes_manual_only_blind(tmp_path):
    path = tmp_path / "g.csv"
    _write_csv(path, _base_fields(), [{
        "code": "c1", "source": "disagreement",
        "silver_grain_type": "", "manual_grain_type": "rice",
        "manual_grain_type_status": "confident",
        "manual_grain_type_at": "2026-05-14T10:00:00+00:00",
        "manual_grain_type_note": "",
        "silver_pasta_shape": "penne", "manual_pasta_shape": "",
        "manual_pasta_shape_status": "empty",
        "manual_pasta_shape_at": "", "manual_pasta_shape_note": "",
    }])
    migrate_csv(path, attrs=["grain_type", "pasta_shape"])
    _, rows = _read_csv(path)
    r = rows[0]
    assert r["manual_grain_type_status"] == "manual_only"
    assert r["manual_grain_type_mode"] == "blind"


def test_unsure_status_preserved_with_blind_mode(tmp_path):
    path = tmp_path / "g.csv"
    _write_csv(path, _base_fields(), [{
        "code": "c1", "source": "brand_disjoint_test",
        "silver_grain_type": "wheat", "manual_grain_type": "",
        "manual_grain_type_status": "unsure",
        "manual_grain_type_at": "2026-05-14T10:00:00+00:00",
        "manual_grain_type_note": "",
        "silver_pasta_shape": "", "manual_pasta_shape": "",
        "manual_pasta_shape_status": "empty",
        "manual_pasta_shape_at": "", "manual_pasta_shape_note": "",
    }])
    migrate_csv(path, attrs=["grain_type", "pasta_shape"])
    _, rows = _read_csv(path)
    r = rows[0]
    assert r["manual_grain_type_status"] == "unsure"
    assert r["manual_grain_type_mode"] == "blind"


def test_at_not_modified_by_prefill(tmp_path):
    path = tmp_path / "g.csv"
    _write_csv(path, _base_fields(), [{
        "code": "c1", "source": "gold_tier_control",
        "silver_grain_type": "wheat", "manual_grain_type": "",
        "manual_grain_type_status": "empty",
        "manual_grain_type_at": "", "manual_grain_type_note": "",
        "silver_pasta_shape": "penne", "manual_pasta_shape": "",
        "manual_pasta_shape_status": "empty",
        "manual_pasta_shape_at": "", "manual_pasta_shape_note": "",
    }])
    migrate_csv(path, attrs=["grain_type", "pasta_shape"])
    _, rows = _read_csv(path)
    # auto pre-fill must NOT stamp a timestamp; that would corrupt pace metrics
    assert rows[0]["manual_grain_type_at"] == ""
    assert rows[0]["manual_pasta_shape_at"] == ""


def test_idempotent(tmp_path):
    path = tmp_path / "g.csv"
    _write_csv(path, _base_fields(), [{
        "code": "c1", "source": "brand_disjoint_test",
        "silver_grain_type": "wheat", "manual_grain_type": "wheat",
        "manual_grain_type_status": "confident",
        "manual_grain_type_at": "2026-05-14T10:00:00+00:00",
        "manual_grain_type_note": "",
        "silver_pasta_shape": "penne", "manual_pasta_shape": "",
        "manual_pasta_shape_status": "empty",
        "manual_pasta_shape_at": "", "manual_pasta_shape_note": "",
    }])
    migrate_csv(path, attrs=["grain_type", "pasta_shape"])
    _, rows1 = _read_csv(path)
    migrate_csv(path, attrs=["grain_type", "pasta_shape"])
    _, rows2 = _read_csv(path)
    assert rows1 == rows2, "migration must be idempotent"


def test_backup_created(tmp_path):
    path = tmp_path / "g.csv"
    _write_csv(path, _base_fields(), [{
        "code": "c1", "source": "brand_disjoint_test",
        "silver_grain_type": "wheat", "manual_grain_type": "wheat",
        "manual_grain_type_status": "confident",
        "manual_grain_type_at": "2026-05-14T10:00:00+00:00",
        "manual_grain_type_note": "",
        "silver_pasta_shape": "", "manual_pasta_shape": "",
        "manual_pasta_shape_status": "empty",
        "manual_pasta_shape_at": "", "manual_pasta_shape_note": "",
    }])
    migrate_csv(path, attrs=["grain_type", "pasta_shape"])
    assert path.with_suffix(".csv.bak").exists()


def test_mode_column_position_after_at(tmp_path):
    path = tmp_path / "g.csv"
    _write_csv(path, _base_fields(), [{
        "code": "c1", "source": "brand_disjoint_test",
        "silver_grain_type": "wheat", "manual_grain_type": "wheat",
        "manual_grain_type_status": "confident",
        "manual_grain_type_at": "", "manual_grain_type_note": "",
        "silver_pasta_shape": "", "manual_pasta_shape": "",
        "manual_pasta_shape_status": "empty",
        "manual_pasta_shape_at": "", "manual_pasta_shape_note": "",
    }])
    migrate_csv(path, attrs=["grain_type", "pasta_shape"])
    fieldnames, _ = _read_csv(path)
    for attr in ["grain_type", "pasta_shape"]:
        at_idx = fieldnames.index(f"manual_{attr}_at")
        mode_idx = fieldnames.index(f"manual_{attr}_mode")
        assert mode_idx == at_idx + 1, f"{attr}: _mode must follow _at"


def test_partial_schema_does_not_crash(tmp_path):
    """If the CSV lacks _at columns for some attrs, those attrs are skipped
    (no crash, no truncation). Caller asks for all 8 attrs but CSV only has 1."""
    path = tmp_path / "g.csv"
    # Only grain_type has the full schema; the other 7 attrs are absent
    _write_csv(
        path,
        [
            "code", "source",
            "silver_grain_type", "manual_grain_type",
            "manual_grain_type_status", "manual_grain_type_at", "manual_grain_type_note",
        ],
        [{
            "code": "x1", "source": "test",
            "silver_grain_type": "wheat", "manual_grain_type": "wheat",
            "manual_grain_type_status": "confident",
            "manual_grain_type_at": "2026-05-14T10:00:00+00:00",
            "manual_grain_type_note": "",
        }],
    )
    # Caller passes all 8 attrs even though CSV only has 1 -- must not crash
    migrate_csv(path, attrs=[
        "grain_type", "pasta_shape", "is_filled", "is_organic",
        "is_gluten_free", "is_vegan", "nutri_score_grade", "protein_class",
    ])
    fieldnames, rows = _read_csv(path)
    # grain_type migrated
    assert "manual_grain_type_mode" in fieldnames
    assert rows[0]["manual_grain_type_status"] == "confirmed"
    assert rows[0]["manual_grain_type_mode"] == "blind"
    # Other attrs not present in input -> not added to schema
    assert "manual_pasta_shape_mode" not in fieldnames
    assert "manual_is_vegan_mode" not in fieldnames
    # Output is not truncated (header + 1 row)
    assert len(rows) == 1
