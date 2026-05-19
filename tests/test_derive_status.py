"""Truth table for derive_status."""
import pytest

from src.manual_label.status import derive_status


@pytest.mark.parametrize("silver,manual,prev,expected", [
    # unsure is sticky regardless of values
    ("wheat", "wheat", "unsure", "unsure"),
    ("",      "rice",  "unsure", "unsure"),
    ("wheat", "",      "unsure", "unsure"),
    # empty manual -> empty (even if silver is set)
    ("wheat", "",      "empty",     "empty"),
    ("wheat", "",      "auto",      "empty"),
    ("wheat", "",      "confirmed", "empty"),
    # silver empty + manual nonempty -> manual_only
    ("",      "rice",  "empty",     "manual_only"),
    ("",      "rice",  "manual_only", "manual_only"),
    # manual == silver (both nonempty) -> confirmed
    ("wheat", "wheat", "empty",     "confirmed"),
    ("wheat", "wheat", "auto",      "confirmed"),
    ("wheat", "wheat", "override",  "confirmed"),
    # manual != silver, both nonempty -> override
    ("wheat", "rice",  "empty",     "override"),
    ("wheat", "rice",  "auto",      "override"),
    ("wheat", "rice",  "confirmed", "override"),
])
def test_derive_status_table(silver, manual, prev, expected):
    assert derive_status(silver, manual, prev) == expected


def test_whitespace_treated_as_empty():
    assert derive_status("  ", "wheat", "empty") == "manual_only"
    assert derive_status("wheat", "   ", "empty") == "empty"


def test_save_row_uses_derive_when_status_omitted(tmp_path):
    import csv
    import importlib.util
    import sys
    from pathlib import Path

    # datasets/ cannot be made a package (it shadows the HF datasets library),
    # so we load app.py directly via importlib for this one test only.
    _APP_PATH = Path(__file__).resolve().parents[1] / "datasets" / "manual_label" / "app.py"
    _spec = importlib.util.spec_from_file_location("manual_label_app_saverow", _APP_PATH)
    app_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(app_mod)

    csv_path = tmp_path / "pasta_gold_250.csv"
    fieldnames = [
        "code", "product_name", "brands", "ingredients_text", "quantity", "lang",
        "source", "silver_grain_type", "manual_grain_type",
        "manual_grain_type_status", "manual_grain_type_at", "manual_grain_type_note",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({"code": "x1", "silver_grain_type": "wheat"})
    # Point app at this CSV
    monkey_root = app_mod.ROOT
    app_mod.ROOT = tmp_path
    try:
        ok = app_mod.save_row("pasta", "x1", {"grain_type": "rice"}, None)
        assert ok
        with csv_path.open() as f:
            row = next(csv.DictReader(f))
        assert row["manual_grain_type"] == "rice"
        assert row["manual_grain_type_status"] == "override"  # silver=wheat, manual=rice
    finally:
        app_mod.ROOT = monkey_root
