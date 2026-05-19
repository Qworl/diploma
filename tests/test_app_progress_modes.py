"""category_progress reports override rates split by mode."""
import csv
import importlib.util
import sys
from pathlib import Path

# datasets/ cannot be made a package (it shadows the HF datasets library),
# so we load app.py directly via importlib.
_APP_PATH = Path(__file__).resolve().parents[1] / "datasets" / "manual_label" / "app.py"
_spec = importlib.util.spec_from_file_location("manual_label_app_progress", _APP_PATH)
app_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app_mod)


def _write_progress_csv(path: Path) -> None:
    fn = [
        "code", "source",
        "silver_grain_type", "manual_grain_type",
        "manual_grain_type_status", "manual_grain_type_at",
        "manual_grain_type_mode", "manual_grain_type_note",
    ]
    rows = [
        # blind confirmed
        {"code": "b1", "source": "x", "silver_grain_type": "wheat",
         "manual_grain_type": "wheat", "manual_grain_type_status": "confirmed",
         "manual_grain_type_at": "", "manual_grain_type_mode": "blind",
         "manual_grain_type_note": ""},
        # blind override
        {"code": "b2", "source": "x", "silver_grain_type": "wheat",
         "manual_grain_type": "rice", "manual_grain_type_status": "override",
         "manual_grain_type_at": "", "manual_grain_type_mode": "blind",
         "manual_grain_type_note": ""},
        # prefill confirmed
        {"code": "p1", "source": "x", "silver_grain_type": "wheat",
         "manual_grain_type": "wheat", "manual_grain_type_status": "confirmed",
         "manual_grain_type_at": "", "manual_grain_type_mode": "prefill",
         "manual_grain_type_note": ""},
        # prefill confirmed (2)
        {"code": "p2", "source": "x", "silver_grain_type": "wheat",
         "manual_grain_type": "wheat", "manual_grain_type_status": "confirmed",
         "manual_grain_type_at": "", "manual_grain_type_mode": "prefill",
         "manual_grain_type_note": ""},
        # prefill override
        {"code": "p3", "source": "x", "silver_grain_type": "wheat",
         "manual_grain_type": "rice", "manual_grain_type_status": "override",
         "manual_grain_type_at": "", "manual_grain_type_mode": "prefill",
         "manual_grain_type_note": ""},
        # auto (not done)
        {"code": "a1", "source": "x", "silver_grain_type": "wheat",
         "manual_grain_type": "wheat", "manual_grain_type_status": "auto",
         "manual_grain_type_at": "", "manual_grain_type_mode": "prefill",
         "manual_grain_type_note": ""},
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(rows)


def test_per_mode_override_rate(tmp_path):
    csv_path = tmp_path / "pasta_gold_250.csv"
    _write_progress_csv(csv_path)
    monkey = app_mod.ROOT
    app_mod.ROOT = tmp_path
    try:
        prog = app_mod.category_progress("pasta")
        assert prog["total"] == 6
        # done = confirmed + override + manual_only + unsure (not auto/empty)
        assert prog["done"] == 5
        rates = prog["override_rate_by_mode"]["grain_type"]
        # blind: 1 override out of 2 audited = 0.5
        assert rates["blind"]["n_audited"] == 2
        assert abs(rates["blind"]["override_rate"] - 0.5) < 1e-9
        # prefill: 1 override out of 3 audited (auto excluded) = 1/3
        assert rates["prefill"]["n_audited"] == 3
        assert abs(rates["prefill"]["override_rate"] - 1/3) < 1e-9
    finally:
        app_mod.ROOT = monkey


def test_filter_only_auto(tmp_path):
    csv_path = tmp_path / "pasta_gold_250.csv"
    _write_progress_csv(csv_path)
    monkey = app_mod.ROOT
    app_mod.ROOT = tmp_path
    try:
        html = app_mod.render_category("pasta", only="auto")
        # Only row a1 should appear
        assert "/pasta/a1" in html
        assert "/pasta/b1" not in html
        assert "/pasta/p1" not in html
    finally:
        app_mod.ROOT = monkey


def test_per_mode_override_rate_includes_llm(tmp_path):
    """category_progress reports llm-mode override rate separately."""
    csv_path = tmp_path / "pasta_gold_250.csv"
    fn = [
        "code", "source",
        "silver_grain_type", "manual_grain_type",
        "manual_grain_type_status", "manual_grain_type_at",
        "manual_grain_type_mode", "manual_grain_type_note",
    ]
    rows = [
        # 1 blind confirmed
        {"code": "b1", "source": "x", "silver_grain_type": "wheat",
         "manual_grain_type": "wheat", "manual_grain_type_status": "confirmed",
         "manual_grain_type_at": "", "manual_grain_type_mode": "blind",
         "manual_grain_type_note": ""},
        # 1 prefill confirmed
        {"code": "p1", "source": "x", "silver_grain_type": "wheat",
         "manual_grain_type": "wheat", "manual_grain_type_status": "confirmed",
         "manual_grain_type_at": "", "manual_grain_type_mode": "prefill",
         "manual_grain_type_note": ""},
        # 2 llm: 1 confirmed + 1 override
        {"code": "l1", "source": "x", "silver_grain_type": "wheat",
         "manual_grain_type": "wheat", "manual_grain_type_status": "confirmed",
         "manual_grain_type_at": "", "manual_grain_type_mode": "llm",
         "manual_grain_type_note": ""},
        {"code": "l2", "source": "x", "silver_grain_type": "wheat",
         "manual_grain_type": "rice", "manual_grain_type_status": "override",
         "manual_grain_type_at": "", "manual_grain_type_mode": "llm",
         "manual_grain_type_note": ""},
    ]
    import csv as _csv
    with csv_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(rows)

    # Reuse the importlib-loaded app_mod from the top of the file
    monkey = app_mod.ROOT
    app_mod.ROOT = tmp_path
    try:
        prog = app_mod.category_progress("pasta")
        rates = prog["override_rate_by_mode"]["grain_type"]
        assert rates["llm"]["n_audited"] == 2
        assert rates["llm"]["n_override"] == 1
        assert abs(rates["llm"]["override_rate"] - 0.5) < 1e-9
        # blind/prefill still populated for back-compat
        assert rates["blind"]["n_audited"] == 1
        assert rates["prefill"]["n_audited"] == 1
    finally:
        app_mod.ROOT = monkey


def test_filter_only_override(tmp_path):
    csv_path = tmp_path / "pasta_gold_250.csv"
    _write_progress_csv(csv_path)
    monkey = app_mod.ROOT
    app_mod.ROOT = tmp_path
    try:
        html = app_mod.render_category("pasta", only="override")
        assert "/pasta/b2" in html
        assert "/pasta/p3" in html
        assert "/pasta/b1" not in html
    finally:
        app_mod.ROOT = monkey
