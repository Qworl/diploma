"""Test that cold_start_simulation supports --category for chocolate/cheeses replication."""
import json
import os
import subprocess
import tempfile


def test_cold_start_supports_category_arg():
    result = subprocess.run(
        ["python", "-m", "src.eval.catalog_completion.cold_start_simulation", "--help"],
        capture_output=True, text=True,
    )
    assert "--category" in result.stdout, (
        "Expected --category arg:\n" + result.stdout
    )


def test_cold_start_supports_chocolate():
    """Smoke test: with --category chocolate --no-llm, should produce JSON output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        tmp_out = tf.name
    try:
        result = subprocess.run(
            ["python", "-m", "src.eval.catalog_completion.cold_start_simulation",
             "--category", "chocolate", "--no-llm", "--out", tmp_out],
            capture_output=True, text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
        assert result.returncode == 0, f"Failed: {result.stderr[-2000:]}"
        with open(tmp_out) as f:
            data = json.load(f)
        assert "fill_rate" in data, f"Missing fill_rate, keys: {list(data.keys())}"
        assert "accuracy_on_gold" in data, f"Missing accuracy_on_gold, keys: {list(data.keys())}"
    finally:
        os.unlink(tmp_out)
