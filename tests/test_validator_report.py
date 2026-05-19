"""Smoke test for the validator_report CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PARQUET = REPO / "datasets/processed/validator_comparison_pasta.parquet"
SUMMARY = REPO / "datasets/processed/validator_comparison_pasta_summary.json"


@pytest.mark.slow
def test_validator_report_emits_summary_json():
    assert PARQUET.exists(), (
        "Task 3 (validator_comparison.py) must run first to produce the parquet."
    )
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "src.eval.validator_report",
         "--in", str(PARQUET), "--out", str(SUMMARY)],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert SUMMARY.exists()
    payload = json.loads(SUMMARY.read_text())
    # Required top-level sections
    for key in ("overall", "by_attr", "hypotheses", "static_policy_baseline",
                "n_cells", "validator_columns"):
        assert key in payload, f"Missing key: {key}"
    # 4 validators (Bayes dropped per Task 0)
    assert set(payload["validator_columns"]) == {
        "xgb_uncertainty", "mahalanobis", "layer_disagree", "ece_attr",
    }
    # Hypothesis report cards
    for h in ("H1", "H2", "H3"):
        assert h in payload["hypotheses"]
        assert "verdict" in payload["hypotheses"][h]
        assert payload["hypotheses"][h]["verdict"] in {"accept", "reject", "inconclusive"}
    # Markdown body printed to stdout
    assert "| Validator |" in result.stdout
