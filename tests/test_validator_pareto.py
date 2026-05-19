"""Smoke test for the Pareto-plot CLI."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
PARQUET_IN = REPO / "datasets/processed/validator_comparison_pasta.parquet"
PNG_OUT = REPO / "datasets/processed/validator_pareto_pasta.png"
PARQUET_OUT = REPO / "datasets/processed/validator_pareto_pasta.parquet"


@pytest.mark.slow
def test_validator_pareto_writes_artefacts():
    assert PARQUET_IN.exists(), "Run Task 3 first."
    env = os.environ.copy(); env["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "src.eval.validator_pareto",
         "--in", str(PARQUET_IN),
         "--png", str(PNG_OUT),
         "--out", str(PARQUET_OUT)],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert PNG_OUT.exists()
    assert PARQUET_OUT.exists()
    df = pd.read_parquet(PARQUET_OUT)
    for col in ("validator", "routing_rate", "accuracy_on_unrouted", "recall", "precision"):
        assert col in df.columns, f"missing col {col}"
    # Should include random and static-policy series
    assert "random" in df["validator"].unique()
    assert "static_policy" in df["validator"].unique()
