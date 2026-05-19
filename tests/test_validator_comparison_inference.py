"""Smoke test: validator_comparison main builds the expected parquet schema."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "datasets/processed/validator_comparison_pasta.parquet"

REQUIRED_COLUMNS = {
    "code", "attr", "manual_value", "silver_value",
    "cascade_pred", "cascade_layer", "status", "mode",
    "regex_pred", "ml_pred", "xgb_max_prob", "xgb_uncertainty",
    "mahalanobis", "layer_disagree", "ece_attr",
    "is_error", "has_manual",
}


@pytest.mark.slow
def test_validator_comparison_writes_expected_schema():
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "src.eval.validator_comparison",
         "--out", str(OUT)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert OUT.exists(), f"Output parquet missing: {OUT}"

    df = pd.read_parquet(OUT)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    # Expect 8 attrs * 239 products = 1912 rows
    assert len(df) == 1912, f"Row count {len(df)} != 1912 (8 attrs * 239 products)"

    # Sanity: cascade ran, is_error is well-typed
    assert df["is_error"].dtype == bool
    # At least some XGB scores should be present (ML covers most attrs)
    assert df["xgb_max_prob"].notna().sum() > 1000
