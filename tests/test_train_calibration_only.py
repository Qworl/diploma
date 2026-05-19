"""Test that train.py supports --calibration-only flag without retraining XGBoost."""
import os
import subprocess
from pathlib import Path

import pytest


def test_calibration_only_flag_exists():
    """--calibration-only flag should be exposed in train.py CLI."""
    result = subprocess.run(
        ["python", "-m", "src.pipeline.ml.train", "--help"],
        capture_output=True, text=True,
    )
    assert "--calibration-only" in result.stdout, (
        "Expected --calibration-only in CLI help, got:\n" + result.stdout
    )


def test_calibration_only_does_not_retrain_xgb(tmp_path, monkeypatch):
    """With --calibration-only, existing XGBoost model files should be untouched."""
    pkl_path = Path("models/pasta_stratified_grain_type_xgb.pkl")
    if not pkl_path.exists():
        pytest.skip("No baseline XGBoost model to verify untouched")

    before_mtime = pkl_path.stat().st_mtime

    result = subprocess.run(
        ["python", "-m", "src.pipeline.ml.train",
         "--category", "pasta_stratified",
         "--calibration-only"],
        capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, f"Train failed: {result.stderr}"

    after_mtime = pkl_path.stat().st_mtime
    assert after_mtime == before_mtime, "XGBoost pkl was modified despite --calibration-only"
