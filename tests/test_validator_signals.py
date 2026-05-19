"""Unit tests for the four validator signal computations."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.eval.validator_signals import (
    layer_disagreement_score,
    load_per_attr_ece,
    per_attr_ece_score,
    xgb_uncertainty_score,
)


# ---- XGB max-prob inversion ------------------------------------------------
def test_xgb_uncertainty_one_minus_max_prob():
    proba = np.array([0.7, 0.2, 0.1])
    assert xgb_uncertainty_score(proba) == pytest.approx(0.3)


def test_xgb_uncertainty_high_when_uniform():
    proba = np.array([0.34, 0.33, 0.33])
    assert xgb_uncertainty_score(proba) == pytest.approx(1 - 0.34)


def test_xgb_uncertainty_none_when_no_proba():
    assert xgb_uncertainty_score(None) is None


# ---- Layer disagreement ----------------------------------------------------
def test_layer_disagreement_zero_when_equal():
    assert layer_disagreement_score("penne", "penne") == 0.0


def test_layer_disagreement_one_when_different():
    assert layer_disagreement_score("penne", "spaghetti") == 1.0


def test_layer_disagreement_none_when_regex_missing():
    assert layer_disagreement_score(None, "penne") is None


def test_layer_disagreement_none_when_ml_missing():
    assert layer_disagreement_score("penne", None) is None


def test_layer_disagreement_normalises_case_and_bool():
    assert layer_disagreement_score("True", True) == 0.0
    assert layer_disagreement_score(" Penne ", "penne") == 0.0


# ---- Per-attr ECE loader ---------------------------------------------------
def test_load_per_attr_ece_reads_ece_raw(tmp_path: Path):
    calib_dir = tmp_path
    payload = {"attr": "grain_type", "ece_raw": 0.123, "ece_calibrated": None}
    (calib_dir / "pasta_stratified_grain_type_calibration.json").write_text(json.dumps(payload))
    ece = load_per_attr_ece(
        category="pasta_stratified",
        attrs=("grain_type",),
        calib_dir=str(calib_dir),
    )
    assert ece == {"grain_type": pytest.approx(0.123)}


def test_load_per_attr_ece_prefers_calibrated_when_present(tmp_path: Path):
    payload = {"attr": "x", "ece_raw": 0.20, "ece_calibrated": 0.05}
    (tmp_path / "pasta_stratified_x_calibration.json").write_text(json.dumps(payload))
    ece = load_per_attr_ece("pasta_stratified", ("x",), calib_dir=str(tmp_path))
    assert ece == {"x": pytest.approx(0.05)}


def test_load_per_attr_ece_missing_file_is_omitted(tmp_path: Path):
    ece = load_per_attr_ece("pasta_stratified", ("missing",), calib_dir=str(tmp_path))
    assert ece == {}


def test_per_attr_ece_score_returns_constant_for_attr():
    ece = {"grain_type": 0.12, "is_filled": 0.03}
    assert per_attr_ece_score("grain_type", ece) == 0.12
    assert per_attr_ece_score("is_filled", ece) == 0.03
    assert per_attr_ece_score("unknown", ece) is None
