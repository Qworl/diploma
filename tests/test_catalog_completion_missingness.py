"""Empirical missingness profile from silver standards."""
import json
import tempfile

import pandas as pd

from src.eval.catalog_completion.missingness import (
    compute_missingness_profile,
    save_profile,
    load_profile,
)


def _fixture():
    return pd.DataFrame({
        "product_name": ["a", "b", "c", "d"],
        "brands": ["x", "x", None, "y"],
        "grain_type": ["wheat", None, None, "rice"],   # 50% missing
        "is_organic": [True, False, True, False],      # 0% missing -> clamped to 5%
    })


def test_missingness_rates_clamped_to_open_interval():
    prof = compute_missingness_profile(_fixture(), target_attrs=["grain_type", "is_organic"])
    assert 0.45 <= prof["target_attrs"]["grain_type"] <= 0.55
    assert prof["target_attrs"]["is_organic"] == 0.05  # clamped from 0


def test_partner_field_missingness_recorded():
    prof = compute_missingness_profile(_fixture(), target_attrs=["grain_type"])
    assert prof["partner_attrs"]["brands"] == 0.25  # 1/4 missing
    assert prof["partner_attrs"]["product_name"] == 0.05  # clamped


def test_roundtrip(tmp_path):
    prof = compute_missingness_profile(_fixture(), target_attrs=["grain_type"])
    p = tmp_path / "prof.json"
    save_profile(prof, str(p))
    back = load_profile(str(p))
    assert back == prof


def test_n_rows_recorded():
    prof = compute_missingness_profile(_fixture(), target_attrs=["grain_type"])
    assert prof["n_rows"] == 4
