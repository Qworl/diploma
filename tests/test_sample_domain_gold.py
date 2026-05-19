"""Tests for src.manual_label.sample_domain_gold (Trek E).

Synthetic silver fixtures avoid touching real parquets; the goal is to
verify pool ratios, attribute coverage, and Trek-D schema completeness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.manual_label.sample_domain_gold import (
    SamplingError,
    build_sample,
)
from src.manual_label.schemas_loader import load_domain_attrs


def _synthetic_chocolate_silver(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    attrs = list(load_domain_attrs("chocolate"))
    n_attrs = len(attrs)
    rows = []
    for i in range(n):
        # Distribute null-count so all three pools are populated.
        target_nulls = int(rng.integers(0, 5))
        null_attrs = set(rng.choice(attrs, size=target_nulls, replace=False))
        row = {
            "code": f"P{i:06d}",
            "product_name": f"Chocolate #{i}",
            "brands": f"Brand{i % 10}",
            "ingredients_text": "cocoa, sugar",
            "quantity": "100g",
        }
        for a in attrs:
            if a in null_attrs:
                row[a] = None
            elif a == "chocolate_type":
                row[a] = ["dark", "milk", "white", "filled", "other"][i % 5]
            elif a == "contains_nuts" or a == "is_organic":
                row[a] = bool(i % 2)
            else:
                # Pick first enum value if available, else string.
                spec = load_domain_attrs("chocolate")[a]
                vals = spec.get("values") or ["x"]
                row[a] = vals[0]
        rows.append(row)
    return pd.DataFrame(rows)


def _synthetic_cheeses_silver(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    attrs = list(load_domain_attrs("cheeses"))
    rows = []
    for i in range(n):
        target_nulls = int(rng.integers(0, 5))
        null_attrs = set(rng.choice(attrs, size=target_nulls, replace=False))
        row = {
            "code": f"C{i:06d}",
            "product_name": f"Cheese #{i}",
            "brands": f"Dairy{i % 10}",
            "ingredients_text": "milk",
            "quantity": "200g",
        }
        for a in attrs:
            if a in null_attrs:
                row[a] = None
            elif a == "texture":
                row[a] = ["hard", "soft", "fresh", "cream", "blue"][i % 5]
            elif a in ("is_pdo", "is_organic", "is_ultra_processed"):
                row[a] = bool(i % 2)
            else:
                spec = load_domain_attrs("cheeses")[a]
                vals = spec.get("values") or ["x"]
                row[a] = vals[0]
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_sample_chocolate_pool_ratios():
    silver = _synthetic_chocolate_silver(n=500)
    sample = build_sample(domain="chocolate", silver=silver, n_total=100, seed=42)
    assert len(sample) == 100
    pool_counts = sample["source"].value_counts().to_dict()
    assert pool_counts.get("pool_a_typical", 0) == 60
    assert pool_counts.get("pool_b_silver_empty", 0) == 25
    assert pool_counts.get("pool_c_hard", 0) == 15


def test_build_sample_cheeses_writes_trek_d_schema():
    silver = _synthetic_cheeses_silver(n=500)
    sample = build_sample(domain="cheeses", silver=silver, n_total=50, seed=7)
    attrs = list(load_domain_attrs("cheeses"))
    for a in attrs:
        for suffix in ("", "_status", "_at", "_mode", "_note"):
            assert f"manual_{a}{suffix}" in sample.columns, f"missing manual_{a}{suffix}"
        assert f"silver_{a}" in sample.columns
    # Initial annotation state must be empty for every manual cell.
    for a in attrs:
        assert (sample[f"manual_{a}"].fillna("") == "").all()
        assert (sample[f"manual_{a}_status"] == "empty").all()
        assert (sample[f"manual_{a}_mode"] == "").all()


def test_build_sample_pool_codes_are_disjoint():
    silver = _synthetic_chocolate_silver(n=800)
    sample = build_sample(domain="chocolate", silver=silver, n_total=200, seed=1)
    assert sample["code"].is_unique
    # Pool A codes have zero silver-nulls in the underlying silver row.
    a_codes = set(sample[sample["source"] == "pool_a_typical"]["code"])
    attrs = list(load_domain_attrs("chocolate"))
    silver["code"] = silver["code"].astype(str)
    cols_present = [a for a in attrs if a in silver.columns]
    a_rows = silver[silver["code"].isin(a_codes)]
    assert a_rows[cols_present].notna().all(axis=1).all(), (
        "Pool A should contain only zero-null rows"
    )


def test_build_sample_raises_if_pool_too_small():
    # n=20 silver rows, request n_total=100 — Pool A alone needs 60 zero-null rows.
    silver = _synthetic_chocolate_silver(n=20)
    with pytest.raises(SamplingError):
        build_sample(domain="chocolate", silver=silver, n_total=100, seed=0)


def test_build_sample_pool_ratio_must_sum_to_100():
    silver = _synthetic_chocolate_silver(n=200)
    with pytest.raises(SamplingError, match="sum to 100"):
        build_sample(domain="chocolate", silver=silver, n_total=50,
                     pool_ratio=(50, 25, 15), seed=0)


def test_build_sample_seed_is_deterministic():
    silver = _synthetic_chocolate_silver(n=400)
    a = build_sample(domain="chocolate", silver=silver, n_total=60, seed=11)
    b = build_sample(domain="chocolate", silver=silver, n_total=60, seed=11)
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True), b.reset_index(drop=True)
    )
