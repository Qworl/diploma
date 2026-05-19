import json
import numpy as np
import pandas as pd
import pytest

from src.pipeline.router.train import (
    train_router,
    RouterArtefacts,
)


def _make_synth_df(n_products=60, n_attrs=4, seed=0):
    """60 products × 4 attrs = 240 pairs. Cascade conf strongly predicts correctness."""
    rng = np.random.default_rng(seed)
    cats = rng.choice(["pasta", "chocolate"], size=n_products)
    rows = []
    for i in range(n_products):
        for j in range(n_attrs):
            conf = rng.uniform(0.3, 0.99)
            correct = int(rng.uniform() < conf)
            rows.append({
                "category": cats[i],
                "code": f"p{i}",
                "attr": ["grain_type", "is_organic", "chocolate_type", "is_filled"][j],
                "cascade_pred": "x",
                "silver_gt": "x" if correct else "y",
                "cascade_conf": conf,
                "cascade_layer": "ml",
                "llm_pred": "x",
                "product_name": f"product {i} {cats[i]}",
                "brands": "Barilla" if i % 3 else "Lindt",
                "cascade_correct": correct,
            })
    return pd.DataFrame(rows)


def test_train_router_returns_artefacts_with_predict_proba():
    df = _make_synth_df(seed=42)
    artefacts = train_router(df, seed=42, n_estimators=50)
    assert isinstance(artefacts, RouterArtefacts)
    proba = artefacts.predict_proba(df.head(5))
    assert proba.shape == (5,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_train_router_calibration_produces_valid_probabilities():
    df = _make_synth_df(seed=42)
    artefacts = train_router(df, seed=42, n_estimators=50)
    val_proba = artefacts.predict_proba(df.tail(20))
    assert not np.isnan(val_proba).any()


def test_train_router_does_not_leak_test_into_train():
    df = _make_synth_df(n_products=20, seed=1)
    artefacts = train_router(df, seed=1, n_estimators=20)
    assert artefacts.train_codes.isdisjoint(artefacts.test_codes)
    assert artefacts.train_codes.isdisjoint(artefacts.val_codes)


def test_train_router_loco_excludes_holdout_category():
    # Build deterministic df with 240 rows: 120 pasta + 120 chocolate
    rng = np.random.default_rng(42)
    rows = []
    for cat, start in [("pasta", 0), ("chocolate", 120)]:
        for i in range(120):
            pid = start + i
            conf = rng.uniform(0.3, 0.99)
            correct = int(rng.uniform() < conf)
            rows.append({
                "category": cat,
                "code": f"p{pid}",
                "attr": "grain_type",
                "cascade_pred": "x",
                "silver_gt": "x" if correct else "y",
                "cascade_conf": conf,
                "cascade_layer": "ml",
                "llm_pred": "x",
                "product_name": f"product {pid} {cat}",
                "brands": "Barilla" if pid % 3 else "Lindt",
                "cascade_correct": correct,
            })
    df = pd.DataFrame(rows)

    from src.pipeline.router.train import train_router_loco
    artefacts = train_router_loco(df, holdout_category="chocolate", seed=42, n_estimators=30)

    train_cats = df[df["code"].isin(artefacts.train_codes)]["category"].unique()
    assert set(train_cats) == {"pasta"}
    test_cats = df[df["code"].isin(artefacts.test_codes)]["category"].unique()
    assert set(test_cats) == {"chocolate"}
