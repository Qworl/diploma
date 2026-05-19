"""Cascade predictor must produce one row per (code, attr) with predicted + confidence."""
from pathlib import Path
import pandas as pd
import pytest

from src.eval.cascade_predict import predict_cascade


def _make_pasta_fixture(tmp_path: Path):
    # Two known pasta products from the silver standard
    return pd.DataFrame([
        {"code": "8001234567890", "product_name": "Spaghetti integrali", "brands": "Barilla",
         "ingredients_text": "Semolina di grano duro integrale", "quantity": "500 g",
         "categories_tags": "en:pastas"},
        {"code": "8009876543210", "product_name": "Penne rigate", "brands": "De Cecco",
         "ingredients_text": "Semola di grano duro", "quantity": "500 g",
         "categories_tags": "en:pastas"},
    ])


def test_predict_returns_long_format_with_required_columns(tmp_path):
    df = _make_pasta_fixture(tmp_path)
    out = predict_cascade(df, category="pasta_stratified")
    required = {"code", "attr", "predicted", "confidence", "layer"}
    assert required.issubset(out.columns), f"missing cols: {required - set(out.columns)}"
    assert len(out) > 0
    assert set(out["code"].unique()) == {"8001234567890", "8009876543210"}


def test_predict_layer_field_uses_known_vocab(tmp_path):
    df = _make_pasta_fixture(tmp_path)
    out = predict_cascade(df, category="pasta_stratified")
    valid = {"regex", "ml", "bayes", "llm", "abstain"}
    assert set(out["layer"].unique()).issubset(valid), \
        f"unexpected layer values: {set(out['layer'].unique()) - valid}"


def test_predict_confidence_in_unit_range(tmp_path):
    df = _make_pasta_fixture(tmp_path)
    out = predict_cascade(df, category="pasta_stratified")
    nonnull = out["confidence"].dropna()
    assert ((nonnull >= 0.0) & (nonnull <= 1.0)).all()


def test_predict_handles_empty_dataframe():
    """Empty input → empty DataFrame with the required columns, no crash."""
    out = predict_cascade(pd.DataFrame(columns=["code", "product_name", "brands",
                                                 "ingredients_text", "quantity"]),
                          category="pasta_stratified")
    required = {"code", "attr", "predicted", "confidence", "layer"}
    assert required.issubset(out.columns) or out.empty
    assert len(out) == 0
