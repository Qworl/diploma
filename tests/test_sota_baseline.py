import pandas as pd
import pytest

from src.eval.sota_baseline import tokenize_for_opentag, train_eval_opentag


def test_tokenize_lower_split_punct():
    tokens = tokenize_for_opentag("Barilla 500g Spaghetti N.5")
    assert tokens == ["barilla", "500g", "spaghetti", "n", "5"]


def test_train_eval_returns_required_keys(tmp_path):
    df = pd.DataFrame({
        "product_name": [f"Pasta {i}" for i in range(50)],
        "brands": ["Barilla"] * 50,
        "grain_type": ["wheat"] * 40 + ["corn"] * 10,
    })
    result = train_eval_opentag(df, target_col="grain_type",
                                 max_epochs=1, batch_size=8)
    assert "accuracy" in result
    assert "macro_f1" in result
    assert 0 <= result["accuracy"] <= 1
