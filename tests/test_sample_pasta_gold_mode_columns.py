"""Verify sample CSV includes empty manual_<attr>_mode columns."""
import pandas as pd
from src.manual_label.sample_pasta_gold import build_sample
from src.manual_label.schemas_loader import load_pasta_attrs


def _minimal_inputs():
    silver = pd.DataFrame({
        "code": [f"c{i}" for i in range(60)],
        "product_name": [f"Pasta {i}" for i in range(60)],
        "brands": [f"B{i % 10}" for i in range(60)],
        "ingredients_text": ["wheat"] * 60,
        "quantity": ["500g"] * 60,
        "lang": ["en"] * 60,
        "silver_grain_type": ["wheat"] * 60,
        "silver_pasta_shape": ["spaghetti"] * 60,
        "silver_is_filled": ["False"] * 60,
        "silver_is_organic": ["False"] * 60,
        "silver_is_gluten_free": ["False"] * 60,
        "silver_is_vegan": ["True"] * 60,
        "silver_nutri_score_grade": ["A"] * 30 + [None] * 30,
        "silver_protein_class": ["med"] * 30 + [None] * 30,
    })
    split = pd.DataFrame({
        "code": [f"c{i}" for i in range(60)],
        "split": ["test"] * 20 + ["train"] * 30 + ["val"] * 10,
    })
    dis = pd.DataFrame({
        "code": [f"c{i}" for i in range(20, 30)],
        "attr": ["grain_type"] * 10,
        "cascade_pred": ["wheat"] * 10,
        "llm_pred": ["rice"] * 10,
    })
    return silver, split, dis


def test_mode_columns_present_and_empty():
    silver, split, dis = _minimal_inputs()
    df = build_sample(
        silver_extended=silver, split=split, disagreement=dis,
        n_total=20, n_test=15, n_disagreement=3, n_control=2, seed=42,
    )
    attrs = load_pasta_attrs()
    for attr in attrs:
        assert f"manual_{attr}_mode" in df.columns, f"missing _mode for {attr}"
        assert (df[f"manual_{attr}_mode"] == "").all(), f"non-empty default for {attr}"


def test_mode_column_position_after_at():
    """`_mode` should appear immediately after `_at` for readability."""
    silver, split, dis = _minimal_inputs()
    df = build_sample(
        silver_extended=silver, split=split, disagreement=dis,
        n_total=20, n_test=15, n_disagreement=3, n_control=2, seed=42,
    )
    cols = list(df.columns)
    attrs = load_pasta_attrs()
    for attr in attrs:
        at_idx = cols.index(f"manual_{attr}_at")
        mode_idx = cols.index(f"manual_{attr}_mode")
        assert mode_idx == at_idx + 1, f"{attr}: _mode should follow _at"
