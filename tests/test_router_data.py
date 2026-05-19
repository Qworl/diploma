import pandas as pd
import pytest
from src.pipeline.router.data import (
    load_joined_cat,
    build_training_dataset,
    by_product_split,
)


def test_load_joined_cat_returns_expected_schema(tmp_path):
    """Mock experiment_per_product + direct_llm_eval → joined df has all required columns."""
    cascade = pd.DataFrame({
        "config": ["regex_ml_bayes"] * 3,
        "code": ["1", "1", "2"],
        "attr": ["a", "b", "a"],
        "gt": ["x", "y", "x"],
        "pred": ["x", "z", "x"],
        "conf": [0.9, 0.5, 0.95],
        "layer": ["ml", "bayes", "regex"],
    })
    direct = pd.DataFrame({
        "category": ["pasta"] * 3,
        "code": ["1", "1", "2"],
        "attr": ["a", "b", "a"],
        "gt": ["x", "y", "x"],
        "pred": ["x", "y", "z"],
        "predicted_non_null": [1, 1, 1],
        "gt_non_null": [1, 1, 1],
        "correct_when_both_present": [1, 1, 0],
    })
    cascade.to_parquet(tmp_path / "experiment_per_product_pasta_stratified.parquet")
    direct.to_parquet(tmp_path / "direct_llm_eval_pasta_stratified.parquet")

    joined = load_joined_cat("pasta", tmp_path)
    assert len(joined) == 3
    assert {"category", "code", "attr", "cascade_pred", "cascade_conf",
            "cascade_layer", "llm_pred", "silver_gt"}.issubset(joined.columns)


def test_build_training_dataset_target_is_binary(tmp_path):
    """build_training_dataset adds cascade_correct ∈ {0, 1}, drops rows w/o silver_gt."""
    cascade = pd.DataFrame({
        "config": ["regex_ml_bayes"] * 4,
        "code": ["1", "1", "2", "2"],
        "attr": ["a", "b", "a", "b"],
        "gt": ["x", None, "x", "y"],  # row 2 has no silver
        "pred": ["x", "y", "z", "y"],
        "conf": [0.9, 0.5, 0.4, 0.8],
        "layer": ["ml"] * 4,
    })
    direct = pd.DataFrame({
        "category": ["pasta"] * 4,
        "code": ["1", "1", "2", "2"],
        "attr": ["a", "b", "a", "b"],
        "gt": ["x", None, "x", "y"],
        "pred": ["x", "y", "x", "y"],
        "predicted_non_null": [1] * 4,
        "gt_non_null": [1, 0, 1, 1],
        "correct_when_both_present": [1, 0, 1, 1],
    })
    cascade.to_parquet(tmp_path / "experiment_per_product_pasta_stratified.parquet")
    direct.to_parquet(tmp_path / "direct_llm_eval_pasta_stratified.parquet")

    df = build_training_dataset(["pasta"], tmp_path)
    # Row with silver_gt=None dropped; 3 rows survive.
    assert len(df) == 3
    assert df["cascade_correct"].isin([0, 1]).all()
    # Row 0: pred=x, silver=x → correct=1; Row 2: pred=z, silver=x → correct=0; Row 3: pred=y, silver=y → correct=1
    assert df["cascade_correct"].tolist() == [1, 0, 1]


def test_by_product_split_no_product_leakage():
    """All rows of a single product code stay in one split."""
    df = pd.DataFrame({
        "code": [f"p{i // 5}" for i in range(50)],  # 10 products × 5 attrs
        "attr": ["a", "b", "c", "d", "e"] * 10,
        "category": ["pasta"] * 50,
        "cascade_correct": [0, 1] * 25,
    })
    train, val, test = by_product_split(df, seed=42)
    train_codes = set(train["code"])
    val_codes = set(val["code"])
    test_codes = set(test["code"])
    assert train_codes.isdisjoint(val_codes)
    assert train_codes.isdisjoint(test_codes)
    assert val_codes.isdisjoint(test_codes)
    # Split ratios approximately 60/20/20 ± 1 product
    assert abs(len(train_codes) - 6) <= 1
    assert abs(len(val_codes) - 2) <= 1
    assert abs(len(test_codes) - 2) <= 1
