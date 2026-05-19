"""Tests для src.data.split.brand_disjoint."""
from __future__ import annotations
import pandas as pd
import pytest

# Будет импортироваться в task B.2:
from src.data.split.brand_disjoint import brand_disjoint_split


@pytest.fixture
def toy_data():
    """4 бренда × 5 продуктов = 20 строк. Brand A = 5, B = 5, C = 5, D = 5."""
    rows = []
    for brand, n in [("A", 5), ("B", 5), ("C", 5), ("D", 5)]:
        for i in range(n):
            rows.append({"code": f"{brand}{i}", "brand": brand, "y": i % 3})
    return pd.DataFrame(rows)


def test_no_brand_overlap_between_splits(toy_data):
    splits = brand_disjoint_split(toy_data, brand_col="brand",
                                   ratios=(0.6, 0.2, 0.2), seed=42)
    train_brands = set(splits["train"]["brand"])
    val_brands = set(splits["val"]["brand"])
    test_brands = set(splits["test"]["brand"])
    assert not (train_brands & val_brands)
    assert not (train_brands & test_brands)
    assert not (val_brands & test_brands)


def test_split_ratios_approximately_correct(toy_data):
    splits = brand_disjoint_split(toy_data, brand_col="brand",
                                   ratios=(0.6, 0.2, 0.2), seed=42)
    n = len(toy_data)
    # На 20 строках допуск ±15% за счёт целочисленных бакетов
    assert 0.45 <= len(splits["train"]) / n <= 0.75
    assert 0.10 <= len(splits["val"]) / n <= 0.30
    assert 0.10 <= len(splits["test"]) / n <= 0.30


def test_class_coverage_warning_raised(toy_data):
    """Класс, который встречается у одного бренда — попадёт в один split, warning."""
    df = toy_data.copy()
    df.loc[df["brand"] == "A", "y"] = 99  # уникальный класс только у бренда A
    with pytest.warns(UserWarning, match="class coverage"):
        brand_disjoint_split(df, brand_col="brand", ratios=(0.6, 0.2, 0.2),
                              seed=42, check_class_col="y")


def test_reproducibility_with_same_seed(toy_data):
    s1 = brand_disjoint_split(toy_data, brand_col="brand", ratios=(0.6, 0.2, 0.2), seed=42)
    s2 = brand_disjoint_split(toy_data, brand_col="brand", ratios=(0.6, 0.2, 0.2), seed=42)
    for split in ("train", "val", "test"):
        pd.testing.assert_frame_equal(s1[split].reset_index(drop=True),
                                       s2[split].reset_index(drop=True))
