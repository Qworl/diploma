"""Tests for Cohen's κ inter-annotator agreement."""
import pandas as pd
import pytest

from src.eval.manual_label_iaa import compute_kappa_table, kappa_for_pair


def test_perfect_agreement_kappa_is_one():
    a = pd.Series(["wheat", "rice", "corn", "wheat", "rice"])
    b = pd.Series(["wheat", "rice", "corn", "wheat", "rice"])
    assert kappa_for_pair(a, b) == pytest.approx(1.0)


def test_complete_disagreement_kappa_negative():
    a = pd.Series(["wheat"] * 5 + ["rice"] * 5)
    b = pd.Series(["rice"] * 5 + ["wheat"] * 5)
    assert kappa_for_pair(a, b) < 0


def test_empty_values_excluded():
    a = pd.Series(["wheat", "", None, "rice"])
    b = pd.Series(["wheat", "rice", "wheat", "rice"])
    # Rows 0 and 3 are the only usable ones (both annotators non-empty).
    # On these, agreement is perfect ⇒ κ = 1.0.
    assert kappa_for_pair(a, b) == pytest.approx(1.0)


def test_compute_kappa_table_per_attribute():
    gold = pd.DataFrame({
        "code": ["c1", "c2", "c3"],
        "manual_grain_type": ["wheat", "rice", "corn"],
        "manual_is_organic": ["True", "False", "True"],
    })
    proxy = pd.DataFrame({
        "code": ["c1", "c2", "c3"],
        "proxy_grain_type": ["wheat", "rice", "wheat"],   # 2/3 agree
        "proxy_is_organic": ["True", "False", "True"],    # 3/3 agree
    })
    table = compute_kappa_table(gold, proxy, attrs=["grain_type", "is_organic"])
    assert set(table["attribute"]) == {"grain_type", "is_organic"}
    assert table.set_index("attribute").loc["is_organic", "kappa"] == pytest.approx(1.0)
    assert table.set_index("attribute").loc["grain_type", "n"] == 3
