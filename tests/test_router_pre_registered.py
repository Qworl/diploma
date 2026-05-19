"""Tests для pre-registered evaluation."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.eval.router_pre_registered import (
    PRE_REGISTERED_BUDGETS, BONFERRONI_N,
    bonferroni_corrected_alpha, evaluate_h1,
)


def test_pre_registered_budgets_are_25_40_50():
    assert PRE_REGISTERED_BUDGETS == (0.25, 0.40, 0.50)
    assert BONFERRONI_N == 3


def test_bonferroni_alpha_correct():
    assert abs(bonferroni_corrected_alpha(0.05) - 0.05/3) < 1e-9
    assert abs(bonferroni_corrected_alpha(0.01) - 0.01/3) < 1e-9


def test_h1_pass_when_one_budget_significant():
    """H1 принята, если router > static значимо хотя бы на одном из 3 бюджетов."""
    stats = pd.DataFrame({
        "budget_target": [0.25, 0.40, 0.50],
        "delta": [0.002, 0.018, 0.001],
        "p_mcnemar": [0.45, 0.001, 0.6],
        "ci_lo": [-0.01, 0.005, -0.01],
        "ci_hi": [0.012, 0.030, 0.012],
    })
    result = evaluate_h1(stats, alpha=0.05)
    assert result["h1_passed"] is True
    assert result["significant_budgets"] == [0.40]


def test_h1_fail_when_no_budget_significant_after_correction():
    """p=0.024 не значим после Bonferroni @α/3=0.0167."""
    stats = pd.DataFrame({
        "budget_target": [0.25, 0.40, 0.50],
        "delta": [0.002, 0.014, 0.001],
        "p_mcnemar": [0.45, 0.024, 0.6],
        "ci_lo": [-0.01, 0.001, -0.01],
        "ci_hi": [0.012, 0.025, 0.012],
    })
    result = evaluate_h1(stats, alpha=0.05)
    assert result["h1_passed"] is False
    assert result["significant_budgets"] == []
