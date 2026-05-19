"""Decision boundaries for Bonferroni-corrected hypothesis tests."""
import math
import pytest

from src.eval.catalog_completion.hypotheses import (
    BONFERRONI_ALPHA,
    evaluate_h1_coverage_gain,
    evaluate_h2_recovery_accuracy,
    evaluate_h3_electronics_ceiling,
)


def test_bonferroni_alpha_value():
    assert math.isclose(BONFERRONI_ALPHA, 0.05 / 3, abs_tol=1e-9)


def test_h1_passes_when_gain_far_above_30pp():
    # 60% gain over 1000 cells — clearly above 30pp.
    res = evaluate_h1_coverage_gain(coverage_gain_cells=600, n_cells=1000)
    assert res["decision"] == "REJECT_H0"
    assert res["p_value"] < BONFERRONI_ALPHA


def test_h1_fails_when_gain_below_threshold():
    res = evaluate_h1_coverage_gain(coverage_gain_cells=200, n_cells=1000)
    assert res["decision"] == "FAIL_TO_REJECT"


def test_h2_passes_when_recovery_high():
    res = evaluate_h2_recovery_accuracy(n_correct=85, n_filled=100)
    assert res["decision"] == "REJECT_H0"


def test_h2_fails_when_recovery_low():
    res = evaluate_h2_recovery_accuracy(n_correct=60, n_filled=100)
    assert res["decision"] == "FAIL_TO_REJECT"


def test_h3_passes_when_electronics_ceiling_holds():
    # pasta gain 40pp, electronics gain 10pp → ratio 0.25 < 0.5 → reject H0 (i.e. confirm ceiling).
    res = evaluate_h3_electronics_ceiling(
        electronics_gain_cells=100, electronics_n_cells=1000,
        pasta_gain_cells=400, pasta_n_cells=1000,
    )
    assert res["decision"] == "REJECT_H0"


def test_h3_fails_when_electronics_too_strong():
    # electronics gain == pasta gain → ratio 1 > 0.5 → cannot confirm ceiling.
    res = evaluate_h3_electronics_ceiling(
        electronics_gain_cells=400, electronics_n_cells=1000,
        pasta_gain_cells=400, pasta_n_cells=1000,
    )
    assert res["decision"] == "FAIL_TO_REJECT"


def test_h3_skipped_when_data_thin():
    res = evaluate_h3_electronics_ceiling(
        electronics_gain_cells=0, electronics_n_cells=10,
        pasta_gain_cells=400, pasta_n_cells=1000,
        min_n=30,
    )
    assert res["decision"] == "SKIPPED_INSUFFICIENT_DATA"
