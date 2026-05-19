"""Coverage / recovery / CI math on hand-constructed logs."""
import math

import pandas as pd

from src.eval.catalog_completion.metrics import (
    aggregate_metrics,
    wilson_ci,
)


def _logs():
    mask_log = pd.DataFrame([
        {"code": "a", "attr": "x", "masked": True,  "original_value": "wheat"},
        {"code": "a", "attr": "y", "masked": False, "original_value": "true"},
        {"code": "b", "attr": "x", "masked": True,  "original_value": None},
        {"code": "b", "attr": "y", "masked": True,  "original_value": "false"},
    ])
    cascade_log = pd.DataFrame([
        {"code": "a", "attr": "x", "cascade_pred": "wheat", "cascade_layer": "ml",
         "llm_called": False, "cost_tokens": 0},
        {"code": "a", "attr": "y", "cascade_pred": "true",  "cascade_layer": "regex",
         "llm_called": False, "cost_tokens": 0},
        {"code": "b", "attr": "x", "cascade_pred": None,    "cascade_layer": "none",
         "llm_called": True,  "cost_tokens": 1100},
        {"code": "b", "attr": "y", "cascade_pred": "true",  "cascade_layer": "llm",
         "llm_called": True,  "cost_tokens": 1100},
    ])
    return mask_log, cascade_log


def test_coverage_gain_excludes_already_filled_cells():
    mask_log, cascade_log = _logs()
    m = aggregate_metrics(mask_log, cascade_log, n_products=2)
    # 3 cells were masked; cascade filled 2 of those 3 → gain = 2/4 cells = 50pp.
    assert math.isclose(m["coverage_gain_pp"], 50.0, abs_tol=1e-6)


def test_recovery_accuracy_only_on_filled_masked_cells():
    mask_log, cascade_log = _logs()
    m = aggregate_metrics(mask_log, cascade_log, n_products=2)
    # Filled masked cells: a/x ("wheat" vs "wheat" → correct), b/y ("true" vs "false" → wrong)
    # → 1/2 = 0.5
    assert math.isclose(m["recovery_accuracy"], 0.5, abs_tol=1e-6)


def test_llm_calls_per_1000_products():
    mask_log, cascade_log = _logs()
    m = aggregate_metrics(mask_log, cascade_log, n_products=2)
    # 1 row in cascade_log has llm_called=True per product b (we charge once per product
    # via metrics aggregation by `code`).
    assert math.isclose(m["llm_calls_per_1000_products"], 500.0, abs_tol=1e-6)


def test_wilson_ci_returns_lo_hi():
    lo, hi = wilson_ci(8, 10, alpha=0.05)
    assert 0 <= lo < hi <= 1
    assert lo < 0.8 < hi
