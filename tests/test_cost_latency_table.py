import pandas as pd
from src.eval.cost_latency_table import aggregate


def test_aggregate_computes_p50_p95_and_cost_per_1k():
    per_call = pd.DataFrame([
        {"model": "M", "context_mode": "partner_input", "cost_usd": 0.001, "latency_ms": 100.0},
        {"model": "M", "context_mode": "partner_input", "cost_usd": 0.002, "latency_ms": 200.0},
        {"model": "M", "context_mode": "partner_input", "cost_usd": 0.001, "latency_ms": 300.0},
        {"model": "M", "context_mode": "partner_input", "cost_usd": 0.001, "latency_ms": 400.0},
    ])
    out = aggregate(per_call)
    row = out.iloc[0]
    assert row["n_calls"] == 4
    assert row["context_mode"] == "partner_input"
    assert abs(row["cost_per_1k_products_usd"] - (0.005 / 4 * 1000)) < 1e-9
    assert row["p50_latency_ms"] == 250.0
    assert 380 <= row["p95_latency_ms"] <= 400


def test_aggregate_splits_by_context_mode():
    per_call = pd.DataFrame([
        {"model": "M", "context_mode": "partner_input",  "cost_usd": 0.001, "latency_ms": 100.0},
        {"model": "M", "context_mode": "off_grounded",   "cost_usd": 0.003, "latency_ms": 800.0},
    ])
    out = aggregate(per_call)
    assert len(out) == 2
    assert set(out["context_mode"]) == {"partner_input", "off_grounded"}
