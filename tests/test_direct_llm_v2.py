"""Direct LLM runner records latency_ms, in_tokens, out_tokens, cost_usd, and
respects context_mode (partner_input vs off_grounded)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.eval.direct_llm_v2 import run_llm_on_products


def _make_fake_caller(usage, latency_ms):
    captured = {"messages": []}
    def _call(*, messages, model, api_key, **kw):
        captured["messages"].append(messages)
        return {"raw": '{"x": "A"}', "usage": usage, "latency_ms": latency_ms}
    _call.captured = captured
    return _call


def test_runner_records_latency_and_usage(tmp_path):
    products = pd.DataFrame([
        {"code": "1", "product_name": "X", "brands": "B",
         "ingredients_text": "i", "quantity": "1g"},
    ])
    fake = _make_fake_caller({"prompt_tokens": 100, "completion_tokens": 50}, 1234.0)
    out_path = tmp_path / "out.parquet"
    df = run_llm_on_products(products, domain="pasta", model="openai/gpt-4o-mini",
                             api_key="x", out_path=out_path,
                             context_mode="partner_input", call_fn=fake)
    for col in ("latency_ms", "in_tokens", "out_tokens", "cost_usd", "context_mode"):
        assert col in df.columns
    assert df["latency_ms"].iloc[0] == 1234.0
    assert df["in_tokens"].iloc[0] == 100
    assert df["out_tokens"].iloc[0] == 50
    assert df["cost_usd"].iloc[0] > 0
    assert df["context_mode"].iloc[0] == "partner_input"
    assert out_path.exists()


def test_off_grounded_mode_uses_cached_off_fields(tmp_path):
    products = pd.DataFrame([{"code": "1", "product_name": "X", "brands": "B",
                              "ingredients_text": "i", "quantity": "1g"}])
    cache_dir = tmp_path / "off_cache"
    cache_dir.mkdir()
    (cache_dir / "1.json").write_text(json.dumps({
        "code": "1", "status": 1,
        "product": {
            "product_name": "X", "brands": "B", "quantity": "1g",
            "ingredients_text": "i",
            "categories_tags": ["en:pastas"],
            "labels_tags": ["en:organic"],
            "nutriments": {"fat_100g": 2.0},
        },
    }))
    fake = _make_fake_caller({"prompt_tokens": 50, "completion_tokens": 10}, 500.0)
    df = run_llm_on_products(products, domain="pasta", model="openai/gpt-4o-mini",
                             api_key="x", out_path=tmp_path / "out.parquet",
                             context_mode="off_grounded", off_cache_dir=cache_dir,
                             call_fn=fake)
    assert df["context_mode"].iloc[0] == "off_grounded"
    sent_msg = fake.captured["messages"][0][-1]["content"]
    assert "en:organic" in sent_msg or "organic" in sent_msg.lower()
