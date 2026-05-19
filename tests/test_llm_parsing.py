"""Tests for LLM response parser (src/llm/parsing.py)."""
from unittest.mock import patch

from src.llm.parsing import parse_llm_response, _parse_with_status
from src.pipeline.schemas import PASTA_SCHEMA, BEVERAGE_SCHEMA


def test_parse_llm_response_valid():
    raw = ('{"grain_type": "wheat", "pasta_shape": "spaghetti", '
           '"is_whole_grain": false, "is_organic": false, "is_gluten_free": false, '
           '"is_vegan": true, "nutri_score_grade": "A", "protein_class": "med"}')
    result = parse_llm_response(raw, PASTA_SCHEMA)
    assert result["grain_type"] == "wheat"
    assert result["pasta_shape"] == "spaghetti"
    assert result["is_organic"] is False
    assert result["nutri_score_grade"] == "A"
    assert result["protein_class"] == "med"


def test_parse_llm_response_with_markdown():
    raw = '```json\n{"grain_type": "rice", "pasta_shape": null}\n```'
    result = parse_llm_response(raw, PASTA_SCHEMA)
    assert result["grain_type"] == "rice"
    assert result["pasta_shape"] is None


def test_parse_llm_response_invalid():
    raw = "Sorry, I cannot parse this product"
    result = parse_llm_response(raw, PASTA_SCHEMA)
    assert result == {}


def test_parse_llm_response_validates_values():
    raw = '{"grain_type": "dragon", "is_organic": true}'
    result = parse_llm_response(raw, PASTA_SCHEMA)
    assert "grain_type" not in result
    assert result["is_organic"] is True


def test_parse_with_status_returns_false_on_garbage():
    parsed, ok = _parse_with_status("Sorry, I cannot do this", PASTA_SCHEMA)
    assert parsed == {}
    assert ok is False


def test_parse_with_status_returns_true_when_json_valid_but_empty_fields():
    """JSON parsed but no schema-valid fields → ok=True (not a format error)."""
    parsed, ok = _parse_with_status('{"unrelated": "value"}', PASTA_SCHEMA)
    assert parsed == {}
    assert ok is True


def test_parse_with_status_handles_non_object_json():
    """Top-level array or scalar → ok=False (we expect an object)."""
    parsed, ok = _parse_with_status("[1, 2, 3]", PASTA_SCHEMA)
    assert parsed == {}
    assert ok is False


def test_enrich_product_retries_on_parse_failure():
    """First call returns garbage; retry returns valid JSON → final result is parsed."""
    from src.pipeline.llm_fallback import enrich_product

    responses = iter([
        "I cannot answer this product",
        '{"grain_type": "wheat", "is_organic": true}',
    ])

    def fake_openrouter(messages, model, api_key, *, enforce_json=True):
        return {"raw": next(responses), "usage": {}, "latency_ms": 0.0}

    with patch("src.pipeline.llm_fallback.enrich.call_openrouter", side_effect=fake_openrouter):
        result = enrich_product(
            {"product_name": "Test"}, PASTA_SCHEMA,
            backend="openrouter", api_key="dummy",
            parse_retries=1,
        )
    assert result == {"grain_type": "wheat", "is_organic": True}


def test_enrich_product_no_retry_on_semantic_miss():
    """JSON parses fine but has no schema-valid fields → no retry, return empty."""
    from src.pipeline.llm_fallback import enrich_product

    call_count = [0]

    def fake_openrouter(messages, model, api_key, *, enforce_json=True):
        call_count[0] += 1
        return {"raw": '{"foo": "bar"}', "usage": {}, "latency_ms": 0.0}

    with patch("src.pipeline.llm_fallback.enrich.call_openrouter", side_effect=fake_openrouter):
        result = enrich_product(
            {"product_name": "Test"}, PASTA_SCHEMA,
            backend="openrouter", api_key="dummy",
            parse_retries=2,
        )
    assert result == {}
    assert call_count[0] == 1  # no retries, parse succeeded
