"""Tests for prompt builder (src/pipeline/llm_fallback/prompts.py)."""

from src.pipeline.llm_fallback import build_prompt
from src.pipeline.schemas import PASTA_SCHEMA


def test_build_prompt_pasta():
    product = {
        "product_name": "Barilla Spaghetti No.5",
        "brands": "Barilla",
        "categories_tags": "pastas,spaghetti",
        "ingredients_text": "Durum wheat semolina",
        "quantity": "500g",
    }
    prompt = build_prompt(product, PASTA_SCHEMA)
    assert "grain_type" in prompt
    assert "nutri_score_grade" in prompt
    assert "protein_class" in prompt


def test_build_prompt_missing_fields():
    product = {"product_name": "Some pasta"}
    prompt = build_prompt(product, PASTA_SCHEMA)
    assert "Some pasta" in prompt


def test_build_prompt_includes_few_shot_by_default():
    product = {"product_name": "Mystery pasta"}
    prompt = build_prompt(product, PASTA_SCHEMA)
    assert "Examples:" in prompt
    # Pasta example #1 (Barilla Spaghetti) should appear
    assert "Barilla Spaghetti" in prompt
    # JSON output of an example should be present
    assert '"grain_type": "wheat"' in prompt


def test_build_prompt_can_disable_examples():
    product = {"product_name": "Mystery pasta"}
    prompt = build_prompt(product, PASTA_SCHEMA, include_examples=False)
    assert "Examples:" not in prompt
    assert "Barilla Spaghetti" not in prompt
