"""LLM fallback — Layer 4 of the cascade.

Public API:
- enrich_product(product, schema): label one product via LLM
- enrich_batch(df, schema): batch wrapper with retries
- build_prompt(product, schema): build few-shot prompt from schema + examples
"""

from src.pipeline.llm_fallback.enrich import enrich_product, enrich_batch
from src.pipeline.llm_fallback.prompts import build_prompt

__all__ = ["enrich_product", "enrich_batch", "build_prompt"]
