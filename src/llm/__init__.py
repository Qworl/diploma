"""LLM transport: HTTP clients + response parsing.

Used by:
- src/pipeline/llm_fallback (Layer 4 of the cascade)
- src/data/manual_label/llm_assisted (LLM-assisted gold labeling)
- src/data/manual_label/label_cli (translation calls)
"""

from src.llm.client import call_openrouter, call_ollama
from src.llm.parsing import parse_llm_response, _parse_with_status

__all__ = ["call_openrouter", "call_ollama", "parse_llm_response", "_parse_with_status"]
