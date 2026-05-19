"""LLM-based enrichment (Layer 4 of the cascade).

`enrich_product` calls the LLM via src/llm/client, parses response via
src/llm/parsing. `enrich_batch` runs it across a DataFrame.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from src.llm.client import call_openrouter, call_ollama
from src.llm.parsing import _parse_with_status
from src.pipeline.llm_fallback.prompts import build_prompt

logger = logging.getLogger(__name__)


_RETRY_FEEDBACK = (
    "Your previous response could not be parsed as JSON. "
    "Respond ONLY with a single valid JSON object matching the schema described above, "
    "no prose, no code fences, no explanation."
)


def enrich_product(product: dict, schema: dict, backend: str = "openrouter",
                   model: str = "anthropic/claude-haiku-4.5",
                   api_key: str | None = None,
                   ollama_url: str = "http://localhost:11434",
                   *, parse_retries: int = 1,
                   enforce_json: bool = True) -> dict:
    """Enrich a single product with LLM-extracted attributes.

    parse_retries: number of additional attempts when JSON parse fails (not when it
    succeeds with empty fields — that's a semantic miss, not a format bug).
    """
    prompt = build_prompt(product, schema)
    messages = [{"role": "user", "content": prompt}]

    if backend == "openrouter":
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set")

        def _call(msgs):
            return call_openrouter(msgs, model, key, enforce_json=enforce_json)["raw"]
    elif backend == "ollama":
        def _call(msgs):
            return call_ollama(msgs, model, ollama_url, enforce_json=enforce_json)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    last_result: dict = {}
    for attempt in range(parse_retries + 1):
        raw = _call(messages)
        last_result, parsed_ok = _parse_with_status(raw, schema)
        if parsed_ok or attempt == parse_retries:
            return last_result
        # Append failed reply + corrective prompt for next attempt
        logger.warning("LLM response not parseable on attempt %d; retrying with feedback", attempt + 1)
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _RETRY_FEEDBACK},
        ]
    return last_result


def _enrich_one(product: dict, schema: dict, backend: str, model: str,
                api_key: str | None, ollama_url: str, max_retries: int) -> dict:
    """Enrich a single product with retries. Used by enrich_batch."""
    for attempt in range(max_retries):
        try:
            attrs = enrich_product(
                product, schema, backend=backend, model=model,
                api_key=api_key, ollama_url=ollama_url,
            )
            return {"code": product.get("code"), **attrs}
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {product.get('product_name', '?')}: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
    return {"code": product.get("code")}


def enrich_batch(df: pd.DataFrame, schema: dict, backend: str = "openrouter",
                 model: str = "anthropic/claude-haiku-4.5",
                 api_key: str | None = None,
                 ollama_url: str = "http://localhost:11434",
                 max_retries: int = 3,
                 max_workers: int = 10) -> pd.DataFrame:
    """Enrich a DataFrame of products in parallel. Returns DataFrame with extracted attributes."""
    products = [row.to_dict() for _, row in df.iterrows()]
    results = [None] * len(products)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(
                _enrich_one, product, schema, backend, model,
                api_key, ollama_url, max_retries,
            ): i
            for i, product in enumerate(products)
        }
        with tqdm(total=len(products), desc="Enriching") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                pbar.update(1)

    return pd.DataFrame(results)
