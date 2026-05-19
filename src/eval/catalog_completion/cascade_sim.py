"""Run the existing regex → ML → bayes → LLM cascade on masked rows.

This wraps `src.eval.cascade_vs_audited_gold.run_cascade` so we don't duplicate
layered-cascade code. Layer 4 (LLM) is opt-in via `enable_llm=True`.

Cost accounting: each LLM call is charged a flat token estimate equal to the
average prompt + response sizes observed in `src.eval.layer4_llm`. We do not
return real $/token (varies by model) — we return token counts; conversion to
dollars happens in the runner using `MODEL_COSTS_USD_PER_1M`.
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

from src.common import MODELS_DIR
from src.eval.cascade_vs_audited_gold import (
    bayes_layer,
    load_bayesian,
    load_ml_models,
    load_thresholds,
    ml_layer,
    regex_layer,
)
from src.pipeline.regex.extractor import RegexExtractor

logger = logging.getLogger(__name__)

# Token estimates from layer4_llm telemetry (avg over the gpt-oss-120b runs).
LLM_PROMPT_TOKENS_AVG = 850
LLM_RESPONSE_TOKENS_AVG = 250
LLM_TOTAL_TOKENS_AVG = LLM_PROMPT_TOKENS_AVG + LLM_RESPONSE_TOKENS_AVG

# USD per 1M tokens for the reference model (gpt-oss-120b on OpenRouter, Nov 2025).
MODEL_COSTS_USD_PER_1M = {
    "gpt-oss-120b": {"prompt": 0.15, "response": 0.60},
    "claude-haiku-4.5": {"prompt": 1.00, "response": 5.00},
}
DEFAULT_LLM_MODEL = "gpt-oss-120b"


def _llm_cost_usd(model: str = DEFAULT_LLM_MODEL) -> float:
    """Per-call USD cost using token-average heuristics."""
    rates = MODEL_COSTS_USD_PER_1M[model]
    return (
        LLM_PROMPT_TOKENS_AVG / 1_000_000 * rates["prompt"]
        + LLM_RESPONSE_TOKENS_AVG / 1_000_000 * rates["response"]
    )


def run_cascade_on_masked(
    masked_df: pd.DataFrame,
    embeddings: np.ndarray,
    attrs: list[str],
    category_model_prefix: str,
    *,
    regex_category: str = "pasta",
    enable_llm: bool = False,
    llm_schema: Optional[dict] = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_backend: str = "openrouter",
) -> pd.DataFrame:
    """Run regex → ML → Bayes (+ optional LLM) on `masked_df`.

    Returns a long-format DataFrame: (code, attr, cascade_pred, cascade_layer,
    cascade_conf, llm_called, cost_tokens). `embeddings` must be aligned
    row-by-row with `masked_df`.
    """
    assert len(embeddings) == len(masked_df), "embeddings/df length mismatch"

    rx = RegexExtractor()
    ml_models = load_ml_models(category_model_prefix, attrs)
    bayes_model, bayes_inf = load_bayesian(category_model_prefix)
    thresholds = load_thresholds(category_model_prefix)

    llm_enrich = None
    if enable_llm:
        from src.pipeline.llm_fallback.enrich import enrich_product
        if llm_schema is None:
            raise ValueError("enable_llm=True requires llm_schema")
        def llm_enrich(product: dict) -> dict:  # noqa: E306
            return enrich_product(product, llm_schema, backend=llm_backend, model=llm_model)

    rows = []
    total = len(masked_df)
    logger.info("cascade_sim: starting loop on %d rows, enable_llm=%s", total, enable_llm)
    for i, (_, row) in enumerate(masked_df.iterrows()):
        if i % 10 == 0:
            logger.info("cascade_sim: %d/%d (%.0f%%)", i, total, i / total * 100)
        extracted: dict = {}
        # Layer 1 — regex
        for a, (v, c) in regex_layer(row, rx, category=regex_category).items():
            if a in attrs and a not in extracted:
                extracted[a] = (v, c, "regex")
        # Layer 2 — ML
        for a, (v, c) in ml_layer(embeddings, i, ml_models, attrs, thresholds).items():
            if a not in extracted:
                extracted[a] = (v, c, "ml")
        # Layer 3 — Bayes
        for a, (v, c) in bayes_layer(
            row, bayes_model, bayes_inf, attrs,
            ml_predictions=extracted, thresholds=thresholds,
        ).items():
            if a not in extracted:
                extracted[a] = (v, c, "bayes")
        # Layer 4 — LLM (only if any attr still uncovered)
        llm_called = False
        cost_tokens = 0
        if llm_enrich is not None:
            uncovered = [a for a in attrs if a not in extracted]
            if uncovered:
                product = row.to_dict()
                try:
                    llm_out = llm_enrich(product)
                    llm_called = True
                    cost_tokens = LLM_TOTAL_TOKENS_AVG
                    for a in uncovered:
                        if a in llm_out and llm_out[a] not in (None, ""):
                            extracted[a] = (llm_out[a], 1.0, "llm")
                except Exception as e:  # noqa: BLE001
                    logger.warning("LLM enrich failed for %s: %s", row.get("code"), e)
        code = str(row.get("code"))
        for a in attrs:
            if a in extracted:
                v, c, layer = extracted[a]
            else:
                v, c, layer = None, 0.0, "none"
            rows.append({
                "code": code,
                "attr": a,
                "cascade_pred": None if v is None else str(v),
                "cascade_conf": float(c),
                "cascade_layer": layer,
                "llm_called": llm_called,
                "cost_tokens": cost_tokens if a == attrs[0] else 0,  # charge once per row
            })
    return pd.DataFrame(rows)
