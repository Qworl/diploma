"""
Benchmark cheap LLM candidates for Layer 4 production fallback.

Compares 4-5 cheap models on a stratified test sample per category.
Reports per-attribute accuracy vs silver standard, latency, $-cost.

Picks winner: highest avg accuracy among models with cost ≤ $5/50k calls.

Usage:
    python -m src.diagnostics.silver.llm_benchmark --category pasta --n 50
    python -m src.diagnostics.silver.llm_benchmark --category chocolate --n 50
    python -m src.diagnostics.silver.llm_benchmark --category beverages --n 50
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import PROCESSED_DIR, TEST_SIZE, RANDOM_STATE, setup_logging
from src.pipeline.schemas import (
    PASTA_SCHEMA, CHOCOLATE_SCHEMA, BEVERAGE_SCHEMA
)
from src.pipeline.llm_fallback import enrich_batch

logger = logging.getLogger(__name__)

CATEGORY_SCHEMAS = {
    "pasta": PASTA_SCHEMA,
    "chocolate": CHOCOLATE_SCHEMA,
    "beverages": BEVERAGE_SCHEMA,
}

CATEGORY_FILES = {
    "pasta": "pasta_silver_standard.parquet",
    "chocolate": "chocolate_silver_standard.parquet",
    "beverages": "beverages_silver_standard.parquet",
}

# Candidate cheap models on OpenRouter — comparable price tier
CANDIDATE_MODELS = [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash-lite",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2.5-7b-instruct",
]

# Approx cost per 50k calls (assuming ~600 in tokens, ~150 out)
APPROX_COST_50K_USD = {
    "anthropic/claude-haiku-4.5": 25.0,
    "openai/gpt-4o-mini": 8.0,
    "google/gemini-2.5-flash-lite": 5.0,
    "meta-llama/llama-3.1-8b-instruct": 2.5,
    "qwen/qwen-2.5-7b-instruct": 5.0,
}

COST_BUDGET_USD = 5.0


def load_test_sample(category: str, n: int) -> pd.DataFrame:
    schema = CATEGORY_SCHEMAS[category]
    path = os.path.join(PROCESSED_DIR, CATEGORY_FILES[category])
    df = pd.read_parquet(path)
    _, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Prefer products with rich ground truth (>=4 non-null attrs from schema)
    gt_count = test_df[list(schema.keys())].notna().sum(axis=1)
    qualified = test_df[gt_count >= 4].reset_index(drop=True)
    if len(qualified) > n:
        qualified = qualified.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)
    logger.info("Sample: %d products (filtered from %d test set)", len(qualified), len(test_df))
    return qualified


def per_attr_accuracy(predictions: pd.DataFrame, truth: pd.DataFrame, schema: dict) -> dict:
    pred = predictions.set_index("code")
    truth_df = truth.set_index("code")
    out = {}
    for attr in schema:
        if attr not in pred.columns or attr not in truth_df.columns:
            out[attr] = None
            continue
        # Join on code, drop rows where truth is NaN
        common = pred[[attr]].join(truth_df[[attr]], rsuffix="_t").dropna(subset=[f"{attr}_t"])
        if len(common) == 0:
            out[attr] = None
            continue
        agree = (common[attr].astype(str) == common[f"{attr}_t"].astype(str)).mean()
        out[attr] = {"acc": float(agree), "n": int(len(common))}
    return out


def benchmark_model(model: str, sample: pd.DataFrame, schema: dict) -> dict:
    t0 = time.time()
    try:
        predictions = enrich_batch(
            sample, schema, backend="openrouter", model=model, max_workers=10,
        )
    except Exception as e:
        logger.error("Model %s crashed: %s", model, e)
        return {"model": model, "avg_acc": 0.0, "latency_ms": -1, "per_attr": {}}
    elapsed = time.time() - t0
    latency_ms = (elapsed / max(len(sample), 1)) * 1000

    per_attr = per_attr_accuracy(predictions, sample, schema)
    accs = [v["acc"] for v in per_attr.values() if v is not None]
    avg_acc = sum(accs) / len(accs) if accs else 0.0

    return {
        "model": model,
        "avg_acc": avg_acc,
        "latency_ms": latency_ms,
        "cost_50k_usd": APPROX_COST_50K_USD.get(model, 0.0),
        "per_attr": per_attr,
    }


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=list(CATEGORY_SCHEMAS.keys()))
    parser.add_argument("--n", type=int, default=50, help="Sample size")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    schema = CATEGORY_SCHEMAS[args.category]
    sample = load_test_sample(args.category, args.n)

    results = []
    for model in CANDIDATE_MODELS:
        cost = APPROX_COST_50K_USD.get(model, 999)
        if cost > COST_BUDGET_USD * 5:
            logger.info("Skipping %s — too expensive ($%.2f/50k)", model, cost)
            continue
        logger.info("Benchmarking %s ($%.2f/50k)...", model, cost)
        r = benchmark_model(model, sample, schema)
        results.append(r)
        logger.info("  avg_acc=%.3f, latency=%.0fms/call", r["avg_acc"], r["latency_ms"])

    # Sort
    results.sort(key=lambda r: -r["avg_acc"])

    # Pick winner: best accuracy at cost ≤ budget
    eligible = [r for r in results if r["cost_50k_usd"] <= COST_BUDGET_USD]
    winner = max(eligible, key=lambda r: r["avg_acc"]) if eligible else max(results, key=lambda r: r["avg_acc"])

    logger.info("=" * 88)
    logger.info("LLM BENCHMARK — %s (n=%d)", args.category, len(sample))
    logger.info("=" * 88)
    logger.info("%-48s %9s %8s %9s", "Model", "avg_acc", "$/50k", "lat ms")
    for r in results:
        marker = " ← WINNER" if r["model"] == winner["model"] else ""
        logger.info(
            "%-48s %8.1f%% %7.1f$ %8.0f%s",
            r["model"], r["avg_acc"] * 100, r["cost_50k_usd"], r["latency_ms"], marker,
        )

    logger.info(
        ">>> Winner: %s (acc %.1f%%, $%.1f/50k)",
        winner["model"], winner["avg_acc"] * 100, winner["cost_50k_usd"],
    )

    out_path = args.output or os.path.join(
        PROCESSED_DIR, f"benchmark_{args.category}_models.json"
    )
    with open(out_path, "w") as f:
        json.dump({"results": results, "winner": winner["model"]}, f, indent=2, default=str)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
