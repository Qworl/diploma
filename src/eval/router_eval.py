"""
Pareto-curve evaluation for cost-aware router and baselines.

For each routing strategy, sweep a budget control parameter and record:
    (strategy, threshold_or_budget, cost, accuracy)

Cost = fraction of rows sent to LLM
Accuracy = end-to-end correctness (cascade-pred where route=False, llm-pred otherwise)
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from src.pipeline.router.baselines import (
    static_confidence_threshold,
    per_attr_static_table,
    random_router,
)

logger = logging.getLogger(__name__)


def compute_routed_accuracy(
    df: pd.DataFrame,
    route_to_llm: np.ndarray,
) -> tuple[float, float]:
    """Apply routing decisions, compute final accuracy + cost (LLM-fraction)."""
    if len(df) != len(route_to_llm):
        raise ValueError("Length mismatch")
    final_pred = np.where(route_to_llm, df["llm_pred"].astype(str),
                          df["cascade_pred"].astype(str))
    silver = df["silver_gt"].astype(str)
    correct = (final_pred == silver)
    return float(correct.mean()), float(route_to_llm.mean())


def pareto_curve_router(
    df: pd.DataFrame,
    p_cascade_correct: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Router: send to LLM iff p_cascade_correct < τ."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 21)
    rows = []
    for tau in thresholds:
        decisions = p_cascade_correct < tau
        acc, cost = compute_routed_accuracy(df, decisions)
        rows.append({
            "strategy": "router", "threshold": float(tau),
            "cost": cost, "accuracy": acc,
        })
    return pd.DataFrame(rows)


def pareto_curve_static(
    df: pd.DataFrame,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Static confidence threshold baseline."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 21)
    rows = []
    for tau in thresholds:
        decisions = static_confidence_threshold(df, tau)
        acc, cost = compute_routed_accuracy(df, decisions)
        rows.append({
            "strategy": "static_threshold", "threshold": float(tau),
            "cost": cost, "accuracy": acc,
        })
    return pd.DataFrame(rows)


def pareto_curve_per_attr_table(
    df: pd.DataFrame, table: dict,
) -> pd.DataFrame:
    """Per-attr static table — single decision, not a curve. Returns one row."""
    decisions = per_attr_static_table(df, table)
    acc, cost = compute_routed_accuracy(df, decisions)
    return pd.DataFrame([{
        "strategy": "per_attr_table", "threshold": float("nan"),
        "cost": cost, "accuracy": acc,
    }])


def pareto_curve_random(
    df: pd.DataFrame,
    budgets: Iterable[float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Random router at varied LLM budgets."""
    if budgets is None:
        budgets = np.linspace(0, 1, 21)
    rows = []
    for b in budgets:
        decisions = random_router(df, llm_budget=b, seed=seed)
        acc, cost = compute_routed_accuracy(df, decisions)
        rows.append({
            "strategy": "random", "threshold": float(b),
            "cost": cost, "accuracy": acc,
        })
    return pd.DataFrame(rows)


def evaluate_all_on_test(
    test_df: pd.DataFrame,
    p_cascade_correct: np.ndarray,
    train_df_for_table: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all strategies into a single long-form Pareto dataframe."""
    from src.pipeline.router.baselines import build_per_attr_table

    train_with_llm = train_df_for_table.copy()
    train_with_llm["llm_correct"] = (
        train_with_llm["llm_pred"].astype(str) == train_with_llm["silver_gt"].astype(str)
    ).astype(int)
    table = build_per_attr_table(train_with_llm)

    parts = [
        pareto_curve_router(test_df, p_cascade_correct),
        pareto_curve_static(test_df),
        pareto_curve_per_attr_table(test_df, table),
        pareto_curve_random(test_df),
    ]
    pure_cascade = pd.DataFrame([{
        "strategy": "cascade_only", "threshold": 0.0,
        "cost": 0.0,
        "accuracy": float((test_df["cascade_pred"].astype(str) == test_df["silver_gt"].astype(str)).mean()),
    }])
    pure_llm = pd.DataFrame([{
        "strategy": "all_llm", "threshold": 1.0,
        "cost": 1.0,
        "accuracy": float((test_df["llm_pred"].astype(str) == test_df["silver_gt"].astype(str)).mean()),
    }])
    return pd.concat(parts + [pure_cascade, pure_llm], ignore_index=True)


def main():
    import argparse
    import os
    import pickle
    import json
    from src.common import MODELS_DIR, PROCESSED_DIR, setup_logging
    from src.pipeline.router.data import (
        FOOD_CATS, build_training_dataset, by_product_split,
    )
    from src.pipeline.router.train import _enrich_with_product_meta
    from src.pipeline.router.features import featurize

    p = argparse.ArgumentParser(description="Pareto-curve evaluation for cost-aware router")
    p.add_argument("--llm-suffix", default="",
                   help="Suffix for alternative LLM predictions, e.g. 'gptoss' reads "
                        "direct_llm_eval_{cat}_stratified_gptoss.parquet. "
                        "Output: router_pareto_{suffix}.parquet")
    args = p.parse_args()

    setup_logging()

    suffix = f"_{args.llm_suffix}" if args.llm_suffix else ""

    if args.llm_suffix:
        # Re-join on the fly using alternative LLM predictions
        logger.info("Loading router data with llm_suffix=%r", args.llm_suffix)
        df = build_training_dataset(FOOD_CATS, PROCESSED_DIR, llm_suffix=args.llm_suffix)
        df = _enrich_with_product_meta(df, PROCESSED_DIR)
    else:
        df = pd.read_parquet(os.path.join(PROCESSED_DIR, "router_train.parquet"))
        df = _enrich_with_product_meta(df, PROCESSED_DIR)

    train, val, test = by_product_split(df)
    logger.info("Test size: %d rows, %d products", len(test), test["code"].nunique())

    with open(os.path.join(MODELS_DIR, "router_xgb.pkl"), "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    calibrator = bundle["calibrator"]

    with open(os.path.join(MODELS_DIR, "router_meta.json")) as f:
        meta = json.load(f)
    brand_set = set(meta["brand_set"])
    # Reconstruct lookup tables from meta (keys stored as lists for JSON compat)
    class_freq_table = {tuple(e["key"]): e["value"]
                        for e in meta.get("class_freq_table", [])}
    brand_attr_acc_table = {tuple(e["key"]): e["value"]
                            for e in meta.get("brand_attr_acc_table", [])}

    X_test, _ = featurize(test, brand_set=brand_set,
                          class_freq_table=class_freq_table,
                          brand_attr_acc_table=brand_attr_acc_table)
    raw = model.predict_proba(X_test)[:, 1]
    p_correct = calibrator.predict(raw)

    pareto = evaluate_all_on_test(test, p_correct, train)
    out = os.path.join(PROCESSED_DIR, f"router_pareto{suffix}.parquet")
    pareto.to_parquet(out, index=False)
    logger.info("Saved %s (%d rows)", out, len(pareto))

    logger.info("Anchor points:")
    for _, r in pareto[pareto["strategy"].isin(["cascade_only", "all_llm"])].iterrows():
        logger.info("  %s: cost=%.0f%% acc=%.3f", r["strategy"], r["cost"] * 100, r["accuracy"])


if __name__ == "__main__":
    main()
