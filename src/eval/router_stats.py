"""
Statistical tests for cost-aware router vs baselines.

- McNemar's test (correlated proportions) for paired predictions
- Paired bootstrap for CI on accuracy difference
- Dominance fraction over a budget sweep
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2

logger = logging.getLogger(__name__)


def routing_to_pred(
    df: pd.DataFrame, decisions: np.ndarray
) -> np.ndarray:
    """Apply routing decisions → array of correctness per row (1/0)."""
    final = np.where(decisions, df["llm_pred"].astype(str).values,
                     df["cascade_pred"].astype(str).values)
    return (final == df["silver_gt"].astype(str).values).astype(int)


def mcnemar_test(
    a_correct: np.ndarray, b_correct: np.ndarray, continuity_correction: bool = True
) -> float:
    """McNemar's test: p-value for H0 = no difference in paired binary outcomes."""
    if len(a_correct) != len(b_correct):
        raise ValueError("Length mismatch")
    b = int(((a_correct == 1) & (b_correct == 0)).sum())
    c = int(((a_correct == 0) & (b_correct == 1)).sum())
    if b + c == 0:
        return 1.0
    if continuity_correction:
        chi_sq = (max(abs(b - c) - 1, 0)) ** 2 / (b + c)
    else:
        chi_sq = (b - c) ** 2 / (b + c)
    return float(1.0 - chi2.cdf(chi_sq, df=1))


def paired_bootstrap_delta(
    a_correct: np.ndarray,
    b_correct: np.ndarray,
    n_iter: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI for Δ accuracy (A - B), paired by index."""
    if len(a_correct) != len(b_correct):
        raise ValueError("Length mismatch")
    n = len(a_correct)
    rng = np.random.default_rng(seed)
    diffs = a_correct.astype(float) - b_correct.astype(float)
    boots = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        boots[i] = diffs[idx].mean()
    alpha = (1 - ci) / 2
    return float(diffs.mean()), float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


def compute_router_vs_static_at_budgets(
    test_df: pd.DataFrame,
    p_cascade_correct: np.ndarray,
    budgets: Iterable[float] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """For each target budget, find the threshold τ for each strategy that achieves it,
    then compute paired McNemar + bootstrap delta for router vs static_threshold."""
    rows = []
    for budget in budgets:
        # Router: pick τ so that fraction routed equals budget (approx).
        sorted_p = np.sort(p_cascade_correct)
        target_idx = int(budget * len(sorted_p))
        router_tau = sorted_p[target_idx] if target_idx < len(sorted_p) else 1.0
        router_dec = p_cascade_correct < router_tau

        # Static threshold: pick τ so that fraction with conf < τ equals budget.
        conf = test_df["cascade_conf"].values
        sorted_c = np.sort(conf)
        target_idx = int(budget * len(sorted_c))
        static_tau = sorted_c[target_idx] if target_idx < len(sorted_c) else 1.0
        static_dec = conf < static_tau

        router_correct = routing_to_pred(test_df, router_dec)
        static_correct = routing_to_pred(test_df, static_dec)
        delta, ci_lo, ci_hi = paired_bootstrap_delta(
            router_correct, static_correct, n_iter=n_bootstrap, seed=seed
        )
        p_mcnemar = mcnemar_test(router_correct, static_correct)
        rows.append({
            "budget_target": float(budget),
            "router_cost": float(router_dec.mean()),
            "static_cost": float(static_dec.mean()),
            "router_accuracy": float(router_correct.mean()),
            "static_accuracy": float(static_correct.mean()),
            "delta": delta,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "router_strictly_better": bool(ci_lo > 0),
            "p_mcnemar": p_mcnemar,
        })
    return pd.DataFrame(rows)


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

    p = argparse.ArgumentParser(description="Statistical tests for cost-aware router vs baselines")
    p.add_argument("--llm-suffix", default="",
                   help="Suffix for alternative LLM predictions, e.g. 'gptoss' reads "
                        "direct_llm_eval_{cat}_stratified_gptoss.parquet. "
                        "Output: router_stats_{suffix}.parquet")
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

    _, _, test = by_product_split(df)

    with open(os.path.join(MODELS_DIR, "router_xgb.pkl"), "rb") as f:
        bundle = pickle.load(f)
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
    raw = bundle["model"].predict_proba(X_test)[:, 1]
    p_correct = bundle["calibrator"].predict(raw)

    stats = compute_router_vs_static_at_budgets(test, p_correct)
    out = os.path.join(PROCESSED_DIR, f"router_stats{suffix}.parquet")
    stats.to_parquet(out, index=False)

    n_dom = stats["router_strictly_better"].sum()
    n_total = len(stats)
    frac_dom = n_dom / n_total
    logger.info("Router strictly dominates static threshold on %d/%d budgets (%.0f%%)",
                 n_dom, n_total, frac_dom * 100)
    if frac_dom >= 0.5:
        logger.info("GATE 2 PASSED — full 'cost-aware routing' narrative")
    else:
        logger.warning("GATE 2 NOT PASSED — fallback narrative needed")

    print()
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
