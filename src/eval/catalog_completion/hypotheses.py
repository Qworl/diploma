"""Bonferroni-corrected pre-registered hypothesis tests for Trek A2.

H1: pasta coverage gain (pp) >= 30 at static-policy LLM cost.
H2: pasta recovery accuracy >= 75% on cells where cascade filled a masked attr.
H3: electronics coverage gain <= half of pasta coverage gain (ceiling claim).

Each test reports {decision, p_value, threshold, ci, alpha}. α = 0.05 / 3
(Bonferroni). All three tests use scipy.stats.binomtest (one-sided).
"""
from __future__ import annotations

from typing import Any

from scipy.stats import binomtest

BONFERRONI_ALPHA = 0.05 / 3  # ≈ 0.01667

# Thresholds from spec
H1_COVERAGE_THRESHOLD_PP = 30.0   # percentage points
H2_RECOVERY_THRESHOLD = 0.75
H3_RATIO_THRESHOLD = 0.5          # electronics gain <= 0.5 * pasta gain


def _decision(p_value: float, alpha: float = BONFERRONI_ALPHA) -> str:
    return "REJECT_H0" if p_value < alpha else "FAIL_TO_REJECT"


def evaluate_h1_coverage_gain(coverage_gain_cells: int, n_cells: int) -> dict[str, Any]:
    """H1: cells filled / total cells >= 30%.
    H0: coverage gain rate <= 0.30. One-sided greater binomial.
    """
    if n_cells == 0:
        return {"decision": "SKIPPED_INSUFFICIENT_DATA"}
    res = binomtest(coverage_gain_cells, n_cells,
                    p=H1_COVERAGE_THRESHOLD_PP / 100, alternative="greater")
    return {
        "name": "H1_pasta_coverage_gain_geq_30pp",
        "n": n_cells,
        "successes": coverage_gain_cells,
        "observed_rate_pp": coverage_gain_cells / n_cells * 100,
        "threshold_pp": H1_COVERAGE_THRESHOLD_PP,
        "p_value": float(res.pvalue),
        "alpha": BONFERRONI_ALPHA,
        "decision": _decision(res.pvalue),
    }


def evaluate_h2_recovery_accuracy(n_correct: int, n_filled: int) -> dict[str, Any]:
    """H2: cascade pred matches ground truth on >=75% of filled-masked cells."""
    if n_filled == 0:
        return {"decision": "SKIPPED_INSUFFICIENT_DATA"}
    res = binomtest(n_correct, n_filled, p=H2_RECOVERY_THRESHOLD, alternative="greater")
    return {
        "name": "H2_pasta_recovery_accuracy_geq_75pct",
        "n": n_filled,
        "successes": n_correct,
        "observed_rate": n_correct / n_filled,
        "threshold": H2_RECOVERY_THRESHOLD,
        "p_value": float(res.pvalue),
        "alpha": BONFERRONI_ALPHA,
        "decision": _decision(res.pvalue),
    }


def evaluate_h3_electronics_ceiling(
    electronics_gain_cells: int,
    electronics_n_cells: int,
    pasta_gain_cells: int,
    pasta_n_cells: int,
    *,
    min_n: int = 30,
) -> dict[str, Any]:
    """H3: electronics gain rate <= 0.5 * pasta gain rate.

    Test: one-sided binomial on electronics with p_null = 0.5 * pasta_rate.
    H0: electronics_rate > p_null. Reject (= confirm ceiling) means
    electronics is significantly below half of pasta.
    """
    if electronics_n_cells < min_n or pasta_n_cells == 0:
        return {"decision": "SKIPPED_INSUFFICIENT_DATA",
                "electronics_n": electronics_n_cells,
                "pasta_n": pasta_n_cells,
                "min_n": min_n}
    pasta_rate = pasta_gain_cells / pasta_n_cells
    p_null = H3_RATIO_THRESHOLD * pasta_rate
    if p_null <= 0:
        return {"decision": "SKIPPED_INSUFFICIENT_DATA",
                "reason": "pasta gain rate is zero"}
    res = binomtest(electronics_gain_cells, electronics_n_cells,
                    p=p_null, alternative="less")
    return {
        "name": "H3_electronics_ceiling_half_of_pasta",
        "electronics_n": electronics_n_cells,
        "electronics_successes": electronics_gain_cells,
        "pasta_gain_rate": pasta_rate,
        "p_null": p_null,
        "p_value": float(res.pvalue),
        "alpha": BONFERRONI_ALPHA,
        "decision": _decision(res.pvalue),
    }


def _load_summary(prefix: str, category: str, tag: str) -> dict | None:
    import json
    import os
    p = f"{prefix}_summary_{category}_{tag}.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main() -> None:
    """Read per-category summary JSONs and evaluate H1-H3.

    Picks the `with_llm` config for pasta H1/H2 (full pipeline) and electronics H3.
    Falls back to `no_llm` when with_llm summary is absent.
    """
    import argparse
    import json
    import logging
    import os

    from src.common import PROCESSED_DIR, setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=os.path.join(PROCESSED_DIR, "catalog_completion"))
    parser.add_argument("--config-tag", default="with_llm",
                        help="Which run config to evaluate (no_llm or with_llm)")
    parser.add_argument("--out",
                        default=os.path.join(PROCESSED_DIR, "catalog_completion_hypotheses.json"))
    args = parser.parse_args()

    pasta = _load_summary(args.prefix, "pasta", args.config_tag)
    # Electronics may only have no_llm; fall back gracefully for H3.
    electronics = _load_summary(args.prefix, "electronics", args.config_tag)
    if electronics is None and args.config_tag != "no_llm":
        electronics = _load_summary(args.prefix, "electronics", "no_llm")
        if electronics is not None:
            logger.info("Electronics with_llm not found — using no_llm for H3")

    results: dict[str, Any] = {"alpha": BONFERRONI_ALPHA, "config_tag": args.config_tag}

    if pasta is None:
        results["H1"] = {"decision": "SKIPPED_INSUFFICIENT_DATA",
                         "reason": "no pasta summary"}
        results["H2"] = {"decision": "SKIPPED_INSUFFICIENT_DATA",
                         "reason": "no pasta summary"}
    else:
        results["H1"] = evaluate_h1_coverage_gain(
            pasta["coverage_gain_cells"], pasta["n_cells"])
        results["H2"] = evaluate_h2_recovery_accuracy(
            pasta["recovery_n_correct"], pasta["recovery_n"])

    # H3 compares coverage gain rates: electronics vs pasta. Use no_llm for
    # electronics when with_llm run is unavailable (cost-gated category).
    if pasta is None or electronics is None:
        results["H3"] = {"decision": "SKIPPED_INSUFFICIENT_DATA",
                         "reason": "missing pasta or electronics summary"}
    else:
        results["H3"] = evaluate_h3_electronics_ceiling(
            electronics["coverage_gain_cells"], electronics["n_cells"],
            pasta["coverage_gain_cells"], pasta["n_cells"],
        )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Wrote %s", args.out)
    for hyp in ("H1", "H2", "H3"):
        logger.info("%s: %s", hyp, results[hyp].get("decision"))


if __name__ == "__main__":
    main()
