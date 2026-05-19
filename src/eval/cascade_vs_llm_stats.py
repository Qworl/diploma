"""
Formal statistical tests for §6.11 — cascade vs direct LLM.

Closes review remark §3.2: «§6.11 уже сравнивает Δ, но без формальных тестов».

Per (category, attr):
1. McNemar's χ² with continuity correction — paired test on same products.
2. Paired bootstrap 95% CI on Δ accuracy = cascade − direct_llm.

Inputs:
- datasets/processed/experiment_per_product_<cat>_stratified.parquet (config=regex_ml_bayes)
- datasets/processed/direct_llm_eval_<cat>_stratified.parquet

Output:
- datasets/processed/cascade_vs_llm_stats.parquet

Usage:
    python scripts/eval_cascade_vs_llm_stats.py --all
    python scripts/eval_cascade_vs_llm_stats.py --category pasta_stratified
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from src.common import PROCESSED_DIR, RANDOM_STATE, setup_logging

logger = logging.getLogger(__name__)

# Все 6 main доменов имеют direct_llm_eval_*.parquet после eval_direct_llm_baseline --all.
CATEGORIES_WITH_DIRECT_LLM = [
    "pasta_stratified",
    "chocolate_stratified",
    "beverages_stratified",
    "cheeses_stratified",
    "cereals_stratified",
    "cosmetics_stratified",
]


def _load_paired(category: str) -> pd.DataFrame:
    """Merge cascade (regex_ml_bayes) и direct LLM на (code, attr).

    Returns: DataFrame [code, attr, gt, cascade_pred, llm_pred,
                        cascade_correct, llm_correct]
    Только rows where хотя бы одна из систем дала predict (pred != None) И gt известен.
    """
    cascade_path = os.path.join(PROCESSED_DIR,
                                f"experiment_per_product_{category}.parquet")
    llm_path = os.path.join(PROCESSED_DIR,
                             f"direct_llm_eval_{category}.parquet")
    if not os.path.exists(cascade_path) or not os.path.exists(llm_path):
        return pd.DataFrame()

    cas = pd.read_parquet(cascade_path)
    cas = cas[cas["config"] == "regex_ml_bayes"][["code", "attr", "gt", "pred"]].copy()
    cas = cas.rename(columns={"pred": "cascade_pred"})

    llm = pd.read_parquet(llm_path)
    llm = llm[["code", "attr", "pred"]].copy()
    llm = llm.rename(columns={"pred": "llm_pred"})

    # Normalise dtypes for safe merge
    cas["code"] = cas["code"].astype(str)
    llm["code"] = llm["code"].astype(str)

    merged = cas.merge(llm, on=["code", "attr"], how="inner")
    # Drop rows where gt unknown
    merged = merged[merged["gt"].notna() & (merged["gt"].astype(str) != "None")].copy()

    # Normalise prediction comparison: cast to str, handle NaN/None
    def _norm(v):
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        s = str(v)
        return None if s.lower() in ("nan", "none", "") else s

    merged["gt_str"] = merged["gt"].apply(_norm)
    merged["cascade_str"] = merged["cascade_pred"].apply(_norm)
    merged["llm_str"] = merged["llm_pred"].apply(_norm)

    # cascade_correct: cascade выдал предсказание И оно совпадает с gt
    merged["cascade_correct"] = (merged["cascade_str"].notna() &
                                  (merged["cascade_str"] == merged["gt_str"])).astype(int)
    merged["llm_correct"] = (merged["llm_str"].notna() &
                              (merged["llm_str"] == merged["gt_str"])).astype(int)

    return merged


def mcnemar_test(b: int, c: int) -> tuple[float, float]:
    """McNemar's χ² with continuity correction.

    b = cascade_correct & llm_wrong, c = cascade_wrong & llm_correct.
    Returns (chi2, p_value). Если b+c < 25 — exact binomial вместо χ²
    (более корректный для малых выборок).
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    if n < 25:
        # Exact binomial test: H0 — равная вероятность b и c
        from scipy.stats import binomtest
        result = binomtest(min(b, c), n, p=0.5, alternative="two-sided")
        return float("nan"), float(result.pvalue)
    chi2 = (abs(b - c) - 1) ** 2 / n
    from scipy.stats import chi2 as chi2_dist
    p = 1.0 - chi2_dist.cdf(chi2, df=1)
    return float(chi2), float(p)


def paired_bootstrap_ci(cascade: np.ndarray, llm: np.ndarray,
                         n_iter: int = 10_000, alpha: float = 0.05,
                         seed: int = RANDOM_STATE) -> tuple[float, float, float]:
    """Paired bootstrap CI for Δ = mean(cascade) − mean(llm).

    Resampling pairs (cascade[i], llm[i]) с возвращением — сохраняется
    парная структура (correlated correctness).
    """
    n = len(cascade)
    if n == 0:
        return 0.0, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    delta = cascade.mean() - llm.mean()
    deltas = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        deltas[i] = cascade[idx].mean() - llm[idx].mean()
    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return float(delta), lo, hi


def analyze(category: str) -> pd.DataFrame:
    paired = _load_paired(category)
    if paired.empty:
        logger.warning("[%s] no paired data — skipping", category)
        return pd.DataFrame()
    logger.info("[%s] %d paired (code, attr) rows", category, len(paired))

    rows = []
    for attr, group in paired.groupby("attr"):
        cas = group["cascade_correct"].values
        llm = group["llm_correct"].values
        n = len(cas)
        # Contingency
        b = int(((cas == 1) & (llm == 0)).sum())
        c = int(((cas == 0) & (llm == 1)).sum())
        a = int(((cas == 1) & (llm == 1)).sum())
        d = int(((cas == 0) & (llm == 0)).sum())

        chi2, p = mcnemar_test(b, c)
        delta, ci_lo, ci_hi = paired_bootstrap_ci(cas, llm)

        rows.append({
            "category": category.replace("_stratified", ""),
            "attr": attr,
            "n_paired": n,
            "cascade_acc": float(cas.mean()),
            "llm_acc": float(llm.mean()),
            "delta_pp": float(delta * 100),
            "ci_lo_pp": float(ci_lo * 100),
            "ci_hi_pp": float(ci_hi * 100),
            "n_cascade_only_correct": b,
            "n_llm_only_correct": c,
            "n_both_correct": a,
            "n_both_wrong": d,
            "mcnemar_chi2": chi2,
            "mcnemar_p": p,
            "significant_at_0.05": p < 0.05,
        })
    return pd.DataFrame(rows)


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--category",
                   choices=CATEGORIES_WITH_DIRECT_LLM)
    p.add_argument("--all", action="store_true",
                   help="Прогнать все категории c готовым direct_llm_eval")
    args = p.parse_args()

    if args.all:
        cats = CATEGORIES_WITH_DIRECT_LLM
    elif args.category:
        cats = [args.category]
    else:
        p.error("укажите --category или --all")

    all_rows = []
    for cat in cats:
        df = analyze(cat)
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        logger.warning("Empty result.")
        return

    full = pd.concat(all_rows, ignore_index=True)
    out = os.path.join(PROCESSED_DIR, "cascade_vs_llm_stats.parquet")
    full.to_parquet(out, index=False)
    logger.info("Saved -> %s", out)

    # Summary
    show = full.copy()
    show["delta_str"] = show.apply(
        lambda r: f"{r.delta_pp:+5.1f} [{r.ci_lo_pp:+5.1f}, {r.ci_hi_pp:+5.1f}]", axis=1)
    show["p_str"] = show["mcnemar_p"].apply(
        lambda p: f"{p:.4f}" if p < 0.001 else f"{p:.3f}")
    show["sig"] = show["significant_at_0.05"].apply(lambda b: "✔" if b else "—")
    cols = ["category", "attr", "n_paired", "cascade_acc", "llm_acc",
            "delta_str", "p_str", "sig"]
    show_print = show[cols].copy()
    show_print["cascade_acc"] = show_print["cascade_acc"].apply(lambda v: f"{v*100:.1f}%")
    show_print["llm_acc"] = show_print["llm_acc"].apply(lambda v: f"{v*100:.1f}%")
    show_print.columns = ["category", "attr", "n", "cascade", "llm",
                          "Δ pp [95% CI]", "McNemar p", "p<0.05"]
    logger.info("\n%s", show_print.to_string(index=False))


if __name__ == "__main__":
    main()
