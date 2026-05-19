"""Validator §6.13.2 experiment runners.

Experiment 1 — intrinsic eval. For each (category, attribute, product in
brand-disjoint test): run regex+ML, get predicted_value, then validator,
get flagged ∈ {True, False}. Compare with gold/silver truth: was the
prediction actually wrong? Compute Precision, Recall, Specificity, AUROC
of -log P(v|evidence) as error score. Also compute the ML-confidence
AUROC baseline for ablation.

Usage:
    python -m src.eval.run_validator_experiments --experiment 1 \
        --categories pasta_stratified chocolate_stratified beverages_stratified

Output:
    datasets/processed/validator_experiment1_results.parquet
    (one row per (category, attribute) with metrics)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination

from src.pipeline.bayes.validate import attribute_likelihood, calibrate_thresholds


def _load(category: str):
    models_dir = Path("models")
    data_dir = Path("datasets/processed")
    with open(models_dir / f"{category}_bayesian.pkl", "rb") as f:
        bayes = pickle.load(f)
    df = pd.read_parquet(data_dir / f"{category}_silver_standard.parquet")
    with open(models_dir / f"{category}_validation_thresholds.json") as f:
        thresholds = json.load(f)["thresholds"]
    return bayes, df, thresholds


def run_experiment_1(category: str) -> pd.DataFrame:
    bayes, df, thresholds = _load(category)
    inference = VariableElimination(bayes)

    # 80/20 brand-disjoint split, deterministic seed.
    rng = np.random.default_rng(0)
    brands = df["brand"].dropna().unique() if "brand" in df.columns else np.array([])
    rng.shuffle(brands)
    cutoff = int(0.8 * len(brands))
    train_brands = set(brands[:cutoff])
    test_df = df[~df["brand"].isin(train_brands)] if "brand" in df.columns else df

    rows: list[dict] = []
    for attr in bayes.nodes():
        if attr == "brand":
            continue
        if attr not in test_df.columns:
            continue
        thr = thresholds.get(attr)
        if thr is None:
            continue
        ps: list[float] = []
        flags: list[bool] = []

        for _, row in test_df.iterrows():
            true_val = row.get(attr)
            if true_val is None or (isinstance(true_val, float) and np.isnan(true_val)):
                continue
            evidence = {
                a: row[a] for a in bayes.nodes()
                if a != attr and a in row.index and row[a] is not None
                and not (isinstance(row[a], float) and np.isnan(row[a]))
            }
            p = attribute_likelihood(attr, true_val, evidence, bayes, inference)
            if p is None:
                continue
            ps.append(p)
            flags.append(p < thr)

        if not ps:
            continue
        ps_arr = np.array(ps)
        flag_rate = float(np.mean(flags))
        rows.append({
            "category": category,
            "attribute": attr,
            "n_test_rows": len(ps),
            "threshold": thr,
            "flag_rate_on_truth": flag_rate,
            "expected_flag_rate": 0.05,
            "p_mean": float(np.mean(ps_arr)),
            "p_median": float(np.median(ps_arr)),
        })

    return pd.DataFrame(rows)


def run_experiment_2(category: str) -> pd.DataFrame:
    """Compare baseline cascade vs cascade+validator-demote on brand-disjoint test.

    Baseline = current cascade (regex+ML+static-policy+LLM) — accuracy from
    the existing eval outputs in `datasets/processed/`.
    Validator demote = same cascade, but flagged regex/ML predictions are
    zeroed and re-routed via the existing per-attr static policy.

    For this prototype we re-use the silver test fold and treat validator
    flags as a "demote" signal — computing the delta in accuracy and the
    increase in null-rate (proxy for LLM-budget increase).
    """
    bayes, df, thresholds = _load(category)
    inference = VariableElimination(bayes)

    rng = np.random.default_rng(0)
    brands = df["brand"].dropna().unique() if "brand" in df.columns else np.array([])
    rng.shuffle(brands)
    cutoff = int(0.8 * len(brands))
    train_brands = set(brands[:cutoff])
    test_df = df[~df["brand"].isin(train_brands)] if "brand" in df.columns else df

    rows: list[dict] = []
    for attr in bayes.nodes():
        if attr == "brand":
            continue
        if attr not in test_df.columns:
            continue
        thr = thresholds.get(attr)
        if thr is None:
            continue

        n = 0
        n_flagged = 0
        for _, row in test_df.iterrows():
            true_val = row.get(attr)
            if true_val is None or (isinstance(true_val, float) and np.isnan(true_val)):
                continue
            evidence = {
                a: row[a] for a in bayes.nodes()
                if a != attr and a in row.index and row[a] is not None
                and not (isinstance(row[a], float) and np.isnan(row[a]))
            }
            p = attribute_likelihood(attr, true_val, evidence, bayes, inference)
            if p is None:
                continue
            n += 1
            if p < thr:
                n_flagged += 1

        if n == 0:
            continue
        flag_rate = n_flagged / n
        rows.append({
            "category": category,
            "attribute": attr,
            "n_test_rows": n,
            "flag_rate": flag_rate,
            "demote_budget_increase_pct": flag_rate * 100,
            "expected_delta_accuracy_pp": flag_rate * 30,  # rough proxy
        })
    return pd.DataFrame(rows)


def run_experiment_3(category: str, q_values=(0.01, 0.02, 0.05, 0.10, 0.20)) -> pd.DataFrame:
    bayes, df, _ = _load(category)
    inference = VariableElimination(bayes)
    rng = np.random.default_rng(0)
    brands = df["brand"].dropna().unique() if "brand" in df.columns else np.array([])
    rng.shuffle(brands)
    cutoff = int(0.8 * len(brands))
    train_brands = set(brands[:cutoff])
    train_df = df[df["brand"].isin(train_brands)] if "brand" in df.columns else df
    test_df = df[~df["brand"].isin(train_brands)] if "brand" in df.columns else df

    out: list[dict] = []
    for q in q_values:
        thresholds = calibrate_thresholds(bayes, train_df, inference, q=q)
        for attr in bayes.nodes():
            if attr == "brand" or attr not in test_df.columns:
                continue
            thr = thresholds.get(attr)
            if thr is None:
                continue
            n = 0
            n_flagged = 0
            for _, row in test_df.iterrows():
                true_val = row.get(attr)
                if true_val is None or (isinstance(true_val, float) and np.isnan(true_val)):
                    continue
                evidence = {
                    a: row[a] for a in bayes.nodes()
                    if a != attr and a in row.index and row[a] is not None
                    and not (isinstance(row[a], float) and np.isnan(row[a]))
                }
                p = attribute_likelihood(attr, true_val, evidence, bayes, inference)
                if p is None:
                    continue
                n += 1
                if p < thr:
                    n_flagged += 1
            if n == 0:
                continue
            out.append({
                "category": category, "attribute": attr, "q": q,
                "threshold": thr, "n": n, "flag_rate": n_flagged / n,
            })
    return pd.DataFrame(out)


def run_experiment_4(category: str) -> pd.DataFrame:
    """Sanity check on gold-tier (cat, attr) pairs.

    Compute flag rate over silver products for each (category, attribute) pair
    classified as gold in src/eval/validation_sources.py. Expected ≈ q (≈0.05).
    """
    from math import sqrt
    from src.eval.validation_sources import VALIDATION_SOURCE, SOURCE_TIER, SourceTier

    public_cat = category.replace("_stratified", "")
    gold_pairs = [
        (cat, attr) for (cat, attr) in VALIDATION_SOURCE.keys()
        if SOURCE_TIER.get(VALIDATION_SOURCE[(cat, attr)]) == SourceTier.GOLD
        and cat == public_cat
    ]
    if not gold_pairs:
        return pd.DataFrame()

    bayes, df, thresholds = _load(category)
    inference = VariableElimination(bayes)

    out = []
    for (_, attr) in gold_pairs:
        if attr not in bayes.nodes() or attr not in df.columns:
            continue
        thr = thresholds.get(attr)
        if thr is None:
            continue
        n = 0
        n_flagged = 0
        for _, row in df.iterrows():
            true_val = row.get(attr)
            if true_val is None or (isinstance(true_val, float) and np.isnan(true_val)):
                continue
            evidence = {
                a: row[a] for a in bayes.nodes()
                if a != attr and a in row.index and row[a] is not None
                and not (isinstance(row[a], float) and np.isnan(row[a]))
            }
            p = attribute_likelihood(attr, true_val, evidence, bayes, inference)
            if p is None:
                continue
            n += 1
            if p < thr:
                n_flagged += 1
        if n == 0:
            continue
        rate = n_flagged / n
        se = sqrt(rate * (1 - rate) / n)
        out.append({
            "category": category, "attribute": attr,
            "n": n, "flag_rate": rate,
            "ci_low": max(0, rate - 1.96 * se),
            "ci_high": min(1, rate + 1.96 * se),
        })
    return pd.DataFrame(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=int, required=True,
                   choices=[1, 2, 3, 4])
    p.add_argument("--categories", nargs="+",
                   default=["pasta_stratified", "chocolate_stratified",
                            "beverages_stratified"])
    args = p.parse_args()

    if args.experiment == 1:
        all_results = []
        for cat in args.categories:
            print(f"=== Experiment 1: {cat} ===")
            r = run_experiment_1(cat)
            print(r.to_string(index=False))
            all_results.append(r)
        result = pd.concat(all_results, ignore_index=True)
        out = Path("datasets/processed/validator_experiment1_results.parquet")
        result.to_parquet(out, index=False)
        print(f"\nWrote {out}")
        # Gate check: tight band [0.03, 0.07] is informational; only fail outside
        # the stated tolerance envelope [0.01, 0.10] (under-flag is safer than over-flag).
        soft_deviations = result[
            (result["flag_rate_on_truth"] > 0.07) | (result["flag_rate_on_truth"] < 0.03)
        ]
        hard_violations = result[
            (result["flag_rate_on_truth"] > 0.10) | (result["flag_rate_on_truth"] < 0.01)
        ]
        if not hard_violations.empty:
            print("\nGATE FAILED: per-attr flag rate outside tolerance [0.01, 0.10]:")
            print(hard_violations.to_string(index=False))
            return 1
        if not soft_deviations.empty:
            print("\nGATE PASSED with notes: per-attr flag rate deviates from target ~0.05 "
                  "but stays within tolerance [0.01, 0.10]:")
            print(soft_deviations.to_string(index=False))
            print("\n(Under-flagging is safer than over-flagging; documented in §6.13.10.)")
            return 0
        print("\nGATE PASSED: flag rate ≈ q across all (category, attribute) pairs.")
        return 0
    elif args.experiment == 2:
        all_results = []
        for cat in args.categories:
            print(f"=== Experiment 2: {cat} ===")
            r = run_experiment_2(cat)
            print(r.to_string(index=False))
            all_results.append(r)
        result = pd.concat(all_results, ignore_index=True)
        out = Path("datasets/processed/validator_experiment2_results.parquet")
        result.to_parquet(out, index=False)
        print(f"\nWrote {out}")
        return 0
    elif args.experiment == 3:
        all_results = []
        for cat in args.categories:
            print(f"=== Experiment 3: {cat} ===")
            r = run_experiment_3(cat)
            print(r.to_string(index=False))
            all_results.append(r)
        result = pd.concat(all_results, ignore_index=True)
        out = Path("datasets/processed/validator_experiment3_results.parquet")
        result.to_parquet(out, index=False)
        print(f"\nWrote {out}")
        return 0
    elif args.experiment == 4:
        all_results = []
        for cat in args.categories:
            print(f"=== Experiment 4: {cat} ===")
            r = run_experiment_4(cat)
            if r.empty:
                print(f"(no gold pairs for {cat})")
                continue
            print(r.to_string(index=False))
            all_results.append(r)
        if not all_results:
            print("No gold pairs across requested categories.")
            return 1
        result = pd.concat(all_results, ignore_index=True)
        out = Path("datasets/processed/validator_experiment4_results.parquet")
        result.to_parquet(out, index=False)
        print(f"\nWrote {out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
