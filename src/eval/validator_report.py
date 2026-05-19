"""Trek A1 — comparison table + hypothesis report cards.

Consumes ``validator_comparison_pasta.parquet`` (produced by
``src.eval.validator_comparison``) and writes a structured JSON summary
plus a Markdown table printed to stdout for the thesis notebook.

Usage
-----
    python -m src.eval.validator_report \\
        --in  datasets/processed/validator_comparison_pasta.parquet \\
        --out datasets/processed/validator_comparison_pasta_summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.eval.cascade_vs_audited_gold import PASTA_ATTRS
from src.eval.validator_hypothesis_tests import (
    BONFERRONI_ALPHA,
    paired_bootstrap_auc_diff,
    pareto_dominance_vs_static,
)
from src.eval.validator_metrics import (
    auc_for_validator,
    precision_recall_at_k,
    random_baseline,
    static_policy_baseline,
)

logger = logging.getLogger(__name__)

VALIDATORS = ("xgb_uncertainty", "mahalanobis", "layer_disagree", "ece_attr")
STATIC_ROUTE_ATTRS = {"pasta_shape"}  # §6.14.7 Plan B4 winner
EVAL_SLICE_STATUSES = {"override", "manual_only"}


def _filter_eval_slice(df: pd.DataFrame, strict: bool) -> pd.DataFrame:
    out = df[df["has_manual"]].copy()
    if strict:
        out = out[out["status"].isin(EVAL_SLICE_STATUSES)]
    return out.reset_index(drop=True)


def _validator_block(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """AUC + P/R@k for each validator on a DataFrame."""
    out: dict[str, dict[str, Any]] = {}
    for col in VALIDATORS:
        auc = auc_for_validator(df[col], df["is_error"])
        block: dict[str, Any] = {"auc": auc, "n_with_score": int(df[col].notna().sum())}
        for k in (0.05, 0.10, 0.20):
            p, r, n = precision_recall_at_k(df[col], df["is_error"], k=k)
            block[f"precision_at_{int(k*100)}"] = p
            block[f"recall_at_{int(k*100)}"]   = r
            block[f"n_routed_at_{int(k*100)}"] = n
        out[col] = block
    return out


def _per_attr_auc(df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for attr in PASTA_ATTRS:
        sub = df[df["attr"] == attr]
        out[attr] = {
            col: auc_for_validator(sub[col], sub["is_error"])
            for col in VALIDATORS
        }
        out[attr]["n_cells"] = int(len(sub))
        out[attr]["n_errors"] = int(sub["is_error"].sum())
    return out


def _hypothesis_report(df_all: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """H1, H2, H3 verdicts at α = Bonferroni."""
    # H1: Mahalanobis > XGB max-prob on overall AUC (paired bootstrap)
    h1 = paired_bootstrap_auc_diff(
        df_all["mahalanobis"].values, df_all["xgb_uncertainty"].values,
        df_all["is_error"].values, n_boot=1000, seed=11, alternative="greater",
    )
    h1["verdict"] = (
        "accept" if (h1["p_value"] is not None and h1["p_value"] < BONFERRONI_ALPHA)
        else "reject"
    )

    # H2: Layer-disagreement is the single best on pasta_shape AUC. Test:
    # disagreement vs the second-best validator on the pasta_shape slice.
    sub = df_all[df_all["attr"] == "pasta_shape"]
    other_aucs = {
        col: auc_for_validator(sub[col], sub["is_error"]) or float("-inf")
        for col in VALIDATORS if col != "layer_disagree"
    }
    runner_up = max(other_aucs, key=other_aucs.get)
    h2 = paired_bootstrap_auc_diff(
        sub["layer_disagree"].values, sub[runner_up].values,
        sub["is_error"].values, n_boot=1000, seed=22, alternative="greater",
    )
    h2["runner_up"] = runner_up
    h2["verdict"] = (
        "accept" if (h2["p_value"] is not None and h2["p_value"] < BONFERRONI_ALPHA)
        else "reject"
    )

    # H3: No single validator beats static-policy on overall Pareto.
    dominance = pareto_dominance_vs_static(
        df_all, validator_cols=VALIDATORS, static_attrs=STATIC_ROUTE_ATTRS,
    )
    any_beats = any(v["beats_static"] for v in dominance.values())
    h3 = {
        "dominance": dominance,
        "any_beats_static": any_beats,
        # H3 = "no single validator beats baseline"
        "verdict": "accept" if not any_beats else "reject",
    }
    return {"H1": h1, "H2": h2, "H3": h3}


def _print_markdown(block: dict[str, dict[str, Any]], baseline: dict, n_cells: int) -> None:
    print(f"\n### Trek A1 — validator comparison on audited pasta gold (n_cells={n_cells})\n")
    print("| Validator | AUC | P@10% | R@10% | P@20% | R@20% |")
    print("|---|---:|---:|---:|---:|---:|")
    label = {
        "xgb_uncertainty": "XGB 1−max_prob",
        "mahalanobis": "Mahalanobis",
        "layer_disagree": "Layer disagreement",
        "ece_attr": "Per-attr ECE",
    }
    for col in VALIDATORS:
        b = block[col]

        def f(x: Any) -> str:
            return f"{x:.3f}" if isinstance(x, (int, float)) and x is not None else "n/a"

        print(
            f"| {label[col]} | {f(b['auc'])} | {f(b['precision_at_10'])} | "
            f"{f(b['recall_at_10'])} | {f(b['precision_at_20'])} | "
            f"{f(b['recall_at_20'])} |"
        )
    if baseline["precision"] is not None:
        print(
            f"| **Static policy ({sorted(STATIC_ROUTE_ATTRS)})** | n/a | "
            f"{baseline['precision']:.3f} | {baseline['recall']:.3f} | — | — |"
        )
    else:
        print("| **Static policy** | n/a | n/a | n/a | n/a | n/a |")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path",
                        default=os.path.join(PROCESSED_DIR, "validator_comparison_pasta.parquet"))
    parser.add_argument("--out", dest="out_path",
                        default=os.path.join(PROCESSED_DIR, "validator_comparison_pasta_summary.json"))
    parser.add_argument("--strict-slice", action="store_true",
                        help="Restrict to status in {override, manual_only}")
    args = parser.parse_args()

    df_full = pd.read_parquet(args.in_path)
    df = _filter_eval_slice(df_full, strict=args.strict_slice)
    logger.info("Eval slice: %d cells (strict=%s)", len(df), args.strict_slice)

    overall = _validator_block(df)
    by_attr = _per_attr_auc(df)
    baseline = static_policy_baseline(df, attrs_to_route=STATIC_ROUTE_ATTRS)
    rnd_05 = random_baseline(df["is_error"], k=0.05)
    rnd_10 = random_baseline(df["is_error"], k=0.10)
    rnd_20 = random_baseline(df["is_error"], k=0.20)
    hypotheses = _hypothesis_report(df)

    payload = {
        "n_cells": int(len(df)),
        "n_errors": int(df["is_error"].sum()),
        "strict_slice": bool(args.strict_slice),
        "validator_columns": list(VALIDATORS),
        "overall": overall,
        "by_attr": by_attr,
        "static_policy_baseline": baseline,
        "static_route_attrs": sorted(STATIC_ROUTE_ATTRS),
        "random_baseline": {
            "k=0.05": {"precision": rnd_05[0], "recall": rnd_05[1]},
            "k=0.10": {"precision": rnd_10[0], "recall": rnd_10[1]},
            "k=0.20": {"precision": rnd_20[0], "recall": rnd_20[1]},
        },
        "hypotheses": hypotheses,
        "bonferroni_alpha": BONFERRONI_ALPHA,
    }

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote summary → %s", args.out_path)

    _print_markdown(overall, baseline, n_cells=len(df))

    print("\n### Hypothesis verdicts (α = 0.05/3 = {:.4f})".format(BONFERRONI_ALPHA))
    for name, h in hypotheses.items():
        verdict = h["verdict"]
        if name in ("H1", "H2"):
            print(f"- {name}: verdict={verdict}  Δ={h.get('diff')}  p={h.get('p_value')}  n={h.get('n')}")
        else:
            print(f"- {name}: verdict={verdict}  any_beats_static={h['any_beats_static']}")


if __name__ == "__main__":
    main()
