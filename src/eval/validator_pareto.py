"""Trek A1 — Pareto curves (accuracy on unrouted cells vs LLM-cost).

For each validator, sweeps the routing budget k from 1% to 100% in 1%
steps. At each k:
    routed         = top-k cells by validator score
    unrouted       = the rest
    accuracy_on_unrouted = 1 - error_rate_on_unrouted
    recall         = errors_caught / total_errors
    precision      = errors_caught / n_routed

Random baseline and static-policy are also plotted.
"""
from __future__ import annotations

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.eval.validator_metrics import (
    precision_recall_at_k,
    random_baseline,
    static_policy_baseline,
)
from src.eval.validator_report import (
    STATIC_ROUTE_ATTRS,
    VALIDATORS,
    _filter_eval_slice,
)

logger = logging.getLogger(__name__)


def _sweep_validator(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    n_total = len(df)
    n_err = int(df["is_error"].sum())
    s = df[col].values
    y = df["is_error"].values
    # NaN scores → sink to bottom
    s_clean = np.where(pd.isna(s), -np.inf, s.astype(float))
    order = np.argsort(-s_clean)
    y_sorted = y[order]
    cum_caught = np.cumsum(y_sorted)
    for k_pct in range(1, 101):
        k = k_pct / 100.0
        n_routed = max(1, int(np.ceil(k * n_total)))
        n_caught = int(cum_caught[n_routed - 1])
        n_left   = n_total - n_routed
        err_left = n_err - n_caught
        acc_unrouted = (1 - err_left / n_left) if n_left > 0 else 1.0
        rows.append({
            "validator": col,
            "routing_rate": n_routed / n_total,
            "n_routed": n_routed,
            "n_caught": n_caught,
            "accuracy_on_unrouted": acc_unrouted,
            "recall": (n_caught / n_err) if n_err else None,
            "precision": (n_caught / n_routed) if n_routed else None,
        })
    return pd.DataFrame(rows)


def _random_curve(df: pd.DataFrame) -> pd.DataFrame:
    n_total = len(df)
    n_err = int(df["is_error"].sum())
    rows = []
    for k_pct in range(1, 101):
        k = k_pct / 100.0
        n_routed = max(1, int(np.ceil(k * n_total)))
        # Expectation: caught ≈ base_rate * n_routed
        base_rate = n_err / n_total if n_total else 0.0
        n_caught_exp = base_rate * n_routed
        n_left = n_total - n_routed
        err_left_exp = n_err - n_caught_exp
        acc_unrouted = (1 - err_left_exp / n_left) if n_left > 0 else 1.0
        rows.append({
            "validator": "random",
            "routing_rate": n_routed / n_total,
            "n_routed": n_routed,
            "n_caught": n_caught_exp,
            "accuracy_on_unrouted": acc_unrouted,
            "recall": (n_caught_exp / n_err) if n_err else None,
            "precision": base_rate,
        })
    return pd.DataFrame(rows)


def _static_point(df: pd.DataFrame) -> pd.DataFrame:
    n_total = len(df)
    n_err = int(df["is_error"].sum())
    base = static_policy_baseline(df, attrs_to_route=STATIC_ROUTE_ATTRS)
    n_routed = base["n_routed"]
    routed_mask = df["attr"].isin(STATIC_ROUTE_ATTRS).values
    n_caught = int(df.loc[routed_mask, "is_error"].sum())
    n_left = n_total - n_routed
    err_left = n_err - n_caught
    acc_unrouted = (1 - err_left / n_left) if n_left > 0 else 1.0
    return pd.DataFrame([{
        "validator": "static_policy",
        "routing_rate": base["routing_rate"],
        "n_routed": n_routed,
        "n_caught": n_caught,
        "accuracy_on_unrouted": acc_unrouted,
        "recall": base["recall"],
        "precision": base["precision"],
    }])


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path",
                        default=os.path.join(PROCESSED_DIR, "validator_comparison_pasta.parquet"))
    parser.add_argument("--png", dest="png_path",
                        default=os.path.join(PROCESSED_DIR, "validator_pareto_pasta.png"))
    parser.add_argument("--out", dest="out_path",
                        default=os.path.join(PROCESSED_DIR, "validator_pareto_pasta.parquet"))
    parser.add_argument("--strict-slice", action="store_true")
    args = parser.parse_args()

    df_full = pd.read_parquet(args.in_path)
    df = _filter_eval_slice(df_full, strict=args.strict_slice)

    frames = [_sweep_validator(df, col) for col in VALIDATORS]
    frames.append(_random_curve(df))
    frames.append(_static_point(df))
    out_df = pd.concat(frames, ignore_index=True)
    out_df.to_parquet(args.out_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for col in VALIDATORS:
        sub = out_df[out_df["validator"] == col]
        ax.plot(sub["routing_rate"], sub["accuracy_on_unrouted"], label=col)
    rand = out_df[out_df["validator"] == "random"]
    ax.plot(rand["routing_rate"], rand["accuracy_on_unrouted"],
            linestyle="--", color="grey", label="random")
    stat = out_df[out_df["validator"] == "static_policy"]
    ax.scatter(stat["routing_rate"], stat["accuracy_on_unrouted"],
               marker="*", s=180, color="red", label="static policy", zorder=5)
    ax.set_xlabel("Routing rate (LLM-cost proxy)")
    ax.set_ylabel("Accuracy on unrouted cells")
    ax.set_title("Trek A1 — Pareto (pasta audited gold)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.png_path, dpi=120)
    logger.info("Wrote %s and %s", args.png_path, args.out_path)


if __name__ == "__main__":
    main()
