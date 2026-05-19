"""Trek A2 — per-category Pareto plot (coverage gain vs LLM cost).

Usage:
    OMP_NUM_THREADS=1 python -m src.eval.catalog_completion.pareto_plot
"""
from __future__ import annotations

import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                       "docs", "thesis", "figures")


def _load(prefix: str, category: str) -> list[dict]:
    out = []
    for tag in ("no_llm", "with_llm"):
        p = f"{prefix}_summary_{category}_{tag}.json"
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
                d["_tag"] = tag
                out.append(d)
    return out


def _static_policy_point(category: str) -> dict | None:
    """Returns the static-policy operating point from Plan B4 router_pareto
    (pasta only — other cats not yet evaluated)."""
    path = os.path.join(PROCESSED_DIR, "router_pareto_gold.parquet")
    if not os.path.exists(path) or category != "pasta":
        return None
    df = pd.read_parquet(path)
    # Plan B4 winner: static per-attr policy at ~34% LLM cost (see MEMORY notes).
    if "config" in df.columns:
        sub = df[df["config"].astype(str).str.contains("static", case=False, na=False)]
        if len(sub):
            row = sub.iloc[0]
            return {
                "coverage_gain_pp": float(row.get("coverage_pp", row.get("coverage", 0)) or 0),
                "llm_cost_usd_per_1000_products": float(row.get("cost_usd", 0) or 0),
            }
    return None


def plot_category(category: str, out_path: str) -> None:
    prefix = os.path.join(PROCESSED_DIR, "catalog_completion")
    pts = _load(prefix, category)
    if not pts:
        logger.warning("No summary JSONs found for %s — skip", category)
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [p["llm_cost_usd_per_1000_products"] for p in pts]
    ys = [p["coverage_gain_pp"] for p in pts]
    labels = [p["_tag"] for p in pts]
    ax.plot(xs, ys, "o-", color="tab:blue", label="Trek A2 cascade")
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 5))

    sp = _static_policy_point(category)
    if sp is not None:
        ax.scatter([sp["llm_cost_usd_per_1000_products"]],
                   [sp["coverage_gain_pp"]],
                   marker="x", s=80, color="tab:red",
                   label="Static-policy (Plan B4 winner)")

    ax.set_xlabel("LLM cost (USD per 1000 products)")
    ax.set_ylabel("Coverage gain (pp)")
    ax.set_title(f"Trek A2 — catalog completion Pareto ({category})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main() -> None:
    setup_logging()
    os.makedirs(FIG_DIR, exist_ok=True)
    for cat in ("pasta", "cheeses", "electronics"):
        out = os.path.join(FIG_DIR, f"trek_a2_pareto_{cat}.png")
        plot_category(cat, out)


if __name__ == "__main__":
    main()
