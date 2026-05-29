"""Clustered (code-grouped) bootstrap CI for E2E accuracy.

Wilson CI on n=16360 cascade-valid cells assumes iid observations. Cells are
grouped by product `code` (8–21 attributes per product → strong intra-cluster
correlation), so iid is violated and Wilson interval underestimates true
uncertainty.

Bootstrap procedure:
  - Resample WITH REPLACEMENT at the code level (not cell level).
  - For each resample, take ALL cells of selected codes; recompute accuracy.
  - Repeat B times (default 1000); report 2.5/97.5 and 5/95 percentiles.

Also outputs:
  - Wilson CI on n=16360 (for direct comparison).
  - Per-category bootstrap CIs (pasta / chocolate / cheeses).
  - Sanity check: bootstrap point estimate ≈ Wilson point estimate.

Backing: `datasets/processed/cascade_preds_{cat}_gold.parquet` —
per-cell predictions produced by `src/eval/end_to_end.py` (LLM-consensus gold).

Output: `datasets/processed/e2e_bootstrap_ci.json`.

Usage: python -m src.eval.bootstrap_ci_grouped
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "datasets" / "processed"
OUTPUT_PATH = DATA_DIR / "e2e_bootstrap_ci.json"

CATEGORIES = ["pasta", "chocolate", "cheeses"]
B = 1000
SEED = 42

# Same schema-exclude as src/eval/end_to_end.py — keeps reproduction strict.
SCHEMA_EXCLUDE = {
    "chocolate_type": {"filled", "other"},
    "chocolate_extra": {"filled", "other", "with_alcohol", "with_coffee"},
}


def _norm(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return str(v).lower().strip()


def load_cells():
    """Load per-cell cascade predictions, apply filters, return DataFrame with
    columns: code, cat, correct (bool)."""
    rows = []
    for cat in CATEGORIES:
        df = pd.read_parquet(DATA_DIR / f"cascade_preds_{cat}_gold.parquet")
        df["cat"] = cat
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)

    # Drop schema-deprecated gold values (same as headline pipeline).
    mask = pd.Series(True, index=df.index)
    for attr, excl in SCHEMA_EXCLUDE.items():
        mask &= ~((df["attr"] == attr) & df["gold_value"].isin(excl))
    df = df[mask]

    # in_scope only.
    df = df[df["in_scope"] == True]  # noqa: E712

    # cascade-valid = cascade_layer != fallback (matches headline denominator).
    df = df[df["cascade_layer"] != "fallback"].copy()

    # Normalise and compute per-cell correctness.
    gn = df["gold_value"].apply(_norm)
    e2e_n = df["e2e_pred"].apply(_norm)
    df["correct"] = (gn == e2e_n) & gn.notna()
    return df[["code", "cat", "correct"]].reset_index(drop=True)


def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval (two-sided)."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _z_from_alpha(alpha)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _z_from_alpha(alpha):
    from statistics import NormalDist

    return NormalDist().inv_cdf(1 - alpha / 2)


def clustered_bootstrap(df, B=1000, seed=42):
    """Cluster bootstrap: resample codes with replacement, take all their cells.

    Returns array of accuracy values (length B).
    """
    rng = np.random.default_rng(seed)
    codes = df["code"].to_numpy()
    correct = df["correct"].to_numpy().astype(np.int8)

    # Group cells by code: store indices per code in arrays.
    # Sort by code for contiguous groups → faster bootstrap via slicing.
    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    correct_sorted = correct[order]
    unique_codes, starts = np.unique(codes_sorted, return_index=True)
    # Append sentinel for ends.
    ends = np.append(starts[1:], len(codes_sorted))
    sizes = ends - starts
    # Precompute cumulative correctness sums per code: total correct per code.
    code_correct_sum = np.add.reduceat(correct_sorted, starts)
    # n cells per code = sizes.
    n_codes = len(unique_codes)

    accs = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n_codes, size=n_codes)
        total_correct = code_correct_sum[idx].sum()
        total_cells = sizes[idx].sum()
        accs[b] = total_correct / total_cells if total_cells > 0 else 0.0
    return accs


def percentile_ci(accs, lo, hi):
    return float(np.percentile(accs, lo)), float(np.percentile(accs, hi))


def summarise(df, label, B=B, seed=SEED):
    n_cells = len(df)
    n_codes = df["code"].nunique()
    k_correct = int(df["correct"].sum())
    point = k_correct / n_cells if n_cells else 0.0
    w_lo, w_hi = wilson_ci(k_correct, n_cells)
    accs = clustered_bootstrap(df, B=B, seed=seed)
    b25_lo, b25_hi = percentile_ci(accs, 2.5, 97.5)
    b5_lo, b5_hi = percentile_ci(accs, 5, 95)
    return {
        "label": label,
        "n_codes": int(n_codes),
        "n_cells": int(n_cells),
        "k_correct": k_correct,
        "B": B,
        "seed": seed,
        "point_estimate": float(point),
        "wilson_ci_95": [float(w_lo), float(w_hi)],
        "bootstrap_ci_2.5_97.5": [b25_lo, b25_hi],
        "bootstrap_ci_5_95": [b5_lo, b5_hi],
        "bootstrap_mean": float(accs.mean()),
        "bootstrap_std": float(accs.std(ddof=1)),
    }


def main():
    df = load_cells()
    print(f"Loaded {len(df)} cascade-valid cells across {df['code'].nunique()} codes")
    print("Per-category:")
    for cat in CATEGORIES:
        d = df[df.cat == cat]
        print(f"  {cat}: n={len(d)} cells / {d['code'].nunique()} codes")

    out = {
        "description": (
            "Code-grouped (clustered) bootstrap CI for E2E accuracy "
            "(None=wrong, cascade-valid denominator). Resample codes with "
            "replacement; recompute accuracy on all their cells. B=1000."
        ),
        "headline_global": summarise(df, label="global (pasta+chocolate+cheeses)"),
        "per_category": {
            cat: summarise(df[df.cat == cat].reset_index(drop=True), label=cat)
            for cat in CATEGORIES
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    g = out["headline_global"]
    print()
    print("=" * 80)
    print("GLOBAL E2E (cascade-valid, None=wrong)")
    print("=" * 80)
    print(f"  n_codes       : {g['n_codes']}")
    print(f"  n_cells       : {g['n_cells']}")
    print(f"  point estimate: {g['point_estimate']*100:.2f}%")
    print(
        f"  Wilson 95%    : [{g['wilson_ci_95'][0]*100:.2f}; "
        f"{g['wilson_ci_95'][1]*100:.2f}]"
    )
    b = g["bootstrap_ci_2.5_97.5"]
    print(
        f"  Bootstrap 95% : [{b[0]*100:.2f}; {b[1]*100:.2f}] "
        f"(clustered by code, B={B})"
    )
    w_w = (g["wilson_ci_95"][1] - g["wilson_ci_95"][0]) * 100
    b_w = (b[1] - b[0]) * 100
    print(f"  Width Wilson  : {w_w:.2f} п.п.")
    print(f"  Width Bootstrap: {b_w:.2f} п.п. ({b_w / w_w:.2f}× of Wilson)")
    print()
    print("Per-category bootstrap (B={}):".format(B))
    for cat, s in out["per_category"].items():
        bb = s["bootstrap_ci_2.5_97.5"]
        ww = s["wilson_ci_95"]
        print(
            f"  {cat:>10s}: acc={s['point_estimate']*100:.2f}%, "
            f"Wilson [{ww[0]*100:.2f}; {ww[1]*100:.2f}], "
            f"Bootstrap [{bb[0]*100:.2f}; {bb[1]*100:.2f}] "
            f"(n_cells={s['n_cells']}, n_codes={s['n_codes']})"
        )
    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
