"""gpt-5.5 vs Opus v2 gold agreement pilot analysis.

Usage:
    python -m src.experiments.gpt55_vs_opus_agreement \
        --pilot-dir datasets/processed/gpt55_pilot \
        --gold datasets/processed/consensus_gold_v2_off_grounded.parquet

Computes cell-level agreement between gpt-5.5 OFF-grounded annotations
and Opus v2 gold standard. Reports per-category, per-attribute breakdown
plus overall agreement and refusal rate.

Verdict thresholds:
  >= 90%  -> GREEN  (valid Opus proxy, scale approved)
  85-90%  -> YELLOW (marginal, consider hybrid weighting)
  < 85%   -> RED    (abort scale)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


_CAT_DOMAINS = {
    "pasta": "pasta",
    "chocolate": "chocolate",
    "cheeses": "cheeses",
}


def load_pilot_predictions(pilot_dir: Path) -> pd.DataFrame:
    """Load all gpt-5.5 pilot parquets and extract (code, attr, value) rows."""
    rows = []
    for cat in _CAT_DOMAINS:
        p = pilot_dir / f"gpt55_pilot_{cat}.parquet"
        if not p.exists():
            print(f"[WARN] Missing pilot parquet for {cat}: {p}")
            continue
        df = pd.read_parquet(p)
        df["code"] = df["code"].astype(str)
        for _, row in df.iterrows():
            try:
                parsed = json.loads(row["parsed_json"])
            except Exception:
                parsed = {}
            for attr, val in parsed.items():
                rows.append({
                    "category": cat,
                    "code": row["code"],
                    "attr": attr,
                    "gpt55_value": None if val is None else str(val),
                    "gpt55_null": val is None,
                })
    return pd.DataFrame(rows)


def normalize_value(v) -> str | None:
    """Normalize value for comparison: lowercase strings, canonical booleans."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    # Normalize boolean variants
    if s in ("true", "1", "yes"):
        return "true"
    if s in ("false", "0", "no"):
        return "false"
    return s


def compute_agreement(pilot: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Compute cell-level agreement between gpt-5.5 predictions and Opus gold.

    Only evaluates cells where Opus gold is non-null.
    Returns dict with overall and per-(cat, attr) stats.
    """
    # Normalize gold values
    gold = gold.copy()
    gold["code"] = gold["code"].astype(str)
    gold["gold_norm"] = gold["gold_value"].apply(normalize_value)
    gold_non_null = gold[~gold["gold_is_null"]].copy()

    # Normalize pilot predictions
    pilot = pilot.copy()
    pilot["gpt55_norm"] = pilot["gpt55_value"].apply(normalize_value)

    # Merge on (category, code, attr)
    merged = gold_non_null.merge(
        pilot[["category", "code", "attr", "gpt55_norm", "gpt55_null"]],
        on=["category", "code", "attr"],
        how="left",
    )
    # Left-only rows (no gpt-5.5 prediction for a gold cell) count as refusal
    merged["gpt55_null"] = merged["gpt55_null"].fillna(True)
    merged["gpt55_norm"] = merged["gpt55_norm"].fillna(None)

    total_cells = len(merged)
    refusal_cells = merged["gpt55_null"].sum()
    # Only count non-refusal cells as agreement candidates
    answered = merged[~merged["gpt55_null"]].copy()
    answered["match"] = answered["gold_norm"] == answered["gpt55_norm"]
    matched_cells = answered["match"].sum()
    # Agreement = matches / total non-null gold cells (refusals count as wrong)
    overall_agreement = matched_cells / total_cells if total_cells > 0 else 0.0
    refusal_rate = refusal_cells / total_cells if total_cells > 0 else 0.0

    # Per-category breakdown
    per_cat = {}
    for cat, grp in merged.groupby("category"):
        cat_total = len(grp)
        cat_refusal = grp["gpt55_null"].sum()
        cat_answered = grp[~grp["gpt55_null"]]
        cat_matched = (cat_answered["gold_norm"] == cat_answered["gpt55_norm"]).sum()
        per_cat[cat] = {
            "total_cells": int(cat_total),
            "refusal_cells": int(cat_refusal),
            "matched_cells": int(cat_matched),
            "agreement": round(cat_matched / cat_total, 4) if cat_total > 0 else 0.0,
            "refusal_rate": round(cat_refusal / cat_total, 4) if cat_total > 0 else 0.0,
        }

    # Per-(cat, attr) breakdown
    per_cat_attr = {}
    for (cat, attr), grp in merged.groupby(["category", "attr"]):
        ga_total = len(grp)
        ga_refusal = grp["gpt55_null"].sum()
        ga_answered = grp[~grp["gpt55_null"]]
        ga_matched = (ga_answered["gold_norm"] == ga_answered["gpt55_norm"]).sum()
        per_cat_attr[f"{cat}/{attr}"] = {
            "total": int(ga_total),
            "matched": int(ga_matched),
            "refusals": int(ga_refusal),
            "agreement": round(ga_matched / ga_total, 4) if ga_total > 0 else 0.0,
        }

    return {
        "total_cells": int(total_cells),
        "matched_cells": int(matched_cells),
        "refusal_cells": int(refusal_cells),
        "overall_agreement": round(overall_agreement, 4),
        "refusal_rate": round(refusal_rate, 4),
        "per_category": per_cat,
        "per_cat_attr": per_cat_attr,
    }


def verdict(agreement: float) -> str:
    if agreement >= 0.90:
        return "GREEN"
    if agreement >= 0.85:
        return "YELLOW"
    return "RED"


def main():
    ap = argparse.ArgumentParser(description="gpt-5.5 vs Opus v2 gold agreement analysis")
    ap.add_argument("--pilot-dir", type=Path, default=Path("datasets/processed/gpt55_pilot"),
                    help="Directory containing gpt55_pilot_{cat}.parquet files")
    ap.add_argument("--gold", type=Path,
                    default=Path("datasets/processed/consensus_gold_v2_off_grounded.parquet"),
                    help="Opus v2 gold parquet (long format)")
    ap.add_argument("--min-codes", type=int, default=10,
                    help="Minimum codes per cat to include in overall agreement (skip smaller)")
    args = ap.parse_args()

    # Load data
    print("Loading gpt-5.5 pilot predictions...")
    pilot = load_pilot_predictions(args.pilot_dir)
    cats_found = sorted(pilot["category"].unique()) if len(pilot) > 0 else []
    print(f"  Pilot cats: {cats_found}")
    for cat in cats_found:
        n_codes = pilot[pilot["category"] == cat]["code"].nunique()
        print(f"    {cat}: {n_codes} codes, {len(pilot[pilot['category'] == cat])} attr-rows")

    print("\nLoading Opus v2 gold...")
    gold = pd.read_parquet(args.gold)
    print(f"  Gold shape: {gold.shape}")

    # Filter gold to pilot codes only, and skip cats with too few codes
    included_cats = []
    for cat in cats_found:
        pilot_codes = set(pilot[pilot["category"] == cat]["code"].astype(str))
        if len(pilot_codes) >= args.min_codes:
            included_cats.append(cat)
        else:
            print(f"  [SKIP] {cat}: only {len(pilot_codes)} codes < min {args.min_codes}")

    pilot_filtered = pilot[pilot["category"].isin(included_cats)]
    gold_filtered = gold[gold["category"].isin(included_cats)].copy()

    # Filter gold to only pilot codes
    pilot_code_pairs = set(zip(pilot_filtered["category"], pilot_filtered["code"].astype(str)))
    gold_filtered = gold_filtered[
        gold_filtered.apply(lambda r: (r["category"], str(r["code"])) in pilot_code_pairs, axis=1)
    ]

    print(f"\nEvaluating {len(included_cats)} categories: {included_cats}")
    results = compute_agreement(pilot_filtered, gold_filtered)

    # Print summary
    print("\n" + "=" * 60)
    print("GPT-5.5 vs Opus v2 Gold — Agreement Summary")
    print("=" * 60)
    print(f"Total gold cells evaluated : {results['total_cells']}")
    print(f"Matched cells              : {results['matched_cells']}")
    print(f"Refusal cells (gpt-5.5 null): {results['refusal_cells']} ({results['refusal_rate']*100:.1f}%)")
    print(f"Overall agreement          : {results['overall_agreement']*100:.1f}%")
    print()
    print("Per-category breakdown:")
    for cat, stats in results["per_category"].items():
        print(f"  {cat}: {stats['agreement']*100:.1f}% agreement "
              f"({stats['matched_cells']}/{stats['total_cells']} cells, "
              f"{stats['refusal_rate']*100:.1f}% refusal)")
    print()
    print("Per (cat/attr) breakdown:")
    for key, stats in sorted(results["per_cat_attr"].items()):
        print(f"  {key:35s}: {stats['agreement']*100:.1f}% "
              f"({stats['matched']}/{stats['total']}, {stats['refusals']} refusals)")

    v = verdict(results["overall_agreement"])
    print()
    print("=" * 60)
    print(f"VERDICT: {v}")
    if v == "GREEN":
        print("  >= 90% agreement — gpt-5.5 is valid as Opus proxy.")
        print("  Scale to ~650 codes per cat is APPROVED.")
    elif v == "YELLOW":
        print("  85-90% agreement — marginal. Consider hybrid weighting in annotations.")
        print("  Scaling requires review.")
    else:
        print("  < 85% agreement — gpt-5.5 is NOT Opus-quality.")
        print("  Scale ABORTED.")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
