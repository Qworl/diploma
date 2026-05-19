"""T13: Cycle-delta comparison — pasta-tuned buckets applied to choc+cheeses.

For each category in [pasta, chocolate, cheeses]:
1. Load cascade predictions on v2 gold codes (from cascade_preds_{cat}_v2_gold.parquet).
2. For bucketed-numeric attrs (protein_class, fat_class):
   - protein_class: post-process using pasta_bucket_boundaries.json.
   - fat_class (cheeses): apply schema-documented hard boundaries [<15, 15-25, 25-32, >32].
   These are NOT re-tuned per cat — this is the methodology.
3. Score vs v2 blind gold (consensus_gold_v2_expanded.parquet).
4. Load v1 prefill baseline (cascade_vs_audited_gold_{cat}.parquet) for the same attrs.
5. Cycle delta = acc_v2_with_new_buckets - acc_v1_prefill_with_old_buckets per (cat, attr).

Outputs:
  datasets/processed/cascade_vs_blind_gold_{pasta,chocolate,cheeses}_v2.parquet

Per-category summary tables are printed to stdout.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from src.eval.cheeses_bucket_tuning import apply_bucket_boundaries, load_bucket_boundaries

logger = logging.getLogger(__name__)

WORKTREE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROCESSED_DIR = os.path.join(WORKTREE_ROOT, "datasets", "processed")
CACHE_DIR = os.path.join(WORKTREE_ROOT, "datasets", "manual_label", "off_cache")

BUCKET_BOUNDARIES_PATH = os.path.join(PROCESSED_DIR, "pasta_bucket_boundaries.json")

# Cheeses fat_class schema-documented boundaries: low<15, medium 15-25, high 25-32, very_high>32
CHEESES_FAT_CLASS_BOUNDARIES: list[float] = [15.0, 25.0, 32.0]
CHEESES_FAT_CLASS_LABELS: list[str] = ["low", "medium", "high", "very_high"]

CATEGORIES = ["pasta", "chocolate", "cheeses"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_off_nutriment(codes: list[str], nutri_key: str) -> dict[str, float | None]:
    """Return {code: float|None} for nutriment_key from OFF cache."""
    result: dict[str, float | None] = {}
    for code in codes:
        fpath = os.path.join(CACHE_DIR, f"{code}.json")
        if not os.path.exists(fpath):
            result[code] = None
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            val = data.get("nutriments", {}).get(nutri_key)
            result[code] = float(val) if val is not None else None
        except Exception:  # noqa: BLE001
            result[code] = None
    return result


def _apply_fixed_boundaries(
    value: float | None,
    boundaries: list[float],
    labels: list[str],
) -> str | None:
    """Apply fixed boundaries (same logic as cheeses_bucket_tuning._apply_boundaries)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(v):
        return None
    for i, boundary in enumerate(boundaries):
        if v < boundary:
            return labels[i]
    return labels[len(boundaries)]


def _postprocess_protein_class(
    preds_df: pd.DataFrame,
    bucket_spec: dict[str, dict],
    cat_codes: list[str],
) -> pd.DataFrame:
    """Replace protein_class predictions with bucket-derived values from OFF cache.

    This ensures protein_class is derived from proteins_100g using the
    pasta-tuned boundaries — NOT from the ML classifier.
    """
    preds_df = preds_df.copy()
    pc_mask = preds_df["attr"] == "protein_class"

    if not pc_mask.any():
        return preds_df

    # Load proteins_100g for all codes
    nutriments = _load_off_nutriment(cat_codes, "proteins_100g")

    def _bucket_protein(row: Any) -> str | None:
        val = nutriments.get(str(row["code"]))
        return apply_bucket_boundaries(val, "protein_class", bucket_spec)

    preds_df.loc[pc_mask, "predicted"] = preds_df[pc_mask].apply(_bucket_protein, axis=1)
    preds_df.loc[pc_mask, "layer"] = "bucket_rule"
    preds_df.loc[pc_mask, "confidence"] = 1.0

    return preds_df


def _postprocess_fat_class(
    preds_df: pd.DataFrame,
    cat_codes: list[str],
) -> pd.DataFrame:
    """Replace fat_class predictions with schema-fixed boundaries from OFF cache fat_100g.

    Boundaries: low<15, medium 15-25, high 25-32, very_high>32.
    """
    preds_df = preds_df.copy()
    fc_mask = preds_df["attr"] == "fat_class"

    if not fc_mask.any():
        return preds_df

    nutriments = _load_off_nutriment(cat_codes, "fat_100g")

    def _bucket_fat(row: Any) -> str | None:
        val = nutriments.get(str(row["code"]))
        return _apply_fixed_boundaries(val, CHEESES_FAT_CLASS_BOUNDARIES, CHEESES_FAT_CLASS_LABELS)

    preds_df.loc[fc_mask, "predicted"] = preds_df[fc_mask].apply(_bucket_fat, axis=1)
    preds_df.loc[fc_mask, "layer"] = "bucket_rule"
    preds_df.loc[fc_mask, "confidence"] = 1.0

    return preds_df


def _compute_v2_accuracy(
    preds_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    cat: str,
) -> dict[str, dict]:
    """Compute accuracy per attr vs v2 blind gold.

    Only non-null gold rows are counted toward the denominator.
    Returns {attr: {n: int, n_correct: int, accuracy: float, n_null_gold: int}}
    """
    cat_gold = gold_df[gold_df["category"] == cat].copy()
    cat_gold["code"] = cat_gold["code"].astype(str)
    preds_df = preds_df.copy()
    preds_df["code"] = preds_df["code"].astype(str)

    merged = preds_df.merge(
        cat_gold[["code", "attr", "gold_value", "gold_is_null"]],
        on=["code", "attr"],
        how="inner",
    )

    result: dict[str, dict] = {}
    for attr in merged["attr"].unique():
        attr_rows = merged[merged["attr"] == attr]
        non_null = attr_rows[~attr_rows["gold_is_null"]]
        n = len(non_null)
        n_correct = (
            (non_null["predicted"].astype(str) == non_null["gold_value"].astype(str)).sum()
            if n > 0
            else 0
        )
        result[attr] = {
            "n": n,
            "n_correct": int(n_correct),
            "accuracy": float(n_correct / n) if n > 0 else float("nan"),
            "n_null_gold": int(attr_rows["gold_is_null"].sum()),
        }

    return result


def _get_v1_accuracy(v1_df: pd.DataFrame) -> dict[str, float]:
    """Compute v1 per-attr accuracy using manual_value vs cascade_pred.

    Rows counted: audited statuses only (confirmed / manual_only / override),
    excluding prefill-auto rows (same filtering as cascade_vs_audited_gold.py).
    """
    audited_statuses = {"confirmed", "manual_only", "override"}
    audited_modes = {"blind", "llm"}

    mask = (
        v1_df["status"].isin(audited_statuses)
        & v1_df["mode"].isin(audited_modes)
    )
    sub = v1_df[mask].copy()
    sub["cascade_pred"] = sub["cascade_pred"].astype(str)
    sub["manual_value"] = sub["manual_value"].astype(str)

    result: dict[str, float] = {}
    for attr in sub["attr"].unique():
        attr_rows = sub[sub["attr"] == attr]
        n = len(attr_rows)
        n_correct = (attr_rows["cascade_pred"] == attr_rows["manual_value"]).sum()
        result[attr] = float(n_correct / n) if n > 0 else float("nan")

    return result


def _print_summary(cat: str, v2_acc: dict[str, dict], v1_acc: dict[str, float]) -> None:
    """Print per-attr summary table with cycle delta."""
    print(f"\n{'='*60}")
    print(f"  {cat.upper()} — Cycle Delta Summary")
    print(f"{'='*60}")
    print(f"{'attr':<25} {'acc_v1':>8} {'acc_v2':>8} {'delta_pp':>10} {'n_v2':>6}")
    print(f"{'-'*60}")

    for attr in sorted(v2_acc.keys()):
        v2 = v2_acc[attr]
        acc_v2 = v2["accuracy"]
        acc_v1 = v1_acc.get(attr, float("nan"))

        if np.isnan(acc_v1) or np.isnan(acc_v2):
            delta_pp = float("nan")
            delta_str = "  n/a"
        else:
            delta_pp = (acc_v2 - acc_v1) * 100.0
            delta_str = f"{delta_pp:+.1f}"

        acc_v1_str = f"{acc_v1:.3f}" if not np.isnan(acc_v1) else "  n/a"
        acc_v2_str = f"{acc_v2:.3f}" if not np.isnan(acc_v2) else "  n/a"
        print(
            f"{attr:<25} {acc_v1_str:>8} {acc_v2_str:>8} {delta_str:>10} {v2['n']:>6}"
        )


def run_cascade_vs_blind_gold(
    categories: list[str] | None = None,
) -> dict[str, dict]:
    """Run T13 comparison for each category. Return {cat: {attr: {metrics...}}}."""
    if categories is None:
        categories = CATEGORIES

    bucket_spec = load_bucket_boundaries(BUCKET_BOUNDARIES_PATH)
    v2_gold = pd.read_parquet(os.path.join(PROCESSED_DIR, "consensus_gold_v2_expanded.parquet"))

    all_results: dict[str, dict] = {}

    for cat in categories:
        logger.info("Processing category: %s", cat)

        # Load v2 cascade predictions
        preds_path = os.path.join(PROCESSED_DIR, f"cascade_preds_{cat}_v2_gold.parquet")
        preds_df = pd.read_parquet(preds_path)
        preds_df["code"] = preds_df["code"].astype(str)

        cat_codes = preds_df["code"].unique().tolist()

        # Post-process bucketed attrs
        if cat in ("pasta", "chocolate"):
            preds_df = _postprocess_protein_class(preds_df, bucket_spec, cat_codes)
        if cat == "cheeses":
            preds_df = _postprocess_fat_class(preds_df, cat_codes)

        # Score vs v2 blind gold
        v2_acc = _compute_v2_accuracy(preds_df, v2_gold, cat)

        # Load v1 baseline
        v1_path = os.path.join(PROCESSED_DIR, f"cascade_vs_audited_gold_{cat}.parquet")
        v1_acc: dict[str, float] = {}
        if os.path.exists(v1_path):
            v1_df = pd.read_parquet(v1_path)
            v1_acc = _get_v1_accuracy(v1_df)
        else:
            logger.warning("V1 baseline not found for %s, cycle delta will be n/a", cat)

        # Print summary
        _print_summary(cat, v2_acc, v1_acc)

        # Build output rows
        rows = []
        for attr, v2 in v2_acc.items():
            acc_v1 = v1_acc.get(attr, float("nan"))
            acc_v2 = v2["accuracy"]
            delta_pp = (
                (acc_v2 - acc_v1) * 100.0
                if not (np.isnan(acc_v1) or np.isnan(acc_v2))
                else float("nan")
            )
            rows.append(
                {
                    "category": cat,
                    "attr": attr,
                    "acc_v1_prefill": acc_v1,
                    "acc_v2_new_buckets": acc_v2,
                    "cycle_delta_pp": delta_pp,
                    "n_v2_non_null": v2["n"],
                    "n_v2_correct": v2["n_correct"],
                    "n_v2_null_gold": v2["n_null_gold"],
                }
            )

        out_df = pd.DataFrame(rows)
        out_path = os.path.join(PROCESSED_DIR, f"cascade_vs_blind_gold_{cat}_v2.parquet")
        out_df.to_parquet(out_path, index=False)
        logger.info("Saved %s", out_path)

        all_results[cat] = {attr: dict(v2_acc[attr]) for attr in v2_acc}
        all_results[cat]["_v1_acc"] = v1_acc

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run_cascade_vs_blind_gold()

    print("\n\n" + "=" * 60)
    print("CYCLE DELTA SUMMARY (cats with delta ≥ +5pp = replicated)")
    print("=" * 60)

    for cat in CATEGORIES:
        if cat not in results:
            continue
        cat_results = results[cat]
        v1_acc = cat_results.pop("_v1_acc", {})
        deltas = []
        for attr, v2 in cat_results.items():
            acc_v1 = v1_acc.get(attr, float("nan"))
            acc_v2 = v2["accuracy"]
            if not (np.isnan(acc_v1) or np.isnan(acc_v2)):
                deltas.append((acc_v2 - acc_v1) * 100.0)

        if deltas:
            mean_delta = np.mean(deltas)
            cats_replicated = sum(1 for d in deltas if d >= 5.0)
            print(f"\n{cat}: mean_delta={mean_delta:+.1f}pp, attrs_with_delta≥+5pp={cats_replicated}/{len(deltas)}")
