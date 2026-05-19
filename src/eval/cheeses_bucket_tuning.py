"""Pasta-only bucket boundary derivation for §6.18.

Finds optimal bin boundaries for bucketed-numeric attributes in the pasta schema
by grid-searching over percentile thresholds, scoring against pasta v2 gold.

Bucketed attrs handled:
  - protein_class: 4-class enum {"0", "low", "med", "high"} from proteins_100g
    (fat_class is NOT in PASTA_SCHEMA; it appears only in CHEESES_SCHEMA)

Only the 239 v1 overlap codes are used for training (as specified in §6.18)
since they constitute the manually-audited pasta gold set.

Outputs models/pasta_bucket_boundaries.json.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORKTREE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(WORKTREE_ROOT, "datasets", "manual_label", "off_cache")
PROCESSED_DIR = os.path.join(WORKTREE_ROOT, "datasets", "processed")
MODELS_DIR = os.path.join(WORKTREE_ROOT, "models")

V1_PASTA_PARQUET = os.path.join(PROCESSED_DIR, "cascade_vs_audited_gold_pasta.parquet")
V2_GOLD_PARQUET = os.path.join(PROCESSED_DIR, "consensus_gold_v2_expanded.parquet")
# Save to datasets/processed/ because models/ is a symlink excluded from git
OUTPUT_JSON = os.path.join(PROCESSED_DIR, "pasta_bucket_boundaries.json")

# Bucket spec: attr -> (nutriment_key, ordered_labels)
BUCKET_ATTRS: dict[str, tuple[str, list[str]]] = {
    "protein_class": ("proteins_100g", ["0", "low", "med", "high"]),
}

# Grid: percentile thresholds to try for each boundary
PERCENTILE_GRID = list(range(5, 96, 5))  # 5, 10, ..., 95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_off_cache(codes: list[str]) -> dict[str, dict[str, Any]]:
    """Return {code: nutriments_dict} for each code that has a cache file."""
    result: dict[str, dict[str, Any]] = {}
    for code in codes:
        fpath = os.path.join(CACHE_DIR, f"{code}.json")
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            result[code] = data.get("nutriments", {})
        except Exception:  # noqa: BLE001
            pass
    return result


def _apply_boundaries(value: float | None, boundaries: list[float], labels: list[str]) -> str | None:
    """Apply [b0, b1, ...] thresholds to produce a label.

    Labels are assigned as follows:
        value < b0                  → labels[0]
        b0 <= value < b1            → labels[1]
        ...
        value >= b_{n-1}            → labels[n]

    Returns None if value is None or NaN.
    """
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


def _score_boundaries(
    values: list[float | None],
    gold_labels: list[str | None],
    boundaries: list[float],
    labels: list[str],
) -> float:
    """Return accuracy of applying boundaries to values vs gold_labels.

    Only rows where both value and gold_label are non-None count toward
    the denominator.
    """
    correct = 0
    total = 0
    for v, g in zip(values, gold_labels):
        if g is None or v is None:
            continue
        pred = _apply_boundaries(v, boundaries, labels)
        if pred == g:
            correct += 1
        total += 1
    if total == 0:
        return 0.0
    return correct / total


def _grid_search_boundaries(
    values: list[float | None],
    gold_labels: list[str | None],
    labels: list[str],
    percentile_grid: list[int] | None = None,
) -> tuple[list[float], float]:
    """Grid-search 2-boundary combinations over percentile values.

    For a 4-class scheme (0/low/med/high), we need 3 boundaries.
    For a 3-class scheme we need 2.

    Returns (best_boundaries, best_accuracy).
    """
    if percentile_grid is None:
        percentile_grid = PERCENTILE_GRID

    valid_values = [v for v in values if v is not None and not np.isnan(v)]
    if not valid_values:
        raise ValueError("No valid numeric values to compute percentiles")

    n_boundaries = len(labels) - 1  # 3 for 4 classes, 2 for 3 classes

    # Candidate thresholds at each percentile
    candidate_thresholds = sorted(set(
        round(float(np.percentile(valid_values, p)), 4) for p in percentile_grid
    ))

    best_acc = -1.0
    best_boundaries: list[float] = []

    # Enumerate strictly increasing boundary combinations
    for combo in itertools.combinations(candidate_thresholds, n_boundaries):
        boundaries = list(combo)
        acc = _score_boundaries(values, gold_labels, boundaries, labels)
        if acc > best_acc:
            best_acc = acc
            best_boundaries = boundaries

    return best_boundaries, best_acc


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def derive_pasta_bucket_boundaries(
    *,
    v1_only: bool = True,
    percentile_grid: list[int] | None = None,
    save: bool = True,
) -> dict[str, dict]:
    """Derive bucket boundaries for pasta bucketed-numeric attrs.

    Args:
        v1_only: If True, use only the 239 v1-overlap codes (default, per §6.18).
                 If False, use all 889 pasta v2 gold codes.
        percentile_grid: Percentile values to try as candidate boundaries.
        save: If True, write result to OUTPUT_JSON.

    Returns:
        dict mapping attr -> boundary spec dict.
    """
    # Load v2 gold
    v2_df = pd.read_parquet(V2_GOLD_PARQUET)
    pasta_v2 = v2_df[v2_df["category"] == "pasta"].copy()

    # Determine code set
    if v1_only:
        v1_df = pd.read_parquet(V1_PASTA_PARQUET)
        v1_codes = set(v1_df["code"].astype(str).unique())
        pasta_v2 = pasta_v2[pasta_v2["code"].astype(str).isin(v1_codes)]
        logger.info("Using %d v1 pasta codes for bucket tuning", len(v1_codes))
    else:
        logger.info("Using all %d pasta v2 gold codes", pasta_v2["code"].nunique())

    result: dict[str, dict] = {}

    for attr, (nutri_key, labels) in BUCKET_ATTRS.items():
        attr_df = pasta_v2[pasta_v2["attr"] == attr].copy()
        # Only non-null gold rows
        attr_df = attr_df[~attr_df["gold_is_null"]].copy()

        if len(attr_df) == 0:
            logger.warning("No non-null gold rows for attr=%s, skipping", attr)
            continue

        codes = attr_df["code"].astype(str).tolist()
        gold_labels_list = attr_df["gold_value"].tolist()

        # Fetch nutriment values from OFF cache
        cache = _load_off_cache(codes)
        nutri_values: list[float | None] = []
        for code in codes:
            nutri = cache.get(code, {})
            val = nutri.get(nutri_key)
            if val is not None:
                try:
                    nutri_values.append(float(val))
                except (TypeError, ValueError):
                    nutri_values.append(None)
            else:
                nutri_values.append(None)

        n_with_nutri = sum(1 for v in nutri_values if v is not None)
        logger.info(
            "attr=%s: %d gold rows, %d with %s from OFF cache",
            attr, len(attr_df), n_with_nutri, nutri_key,
        )

        best_boundaries, best_acc = _grid_search_boundaries(
            nutri_values, gold_labels_list, labels, percentile_grid=percentile_grid
        )

        logger.info(
            "attr=%s: best_boundaries=%s, best_accuracy=%.4f",
            attr, best_boundaries, best_acc,
        )

        result[attr] = {
            "feature": nutri_key,
            "boundaries": best_boundaries,
            "labels": labels,
            "n_train": n_with_nutri,
            "best_accuracy": round(best_acc, 4),
        }

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved bucket boundaries to %s", OUTPUT_JSON)

    return result


def load_bucket_boundaries(path: str | None = None) -> dict[str, dict]:
    """Load pre-computed bucket boundaries from JSON."""
    path = path or OUTPUT_JSON
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_bucket_boundaries(
    value: float | None,
    attr: str,
    boundaries_spec: dict[str, dict],
) -> str | None:
    """Apply saved boundaries for an attr to a single numeric value.

    Returns the label string or None if value is missing / unmappable.
    """
    spec = boundaries_spec.get(attr)
    if spec is None:
        return None
    return _apply_boundaries(value, spec["boundaries"], spec["labels"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    boundaries = derive_pasta_bucket_boundaries(v1_only=True)
    for attr, spec in boundaries.items():
        print(
            f"{attr}: boundaries={spec['boundaries']}, "
            f"labels={spec['labels']}, "
            f"n_train={spec['n_train']}, "
            f"best_accuracy={spec['best_accuracy']:.4f}"
        )
