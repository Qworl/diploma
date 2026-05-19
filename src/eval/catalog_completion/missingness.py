"""Compute partner-typical attribute-missingness rates from silver standards."""
from __future__ import annotations

import json
import logging
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

PARTNER_ATTRS = ("product_name", "brands", "ingredients_text", "quantity")

# Clamp empirical rates to avoid pathological 0% / 100% — masking with p=0 is a
# no-op (no signal) and p=1 makes the row empty (also no signal).
MIN_RATE = 0.05
MAX_RATE = 0.95


def _empty_rate(col: pd.Series) -> float:
    """Fraction of rows where col is null OR whitespace-only string."""
    if col.dtype == bool:
        # Booleans cannot be 'missing' as a value — only NaN counts.
        return float(col.isna().mean())
    null = col.isna()
    try:
        empty_str = col.astype(str).str.strip().eq("")
    except Exception:
        empty_str = pd.Series(False, index=col.index)
    return float((null | empty_str).mean())


def _clamp(p: float) -> float:
    return max(MIN_RATE, min(MAX_RATE, p))


def compute_missingness_profile(
    df: pd.DataFrame,
    target_attrs: Iterable[str],
    partner_attrs: Iterable[str] = PARTNER_ATTRS,
) -> dict:
    """Return {n_rows, partner_attrs: {a: p}, target_attrs: {a: p}}.

    `p` is the empirical fraction of rows where the column is null/empty,
    clamped to [MIN_RATE, MAX_RATE]. For booleans, only NaN counts as missing.
    """
    target_attrs = list(target_attrs)
    partner_attrs = list(partner_attrs)
    out_partner: dict[str, float] = {}
    out_target: dict[str, float] = {}
    for a in partner_attrs:
        if a in df.columns:
            out_partner[a] = _clamp(_empty_rate(df[a]))
    for a in target_attrs:
        if a in df.columns:
            out_target[a] = _clamp(_empty_rate(df[a]))
    return {
        "n_rows": int(len(df)),
        "partner_attrs": out_partner,
        "target_attrs": out_target,
    }


def save_profile(profile: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(profile, f, indent=2, sort_keys=True)


def load_profile(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
