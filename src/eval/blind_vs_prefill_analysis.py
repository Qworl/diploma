"""Analyze agreement between blind-Opus (v2) and prefill-Opus (v1) decisions.

Outputs metrics that drive Checkpoint 1 in spec §11:
  - overall_agreement: fraction of non-null cells where blind == prefill
  - per_attr agreement, refusal_rate, Cohen's kappa
  - flip_direction: when blind disagrees with prefill, does it move toward
    silver or away (anchoring direction signal)

Inputs:
  - prefill_decisions: dict[code, dict[attr, dict]] with "value" key (v1 format
    from src.manual_label.opus_audit_caller).
  - blind_decisions: dict[code, dict[attr, dict]] with "value" key (v2 format
    from src.manual_label.opus_off_grounded_audit, identical schema).
  - silver_df: DataFrame with columns [code, attr1, attr2, ...] containing
    silver values (used only for flip_direction).
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)


def _value(decision_cell: Any) -> Any:
    """Extract value from a decision cell (handles None and dict)."""
    if decision_cell is None:
        return None
    if isinstance(decision_cell, dict):
        return decision_cell.get("value")
    return decision_cell


def _common_codes(prefill: dict, blind: dict) -> list[str]:
    return sorted(set(prefill) & set(blind))


def _all_attrs(decisions: dict) -> list[str]:
    attrs: set[str] = set()
    for cell in decisions.values():
        attrs.update(cell.keys())
    return sorted(attrs)


def compute_agreement(
    prefill: dict[str, dict],
    blind: dict[str, dict],
) -> dict[str, float | int]:
    """Overall agreement: fraction of non-null cells where blind == prefill."""
    codes = _common_codes(prefill, blind)
    attrs = set(_all_attrs(prefill)) | set(_all_attrs(blind))

    n_total = 0
    n_non_null = 0
    n_agree = 0
    for code in codes:
        for attr in attrs:
            pre = _value(prefill.get(code, {}).get(attr))
            bli = _value(blind.get(code, {}).get(attr))
            n_total += 1
            if bli is None:
                continue
            n_non_null += 1
            if pre == bli:
                n_agree += 1

    overall = n_agree / n_non_null if n_non_null else 0.0
    return {
        "overall_agreement": overall,
        "n_total_cells": n_total,
        "n_non_null_cells": n_non_null,
        "n_agree": n_agree,
    }


def compute_per_attr_metrics(
    prefill: dict[str, dict],
    blind: dict[str, dict],
) -> pd.DataFrame:
    """Per-attribute: n_total, n_non_null, agreement, refusal_rate, cohen_kappa."""
    codes = _common_codes(prefill, blind)
    attrs = sorted(set(_all_attrs(prefill)) | set(_all_attrs(blind)))

    rows = []
    for attr in attrs:
        n_total = 0
        n_non_null = 0
        n_agree = 0
        pre_vals: list[Any] = []
        bli_vals: list[Any] = []
        for code in codes:
            pre = _value(prefill.get(code, {}).get(attr))
            bli = _value(blind.get(code, {}).get(attr))
            n_total += 1
            if bli is None:
                continue
            n_non_null += 1
            if pre == bli:
                n_agree += 1
            pre_vals.append(pre if pre is not None else "__NULL__")
            bli_vals.append(bli)

        agreement = n_agree / n_non_null if n_non_null else 0.0
        refusal_rate = (n_total - n_non_null) / n_total if n_total else 0.0

        kappa: float | None
        try:
            kappa = float(cohen_kappa_score(pre_vals, bli_vals)) if len(pre_vals) >= 2 else None
        except ValueError:
            kappa = None

        rows.append({
            "attr": attr,
            "n_total": n_total,
            "n_non_null": n_non_null,
            "n_agree": n_agree,
            "agreement": agreement,
            "refusal_rate": refusal_rate,
            "cohen_kappa": kappa,
        })

    return pd.DataFrame(rows)


def compute_flip_direction(
    prefill: dict[str, dict],
    blind: dict[str, dict],
    silver_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-attribute: when blind disagrees with prefill, count flip direction.

    flip_to_silver: blind value matches silver but prefill does not.
    flip_away_silver: prefill matches silver but blind does not.
    flip_neither: neither matches silver.

    silver_df: columns = [code, attr1, attr2, ...] where attr* are silver values
    (string or None).
    """
    codes = _common_codes(prefill, blind)
    silver_lookup = silver_df.set_index("code") if "code" in silver_df.columns else silver_df
    attrs = sorted(set(_all_attrs(prefill)) | set(_all_attrs(blind)))

    rows = []
    for attr in attrs:
        if attr not in silver_lookup.columns:
            continue
        flip_to_silver = 0
        flip_away_silver = 0
        flip_neither = 0
        for code in codes:
            pre = _value(prefill.get(code, {}).get(attr))
            bli = _value(blind.get(code, {}).get(attr))
            if bli is None or pre == bli:
                continue
            if code not in silver_lookup.index:
                continue
            sil = silver_lookup.loc[code, attr]
            if pd.isna(sil):
                continue
            sil = str(sil) if not isinstance(sil, str) else sil
            if bli == sil and pre != sil:
                flip_to_silver += 1
            elif pre == sil and bli != sil:
                flip_away_silver += 1
            else:
                flip_neither += 1

        rows.append({
            "attr": attr,
            "flip_to_silver": flip_to_silver,
            "flip_away_silver": flip_away_silver,
            "flip_neither": flip_neither,
        })

    return pd.DataFrame(rows)
