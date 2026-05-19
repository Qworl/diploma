"""Pure helpers for the four Trek A1 validator signals.

Conventions
-----------
* All score functions return a float in [0, 1] (Mahalanobis returns raw
  non-negative distance; rescaling happens in the metrics layer).
* Higher score => more likely to be an error => stronger LLM-routing
  recommendation.
* Functions return ``None`` when the signal is not available for that
  cell (e.g. regex did not fire, XGB skipped attr).
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping

import numpy as np


def xgb_uncertainty_score(proba: np.ndarray | None) -> float | None:
    """1 - max(softmax) over the XGB output for a single cell."""
    if proba is None:
        return None
    arr = np.asarray(proba, dtype=float)
    if arr.size == 0:
        return None
    return float(1.0 - arr.max())


def _norm_for_compare(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    s = str(v).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return None
    if s in {"true", "yes"}:
        return True
    if s in {"false", "no"}:
        return False
    return s


def layer_disagreement_score(regex_pred: Any, ml_pred: Any) -> float | None:
    """1.0 if regex and ML disagree; 0.0 if they agree; None if either missing."""
    a = _norm_for_compare(regex_pred)
    b = _norm_for_compare(ml_pred)
    if a is None or b is None:
        return None
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0 if bool(a) == bool(b) else 1.0
    return 0.0 if str(a) == str(b) else 1.0


def load_per_attr_ece(
    category: str,
    attrs: tuple[str, ...],
    calib_dir: str,
) -> dict[str, float]:
    """Read per-attribute ECE from calibration JSONs.

    Prefers ``ece_calibrated`` when non-null, falls back to ``ece_raw``.
    Missing files are silently skipped.
    """
    out: dict[str, float] = {}
    for attr in attrs:
        path = os.path.join(calib_dir, f"{category}_{attr}_calibration.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            payload = json.load(f)
        ece = payload.get("ece_calibrated")
        if ece is None:
            ece = payload.get("ece_raw")
        if ece is None:
            continue
        out[attr] = float(ece)
    return out


def per_attr_ece_score(attr: str, ece_table: Mapping[str, float]) -> float | None:
    """Look up the constant per-attribute ECE for a cell."""
    return ece_table.get(attr)
