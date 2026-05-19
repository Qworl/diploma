"""Uniform cascade predictor over an arbitrary product DataFrame.

Loads per-category XGBoost models, regex extractor, and (optionally) Bayesian
network. Produces long-format predictions: one row per (code, attr) with
predicted value, confidence, and the layer that produced it.

The Bayesian layer is included in the layer vocabulary because it appears in
§6.13 historically but is NOT in production (see spec §6.1). Callers that want
regex→ML-only should pass `include_bayes=False` (default).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.common import DEFAULT_CONFIDENCE_THRESHOLD, MODELS_DIR, get_embeddings
from src.manual_label.schemas_loader import load_domain_attrs
from src.pipeline.ml.infer import load_classifier, load_thresholds, predict_with_threshold
from src.pipeline.regex.extractor import RegexExtractor

import os
import pickle

logger = logging.getLogger(__name__)

_LAYER_REGEX = "regex"
_LAYER_ML = "ml"
_LAYER_ABSTAIN = "abstain"


def _load_hybrid_classifier(category: str, attr: str):
    """Load hybrid XGB model if it exists; return (clf, le) or (None, None)."""
    base = os.path.join(MODELS_DIR, f"{category}_{attr}")
    xgb_path = base + "_xgb_hybrid.pkl"
    le_path = base + "_le_hybrid.pkl"
    if not os.path.exists(xgb_path):
        return None, None
    with open(xgb_path, "rb") as f:
        clf = pickle.load(f)
    le = None
    if os.path.exists(le_path):
        with open(le_path, "rb") as f:
            le = pickle.load(f)
    return clf, le


def _build_regex_cache(
    extractor: RegexExtractor,
    df: pd.DataFrame,
    domain: str,
) -> dict[str, dict[str, object]]:
    """Run regex extraction ONCE per row, returning {code: {attr: value}}."""
    cache: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        code = str(row["code"])
        product_name = str(row.get("product_name") or "")
        ingredients = str(row.get("ingredients_text") or "")
        quantity = str(row.get("quantity") or "")
        results = extractor.extract_all(
            product_name=product_name,
            description=ingredients,
            quantity=quantity,
            category=domain,
        )
        # extract_all returns dict[attr, ExtractionResult]; pull .value where confident
        attr_vals: dict[str, object] = {}
        for attr, result in results.items():
            if result.confidence > 0.0 and result.value is not None:
                attr_vals[attr] = result.value
        cache[code] = attr_vals
    return cache


def predict_cascade(
    df: pd.DataFrame,
    *,
    category: str,
    include_bayes: bool = False,
    threshold_override: Optional[dict] = None,
    use_hybrid: bool = False,
    include_regex: bool = True,
) -> pd.DataFrame:
    """Predict regex → ML cascade for every (code, attr) cell.

    `category` is the full stratified key, e.g. "pasta_stratified".
    Returns long-format DataFrame with columns:
        code (str), attr (str), predicted (str|None),
        confidence (float|None), layer (str ∈ {regex, ml, abstain})

    When `include_regex=False`, skip regex extraction entirely — all predictions
    come from ML (or abstain). Useful for ablation studies.
    """
    domain = category.replace("_stratified", "").replace("_extended", "")
    attrs = list(load_domain_attrs(domain))
    if not attrs:
        raise ValueError(f"No attrs registered for domain {domain}")

    df = df.copy()
    df["code"] = df["code"].astype(str)

    extractor = RegexExtractor()
    thresholds = threshold_override if threshold_override is not None else load_thresholds(category)

    text_cols = ["product_name", "brands", "ingredients_text", "quantity"]
    for c in text_cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")

    # Build regex cache once per row — O(rows), not O(rows × attrs)
    # When include_regex=False, skip extraction and return empty cache
    if include_regex:
        regex_cache = _build_regex_cache(extractor, df, domain)
    else:
        regex_cache = {str(row["code"]): {} for _, row in df.iterrows()}

    text_blob = (
        df["product_name"] + " " + df["brands"] + " "
        + df["ingredients_text"] + " " + df["quantity"]
    ).tolist()

    embeddings: Optional[np.ndarray] = None
    rows: list[dict] = []

    for attr in attrs:
        # Look up pre-computed regex results for this attribute
        regex_vals = [
            regex_cache.get(code, {}).get(attr)
            for code in df["code"].tolist()
        ]

        # Try to load the ML model for this attribute
        # When use_hybrid=True, prefer hybrid model but fall back to silver if not found
        if use_hybrid:
            model, le = _load_hybrid_classifier(category, attr)
            if model is None:
                try:
                    model, le = load_classifier(category, attr)
                except (FileNotFoundError, OSError):
                    model, le = None, None
        else:
            try:
                model, le = load_classifier(category, attr)
            except (FileNotFoundError, OSError):
                model, le = None, None

        attr_thr = thresholds.get(attr, DEFAULT_CONFIDENCE_THRESHOLD)

        for i, (code, regex_val) in enumerate(zip(df["code"].tolist(), regex_vals)):
            if regex_val is not None and str(regex_val) != "":
                rows.append({
                    "code": code,
                    "attr": attr,
                    "predicted": str(regex_val),
                    "confidence": 1.0,
                    "layer": _LAYER_REGEX,
                })
                continue

            if model is None:
                rows.append({
                    "code": code,
                    "attr": attr,
                    "predicted": None,
                    "confidence": None,
                    "layer": _LAYER_ABSTAIN,
                })
                continue

            # Compute embeddings lazily (once for all attrs)
            if embeddings is None:
                embeddings = get_embeddings(text_blob)

            label, conf = predict_with_threshold(model, le, embeddings[i], attr_thr)
            if label is None:
                rows.append({
                    "code": code,
                    "attr": attr,
                    "predicted": None,
                    "confidence": conf,
                    "layer": _LAYER_ABSTAIN,
                })
            else:
                rows.append({
                    "code": code,
                    "attr": attr,
                    "predicted": str(label),
                    "confidence": conf,
                    "layer": _LAYER_ML,
                })

    return pd.DataFrame(rows)
