"""NaiveBayes Layer 1.5 cascade predictor.

Extends predict_cascade with an NB pre-filter: for each (code, attr), if
NB confidence >= tau, use NB prediction; otherwise fall through to hybrid ML.

Entry point:
    predict_cascade_with_nb(df, cat, *, nb_threshold=0.85, use_hybrid=True)

Returns same schema as predict_cascade:
    code, attr, predicted, confidence, layer
where layer in {nb, regex, ml, abstain}.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.eval.cascade_predict import predict_cascade
from src.experiments.nb_layer import load_nb_model, predict_nb

logger = logging.getLogger(__name__)

_LAYER_NB = "nb"


def _build_text_series(df: pd.DataFrame) -> list[str]:
    """Build text strings from partner-available fields."""
    texts = []
    for _, row in df.iterrows():
        parts = []
        for col in ["product_name", "ingredients_text", "brands", "quantity"]:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        texts.append(" ".join(parts))
    return texts


def predict_cascade_with_nb(
    df: pd.DataFrame,
    cat: str,
    *,
    nb_threshold: float = 0.85,
    use_hybrid: bool = True,
    include_regex: bool = False,
    nb_only: bool = False,
) -> pd.DataFrame:
    """Predict using NB Layer 1.5 → ML hybrid cascade.

    Args:
        df: Products DataFrame (needs product_name, brands, ingredients_text, quantity, code).
        cat: Category slug (e.g. "pasta", "chocolate", "cheeses").
        nb_threshold: Minimum NB confidence to accept NB prediction.
        use_hybrid: Use hybrid XGB model when falling through NB.
        include_regex: Whether to include regex layer in ML cascade fallback.
        nb_only: If True, return only NB predictions (no ML fallback). Variant D.

    Returns:
        Long-format DataFrame: code, attr, predicted, confidence, layer.
    """
    df = df.copy()
    df["code"] = df["code"].astype(str)

    texts = _build_text_series(df)
    codes = df["code"].tolist()

    # Load NB models for all attrs
    from src.manual_label.schemas_loader import load_domain_attrs
    attrs = list(load_domain_attrs(cat))

    # Build NB predictions for all (code, attr) pairs
    nb_rows: list[dict] = []
    nb_fired_codes: dict[str, set[str]] = {}  # attr -> set of codes where NB fired

    for attr in attrs:
        nb, vec = load_nb_model(cat, attr)
        if nb is None:
            # No NB model for this attr — will fall through entirely
            continue

        preds = predict_nb(nb, vec, texts, tau=nb_threshold)
        fired_codes: set[str] = set()

        for code, (label, proba) in zip(codes, preds):
            if label is not None:
                nb_rows.append({
                    "code": code,
                    "attr": attr,
                    "predicted": label,
                    "confidence": proba,
                    "layer": _LAYER_NB,
                })
                fired_codes.add(code)
            else:
                nb_rows.append({
                    "code": code,
                    "attr": attr,
                    "predicted": None,
                    "confidence": proba,
                    "layer": "nb_abstain",
                })

        nb_fired_codes[attr] = fired_codes

    nb_df = pd.DataFrame(nb_rows) if nb_rows else pd.DataFrame(
        columns=["code", "attr", "predicted", "confidence", "layer"]
    )

    if nb_only:
        # Variant D: NB only — return NB results; abstains stay as abstain
        if nb_df.empty:
            return pd.DataFrame(columns=["code", "attr", "predicted", "confidence", "layer"])
        result = nb_df[nb_df["layer"] == _LAYER_NB].copy()
        # Fill in abstains for codes/attrs where NB didn't fire
        fired = set(zip(result["code"], result["attr"]))
        missing = []
        for attr in attrs:
            for code in codes:
                if (code, attr) not in fired:
                    missing.append({
                        "code": code,
                        "attr": attr,
                        "predicted": None,
                        "confidence": None,
                        "layer": "abstain",
                    })
        if missing:
            result = pd.concat([result, pd.DataFrame(missing)], ignore_index=True)
        return result

    # Get ML cascade predictions for fallback
    category_key = f"{cat}_stratified"
    ml_preds = predict_cascade(
        df, category=category_key,
        use_hybrid=use_hybrid, include_regex=include_regex
    )
    ml_preds["code"] = ml_preds["code"].astype(str)

    # Merge: NB takes priority where it fired; ML fills gaps
    nb_fired_df = nb_df[nb_df["layer"] == _LAYER_NB].copy()
    nb_fired_set = set(zip(nb_fired_df["code"], nb_fired_df["attr"]))

    ml_fallback = ml_preds[
        ~ml_preds.apply(lambda r: (r["code"], r["attr"]) in nb_fired_set, axis=1)
    ].copy()

    result = pd.concat([nb_fired_df, ml_fallback], ignore_index=True)

    n_nb = len(nb_fired_df)
    n_total = len(result)
    logger.info(
        "[%s] NB fired on %d/%d predictions (%.1f%%) @ tau=%.2f",
        cat, n_nb, n_total, 100 * n_nb / n_total if n_total > 0 else 0.0, nb_threshold,
    )

    return result
