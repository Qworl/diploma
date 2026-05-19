"""
Supplementary §6.14.7 — cascade accuracy on independently audited pasta gold.

Loads the Trek D audited gold (`datasets/manual_label/pasta_gold_250.csv`,
n=239 pasta products, ~1700 audited cells), runs the regex → ML → Bayes
cascade on the same 239 codes, and reports:

    1. acc(cascade | confirmed cells)
       — sanity check: cascade trained on silver should agree with silver.
    2. acc(cascade | override ∪ manual_only cells)
       — independent eval: how does cascade fare on cells where silver was
         wrong or silent?
    3. Per-attribute and by-mode (blind / llm) breakdowns.

Notes
-----
* Cascade config = `regex_ml_bayes` (same as headline §6.14.7).
* Cascade abstains when no layer is confident enough; we report both
  `acc_on_covered` (denominator = covered cells) and `acc_on_audited`
  (denominator = all audited cells, abstains = wrong).
* Blind cohort is Pool A only (anchoring control, n=40 products); llm
  cohort spans all pools.

Usage
-----
    python -m src.eval.cascade_vs_audited_gold \\
        --gold datasets/manual_label/pasta_gold_250.csv \\
        --out datasets/processed/cascade_vs_audited_gold_pasta.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

from src.common import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MODELS_DIR,
    PROCESSED_DIR,
    setup_logging,
)
from src.manual_label.schemas_loader import load_domain_attrs
from src.pipeline.regex.extractor import RegexExtractor
from src.pipeline.schemas import PASTA_SCHEMA

logger = logging.getLogger(__name__)


# Per-domain attribute ordering (kept in sync with src.pipeline.schemas.<domain>).
PASTA_ATTRS = [
    "grain_type",
    "pasta_shape",
    "is_filled",
    "is_organic",
    "is_gluten_free",
    "is_vegan",
    "nutri_score_grade",
    "protein_class",
]
CHOCOLATE_ATTRS = [
    "chocolate_type",
    "cocoa_percentage",
    "contains_nuts",
    "chocolate_extra",
    "is_organic",
    "nutri_score_grade",
    "protein_class",
]
CHEESES_ATTRS = [
    "milk_source",
    "texture",
    "country_of_origin",
    "fat_class",
    "is_pdo",
    "is_organic",
    "is_ultra_processed",
]

_DOMAIN_ATTRS: dict[str, list[str]] = {
    "pasta": PASTA_ATTRS,
    "chocolate": CHOCOLATE_ATTRS,
    "cheeses": CHEESES_ATTRS,
}

# Per-domain: (silver_standard parquet basename, embeddings basename, regex category).
_DOMAIN_ASSETS: dict[str, dict[str, str]] = {
    "pasta": {
        "models_prefix": "pasta_stratified",
        "silver": "pasta_stratified_silver_standard.parquet",
        "emb": "pasta_stratified_embeddings.npy",
        "regex_category": "pasta",
    },
    "chocolate": {
        "models_prefix": "chocolate_stratified",
        "silver": "chocolate_stratified_silver_standard.parquet",
        "emb": "chocolate_stratified_embeddings.npy",
        "regex_category": "chocolate",
    },
    "cheeses": {
        "models_prefix": "cheeses_stratified",
        "silver": "cheeses_stratified_silver_standard.parquet",
        "emb": "cheeses_stratified_embeddings.npy",
        "regex_category": "cheeses",
    },
}

AUDITED_STATUSES = {"confirmed", "override", "manual_only"}
AUDITED_MODES = {"blind", "llm"}


# --------------------------------------------------------------------------- #
# Cascade inference (mirrors src.eval.run_experiments configuration regex_ml_bayes)
# --------------------------------------------------------------------------- #
def load_thresholds(category: str) -> dict:
    path = os.path.join(MODELS_DIR, f"{category}_thresholds.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def load_ml_models(category: str, attrs: list[str]) -> dict:
    models: dict = {}
    for attr in attrs:
        xgb_path = os.path.join(MODELS_DIR, f"{category}_{attr}_xgb.pkl")
        le_path = os.path.join(MODELS_DIR, f"{category}_{attr}_le.pkl")
        if os.path.exists(xgb_path):
            with open(xgb_path, "rb") as f:
                models[f"{attr}_xgb"] = pickle.load(f)
            if os.path.exists(le_path):
                with open(le_path, "rb") as f:
                    models[f"{attr}_le"] = pickle.load(f)
    return models


def load_bayesian(category: str):
    path = os.path.join(MODELS_DIR, f"{category}_bayesian.pkl")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        model = pickle.load(f)
    from pgmpy.inference import VariableElimination

    return model, VariableElimination(model)


def regex_layer(row, rx: RegexExtractor, category: str = "pasta") -> dict:
    name = str(row.get("product_name", "") or "")
    desc = str(row.get("generic_name", "") or "")
    qty = str(row.get("quantity", "") or "")
    results = rx.extract_all(name, desc, qty, category=category)
    return {k: (v.value, v.confidence) for k, v in results.items() if v.value is not None}


def ml_layer(
    embeddings: np.ndarray,
    idx: int,
    models: dict,
    attrs: list[str],
    thresholds: dict | None = None,
) -> dict:
    predictions: dict = {}
    X = embeddings[idx : idx + 1]
    for attr in attrs:
        xgb_key = f"{attr}_xgb"
        le_key = f"{attr}_le"
        if xgb_key not in models:
            continue
        clf = models[xgb_key]
        proba = clf.predict_proba(X)[0]
        max_idx = int(proba.argmax())
        confidence = float(proba[max_idx])
        threshold = (thresholds or {}).get(attr, DEFAULT_CONFIDENCE_THRESHOLD)
        if confidence < threshold:
            continue
        if le_key in models:
            value = models[le_key].inverse_transform([max_idx])[0]
        else:
            value = bool(max_idx)
        predictions[attr] = (value, confidence)
    return predictions


def bayes_layer(
    row,
    bayes_model,
    inference,
    targets: list[str],
    ml_predictions: dict | None = None,
    thresholds: dict | None = None,
) -> dict:
    if not bayes_model or not inference:
        return {}
    predictions: dict = {}
    evidence: dict = {}
    model_nodes = set(bayes_model.nodes())

    if "brand" in model_nodes:
        brand_val = str(row.get("brands", "other") or "other")
        cpd = bayes_model.get_cpds("brand")
        known = list(cpd.state_names["brand"])
        evidence["brand"] = brand_val if brand_val in known else "other"

    if ml_predictions:
        for attr, payload in ml_predictions.items():
            val, conf, _layer = payload
            if attr in model_nodes and conf >= 0.6:
                cpd = bayes_model.get_cpds(attr)
                known = list(cpd.state_names[attr])
                str_val = str(val)
                if str_val in known:
                    evidence[attr] = str_val

    for target in targets:
        if target in evidence or target not in model_nodes:
            continue
        try:
            result = inference.query([target], evidence=evidence)
            probs = {
                str(s): float(result.values[i])
                for i, s in enumerate(result.state_names[target])
            }
            best = max(probs, key=probs.get)
            conf = probs[best]
            threshold = (thresholds or {}).get(target, DEFAULT_CONFIDENCE_THRESHOLD)
            if conf >= threshold:
                predictions[target] = (best, conf)
        except Exception:
            pass
    return predictions


def run_cascade(
    silver_df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    domain: str = "pasta",
) -> pd.DataFrame:
    """Run regex_ml_bayes cascade. Returns long-format DF with one row per (code, attr)."""
    if domain not in _DOMAIN_ASSETS:
        raise KeyError(f"Unknown domain: {domain}")
    attrs = _DOMAIN_ATTRS[domain]
    assets = _DOMAIN_ASSETS[domain]
    rx = RegexExtractor()
    ml_models = load_ml_models(assets["models_prefix"], attrs)
    bayes_model, bayes_inference = load_bayesian(assets["models_prefix"])
    thresholds = load_thresholds(assets["models_prefix"])
    logger.info("Loaded ML models: %s", sorted({k.replace("_xgb", "").replace("_le", "") for k in ml_models}))
    logger.info("Bayesian model: %s", "loaded" if bayes_model else "not found")
    logger.info("Thresholds: %s", thresholds or "default")

    regex_cat = assets["regex_category"]
    rows = []
    for i, (_, row) in enumerate(silver_df.iterrows()):
        extracted: dict = {}
        # Layer 1 — regex
        for attr, (val, conf) in regex_layer(row, rx, regex_cat).items():
            if attr in attrs and attr not in extracted:
                extracted[attr] = (val, conf, "regex")
        # Layer 2 — ML
        for attr, (val, conf) in ml_layer(embeddings, i, ml_models, attrs, thresholds).items():
            if attr not in extracted:
                extracted[attr] = (val, conf, "ml")
        # Layer 3 — Bayes
        for attr, (val, conf) in bayes_layer(
            row, bayes_model, bayes_inference, attrs, ml_predictions=extracted, thresholds=thresholds
        ).items():
            if attr not in extracted:
                extracted[attr] = (val, conf, "bayes")

        code = str(row.get("code"))
        for attr in attrs:
            if attr in extracted:
                val, conf, layer = extracted[attr]
            else:
                val, conf, layer = None, 0.0, "none"
            rows.append({
                "code": code,
                "attr": attr,
                "cascade_pred": val,
                "cascade_conf": conf,
                "cascade_layer": layer,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Comparison utilities
# --------------------------------------------------------------------------- #
def _norm(v: Any) -> Any:
    """Normalize a value for string comparison (booleans, NaN, lower-case strings)."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"", "nan", "none", "null"}:
            return None
        if s in {"true", "yes"}:
            return True
        if s in {"false", "no"}:
            return False
        return s
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v


def _values_equal(a: Any, b: Any) -> bool:
    na, nb = _norm(a), _norm(b)
    if na is None or nb is None:
        return False
    if isinstance(na, bool) or isinstance(nb, bool):
        return bool(na) == bool(nb)
    return str(na).lower() == str(nb).lower()


def build_audited_long(gold: pd.DataFrame, attrs: list[str] | None = None) -> pd.DataFrame:
    if attrs is None:
        attrs = PASTA_ATTRS
    """Wide audit CSV → long format with one row per (code, attr) audited cell."""
    out = []
    for _, row in gold.iterrows():
        code = str(row["code"])
        for attr in attrs:
            status = row.get(f"manual_{attr}_status")
            mode = row.get(f"manual_{attr}_mode")
            manual_val = row.get(f"manual_{attr}")
            silver_val = row.get(f"silver_{attr}")
            out.append({
                "code": code,
                "attr": attr,
                "status": status,
                "mode": mode,
                "manual_value": manual_val,
                "silver_value": silver_val,
            })
    return pd.DataFrame(out)


def compute_metrics(joined: pd.DataFrame, attrs: list[str]) -> dict[str, Any]:
    """Compute accuracy aggregates over the joined audited+cascade table."""
    audited = joined[
        joined["status"].isin(AUDITED_STATUSES) & joined["mode"].isin(AUDITED_MODES)
    ].copy()
    audited["covered"] = audited["cascade_pred"].apply(lambda v: _norm(v) is not None)
    audited["correct"] = audited.apply(
        lambda r: _values_equal(r["cascade_pred"], r["manual_value"]), axis=1
    )

    def _agg(df: pd.DataFrame) -> dict[str, Any]:
        n = int(len(df))
        n_cov = int(df["covered"].sum())
        n_correct = int(df["correct"].sum())
        # acc_on_audited: abstain counted as wrong (denominator = n)
        # acc_on_covered: denominator = n_cov
        acc_on_audited = float(n_correct) / n if n else float("nan")
        acc_on_covered = float(n_correct) / n_cov if n_cov else float("nan")
        coverage = float(n_cov) / n if n else float("nan")
        return {
            "n": n,
            "n_covered": n_cov,
            "n_correct": n_correct,
            "coverage": coverage,
            "acc_on_audited": acc_on_audited,
            "acc_on_covered": acc_on_covered,
        }

    def _split(df: pd.DataFrame, label: str) -> dict[str, Any]:
        return {
            "overall": _agg(df),
            "by_mode": {
                m: _agg(df[df["mode"] == m]) for m in sorted(df["mode"].dropna().unique())
            },
            "by_attr": {
                a: _agg(df[df["attr"] == a]) for a in attrs
            },
        }

    confirmed = audited[audited["status"] == "confirmed"]
    override_or_manual_only = audited[audited["status"].isin({"override", "manual_only"})]

    metrics = {
        "all_audited": _split(audited, "all_audited"),
        "confirmed": _split(confirmed, "confirmed"),
        "override_or_manual_only": _split(override_or_manual_only, "override_or_manual_only"),
        "by_status": {
            s: _agg(audited[audited["status"] == s])
            for s in sorted(audited["status"].unique())
        },
    }
    return metrics, audited


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  n/a"
    return f"{x * 100:5.1f}%"


def print_report(metrics: dict[str, Any], audited: pd.DataFrame,
                 attrs: list[str] | None = None) -> None:
    print("=" * 78)
    print("CASCADE vs AUDITED GOLD")
    print("=" * 78)

    overall = metrics["all_audited"]["overall"]
    print(
        f"\nAll audited cells: n={overall['n']}  covered={overall['n_covered']} "
        f"({_fmt_pct(overall['coverage'])})  "
        f"acc_on_audited={_fmt_pct(overall['acc_on_audited'])}  "
        f"acc_on_covered={_fmt_pct(overall['acc_on_covered'])}"
    )

    for label, key in [
        ("[Sanity]  acc(cascade | confirmed)", "confirmed"),
        ("[Indep ]  acc(cascade | override ∪ manual_only)", "override_or_manual_only"),
    ]:
        block = metrics[key]
        ov = block["overall"]
        print(f"\n{label}")
        print(
            f"  overall: n={ov['n']:4d}  covered={ov['n_covered']:4d} "
            f"({_fmt_pct(ov['coverage'])})  "
            f"acc_on_audited={_fmt_pct(ov['acc_on_audited'])}  "
            f"acc_on_covered={_fmt_pct(ov['acc_on_covered'])}"
        )
        for mode, a in block["by_mode"].items():
            print(
                f"  mode={mode:5s}: n={a['n']:4d}  covered={a['n_covered']:4d} "
                f"({_fmt_pct(a['coverage'])})  "
                f"acc_on_audited={_fmt_pct(a['acc_on_audited'])}  "
                f"acc_on_covered={_fmt_pct(a['acc_on_covered'])}"
            )

    print("\nPer-attribute breakdown (override ∪ manual_only):")
    print(
        f"  {'attr':<22}{'n':>5}{'covered':>10}{'acc_aud':>10}{'acc_cov':>10}"
    )
    for attr, a in metrics["override_or_manual_only"]["by_attr"].items():
        print(
            f"  {attr:<22}{a['n']:>5d}{a['n_covered']:>10d}"
            f"{_fmt_pct(a['acc_on_audited']):>10s}{_fmt_pct(a['acc_on_covered']):>10s}"
        )

    print("\nPer-attribute breakdown (confirmed — sanity):")
    print(
        f"  {'attr':<22}{'n':>5}{'covered':>10}{'acc_aud':>10}{'acc_cov':>10}"
    )
    for attr, a in metrics["confirmed"]["by_attr"].items():
        print(
            f"  {attr:<22}{a['n']:>5d}{a['n_covered']:>10d}"
            f"{_fmt_pct(a['acc_on_audited']):>10s}{_fmt_pct(a['acc_on_covered']):>10s}"
        )

    print("\nBy status (for reference):")
    for status, a in metrics["by_status"].items():
        print(
            f"  status={status:<14}: n={a['n']:4d}  covered={a['n_covered']:4d}  "
            f"acc_on_audited={_fmt_pct(a['acc_on_audited'])}  "
            f"acc_on_covered={_fmt_pct(a['acc_on_covered'])}"
        )
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        default="pasta",
        choices=list(_DOMAIN_ATTRS),
        help="Schema domain (pasta/chocolate/cheeses)",
    )
    parser.add_argument(
        "--gold",
        default=None,
        help="Audited gold CSV (defaults to datasets/manual_label/<domain>_gold_239.csv "
             "or pasta_gold_250.csv for pasta).",
    )
    parser.add_argument(
        "--cascade",
        default=None,
        help=(
            "Optional pre-computed cascade parquet with columns "
            "(code, attr, cascade_pred[, cascade_layer, cascade_conf]). "
            "If omitted, cascade is run on-the-fly using <domain>_stratified models."
        ),
    )
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: cascade_vs_audited_gold_<domain>.json)")
    parser.add_argument("--joined-out", default=None,
                        help="Output parquet path for the joined audited+cascade table")
    args = parser.parse_args()

    domain = args.domain
    attrs = _DOMAIN_ATTRS[domain]
    assets = _DOMAIN_ASSETS[domain]

    if args.gold is None:
        if domain == "pasta":
            args.gold = "datasets/manual_label/pasta_gold_250.csv"
        else:
            args.gold = f"datasets/manual_label/{domain}_gold_239.csv"
    if args.out is None:
        args.out = os.path.join(PROCESSED_DIR, f"cascade_vs_audited_gold_{domain}.json")
    if args.joined_out is None:
        args.joined_out = os.path.join(PROCESSED_DIR, f"cascade_vs_audited_gold_{domain}.parquet")

    # --- Load audited gold ---
    gold = pd.read_csv(args.gold, dtype={"code": str})
    logger.info("Loaded gold: %d products from %s", len(gold), args.gold)
    gold_codes = set(gold["code"].astype(str))

    # --- Cascade predictions ---
    if args.cascade and os.path.exists(args.cascade):
        cascade_df = pd.read_parquet(args.cascade)
        cascade_df["code"] = cascade_df["code"].astype(str)
        logger.info("Loaded cascade preds: %d rows from %s", len(cascade_df), args.cascade)
        needed_cols = {"code", "attr", "cascade_pred"}
        missing = needed_cols - set(cascade_df.columns)
        if missing:
            raise ValueError(f"Cascade parquet missing columns: {missing}")
        for opt in ("cascade_layer", "cascade_conf"):
            if opt not in cascade_df.columns:
                cascade_df[opt] = None
        cascade_df = cascade_df[cascade_df["code"].isin(gold_codes)]
    else:
        silver_path = os.path.join(PROCESSED_DIR, assets["silver"])
        emb_path = os.path.join(PROCESSED_DIR, assets["emb"])
        silver_df = pd.read_parquet(silver_path).reset_index(drop=True)
        silver_df["code"] = silver_df["code"].astype(str)
        embeddings = np.load(emb_path)
        if len(embeddings) != len(silver_df):
            raise RuntimeError(
                f"Embeddings ({len(embeddings)}) and silver ({len(silver_df)}) length mismatch"
            )
        mask = silver_df["code"].isin(gold_codes).values
        sub_silver = silver_df.loc[mask].reset_index(drop=True)
        sub_emb = embeddings[mask]
        logger.info("Running cascade on %d gold-overlapping silver rows (domain=%s)",
                    len(sub_silver), domain)
        cascade_df = run_cascade(sub_silver, sub_emb, domain=domain)
        cascade_df["cascade_pred"] = cascade_df["cascade_pred"].apply(
            lambda v: None if v is None else str(v)
        )
        raw_out = os.path.join(
            PROCESSED_DIR, f"cascade_preds_{domain}_gold.parquet"
        )
        cascade_df.to_parquet(raw_out, index=False)
        logger.info("Saved cascade predictions → %s", raw_out)

    covered_codes = set(cascade_df["code"]) & gold_codes
    logger.info("Cascade covers %d / %d gold codes", len(covered_codes), len(gold_codes))

    # --- Build audited long-format and join ---
    audited_long = build_audited_long(gold, attrs)
    joined = audited_long.merge(
        cascade_df[["code", "attr", "cascade_pred", "cascade_conf", "cascade_layer"]],
        on=["code", "attr"],
        how="left",
    )
    for col in ("manual_value", "silver_value", "cascade_pred"):
        joined[col] = joined[col].apply(lambda v: None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v))
    joined.to_parquet(args.joined_out, index=False)
    logger.info("Saved joined table → %s", args.joined_out)

    metrics, audited = compute_metrics(joined, attrs)
    print_report(metrics, audited, attrs)

    payload = {
        "domain": domain,
        "gold_path": args.gold,
        "n_gold_products": int(len(gold)),
        "n_gold_codes_with_cascade": int(len(covered_codes)),
        "audited_statuses": sorted(AUDITED_STATUSES),
        "audited_modes": sorted(AUDITED_MODES),
        "cascade_config": f"regex_ml_bayes ({assets['models_prefix']} models)",
        "metrics": metrics,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Saved metrics JSON → %s", args.out)


if __name__ == "__main__":
    main()
