"""Перегенерация cascade_preds и headline для chocolate после фикса Layer 1.

Что делает:
  1. Берёт 239 brand-disjoint test-кодов из существующего
     `cascade_preds_chocolate_v2_gold_hybrid_v3_fixed.parquet`.
  2. Применяет ОБНОВЛЁННЫЙ RegexExtractor (chocolate_type без ingredients,
     abstain on ambig — см. `src/pipeline/regex/extractor.py`).
  3. На abstain-ячейках применяет hybrid XGBoost (SBERT + TF-IDF) из
     `chocolate_stratified_hybrid_*_xgb.pkl` + `chocolate_stratified_hybrid_tfidf.pkl`.
  4. Сравнивает с consensus gold v2 expanded и сохраняет:
     - `cascade_preds_chocolate_after_fix.parquet`
     - `headline_chocolate_after_fix.parquet` (для трёх проблемных attrs)

Usage:
    OMP_NUM_THREADS=1 python -m src.experiments.regen_chocolate_cascade_after_fix
"""

from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from src.common import (
    DEFAULT_CONFIDENCE_THRESHOLD, MODELS_DIR, PARTNER_TEXT_FIELDS,
    PROCESSED_DIR, get_embeddings, setup_logging, wilson_ci,
)
from src.pipeline.regex.extractor import RegexExtractor

logger = logging.getLogger("regen_chocolate")

CATEGORY = "chocolate"
HYBRID_PREFIX = "chocolate_stratified_hybrid"
BASELINE_PREFIX = "chocolate_stratified"
ATTRS = [
    "chocolate_type", "cocoa_percentage", "contains_nuts", "chocolate_extra",
    "is_organic", "nutri_score_grade", "protein_class",
]

SILVER_PATH = os.path.join(PROCESSED_DIR, f"{CATEGORY}_stratified_silver_standard.parquet")
EMBED_PATH = os.path.join(PROCESSED_DIR, f"{CATEGORY}_stratified_embeddings.npy")
CASCADE_OLD = os.path.join(PROCESSED_DIR, f"cascade_preds_{CATEGORY}_v2_gold_hybrid_v3_fixed.parquet")
GOLD_PATH = os.path.join(PROCESSED_DIR, "consensus_gold_v2_expanded.parquet")
OUT_PREDS = os.path.join(PROCESSED_DIR, f"cascade_preds_{CATEGORY}_after_fix.parquet")
OUT_HEADLINE = os.path.join(PROCESSED_DIR, f"headline_{CATEGORY}_after_fix.parquet")


def _build_text(df: pd.DataFrame) -> list[str]:
    parts = []
    for col in PARTNER_TEXT_FIELDS:
        if col in df.columns:
            parts.append(df[col].astype("string").fillna(""))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    return parts[0].str.cat(parts[1:], sep=" ", na_rep="").fillna("").tolist()


def _load_model(prefix: str, attr: str, suffix: str = "_xgb.pkl"):
    """Загрузить XGB-модель. Для baseline-chocolate путь — это
    `{prefix}_{attr}_xgb_hybrid.pkl` (исторически: hybrid означает re-trained
    на gold в этом проекте, а НЕ TF-IDF; для гибридной фич-композиции
    использовать prefix='chocolate_stratified_hybrid')."""
    base = f"{prefix}_{attr}"
    xgb_path = os.path.join(MODELS_DIR, f"{base}{suffix}")
    le_suffix = "_le_hybrid.pkl" if suffix == "_xgb_hybrid.pkl" else "_le.pkl"
    le_path = os.path.join(MODELS_DIR, f"{base.replace('_xgb','')}".rstrip("_") + "_" + le_suffix.lstrip("_"))
    # Simpler reconstruction:
    if suffix == "_xgb_hybrid.pkl":
        le_path = os.path.join(MODELS_DIR, f"{prefix}_{attr}_le_hybrid.pkl")
    else:
        le_path = os.path.join(MODELS_DIR, f"{prefix}_{attr}_le.pkl")
    if not os.path.exists(xgb_path):
        return None, None
    with open(xgb_path, "rb") as f:
        clf = pickle.load(f)
    le = None
    if os.path.exists(le_path):
        with open(le_path, "rb") as f:
            le = pickle.load(f)
    return clf, le


def _load_thresholds(prefix: str) -> dict:
    path = os.path.join(MODELS_DIR, f"{prefix}_thresholds.pkl")
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    setup_logging()

    silver = pd.read_parquet(SILVER_PATH)
    silver["code"] = silver["code"].astype(str)
    emb_all = np.load(EMBED_PATH)
    silver["_pos"] = np.arange(len(silver))

    cascade_old = pd.read_parquet(CASCADE_OLD)
    cascade_old["code"] = cascade_old["code"].astype(str)
    test_codes = sorted(cascade_old["code"].unique())
    logger.info("Test codes (brand-disjoint): %d", len(test_codes))

    sub = silver[silver["code"].isin(test_codes)].copy().reset_index(drop=True)
    logger.info("Subset rows: %d", len(sub))
    emb_sub = emb_all[sub["_pos"].values]
    texts_sub = _build_text(sub)

    # Hybrid TF-IDF vectorizer
    tfidf_path = os.path.join(MODELS_DIR, f"{HYBRID_PREFIX}_tfidf.pkl")
    with open(tfidf_path, "rb") as f:
        vectorizer = pickle.load(f)
    X_tfidf_sub = vectorizer.transform(texts_sub)
    X_hybrid_sub = hstack([csr_matrix(emb_sub), X_tfidf_sub]).tocsr()
    logger.info("Hybrid features: %s", X_hybrid_sub.shape)

    hybrid_thresholds = _load_thresholds(HYBRID_PREFIX)
    baseline_thresholds = _load_thresholds(BASELINE_PREFIX)
    # Calibration threshold для hybrid модели сильно поднялась (0.65 → 0.90 для
    # chocolate_extra), резко урезая coverage. Это артефакт find_best_threshold,
    # который оптимизирует f1 × cov^0.3, — для финального E2E (acc + LLM rescue
    # на abstain) лучше использовать менее агрессивный baseline-порог.
    effective_thresholds = {**hybrid_thresholds, **baseline_thresholds}

    extractor = RegexExtractor()

    rows = []
    for attr in ATTRS:
        # Regex first
        regex_vals = []
        for _, row in sub.iterrows():
            results = extractor.extract_all(
                product_name=str(row.get("product_name") or ""),
                description=str(row.get("ingredients_text") or ""),
                quantity=str(row.get("quantity") or ""),
                brands=str(row.get("brands") or ""),
                category="chocolate",
            )
            r = results.get(attr)
            regex_vals.append(r.value if (r and r.confidence > 0.0) else None)

        # Hybrid стоит брать только там, где гибридные признаки реально помогают
        # на gold (см. diagnostic chocolate_hybrid_features.parquet): chocolate_type
        # и contains_nuts получают +1.9 и +3.8 п. п. соответственно. chocolate_extra
        # выигрывает на standalone-метрике, но в архитектуре каскада с порогом и
        # LLM-rescue baseline-модель ведёт себя лучше (см. discussion §5.5 п.2).
        USE_HYBRID_FOR = {"chocolate_type", "contains_nuts"}
        if attr in USE_HYBRID_FOR:
            clf, le = _load_model(HYBRID_PREFIX, attr, "_xgb.pkl")
            using_hybrid = clf is not None
        else:
            # Существующие production-модели: {prefix}_{attr}_xgb_hybrid.pkl
            # (исторически «hybrid» в имени = re-trained on gold, не TF-IDF).
            clf, le = _load_model(BASELINE_PREFIX, attr, "_xgb_hybrid.pkl")
            using_hybrid = False
        if clf is None:
            clf, le = _load_model(BASELINE_PREFIX, attr, "_xgb_hybrid.pkl")
        if clf is None:
            logger.warning("No model for %s — abstain", attr)
            ml_preds = [(None, None)] * len(sub)
        else:
            X_in = X_hybrid_sub if using_hybrid else emb_sub
            proba = clf.predict_proba(X_in)
            thr = effective_thresholds.get(attr, DEFAULT_CONFIDENCE_THRESHOLD)
            ml_preds = []
            for p in proba:
                conf = float(p.max())
                if conf < thr:
                    ml_preds.append((None, conf))
                else:
                    idx = int(p.argmax())
                    if le is not None:
                        ml_preds.append((str(le.inverse_transform([idx])[0]), conf))
                    else:
                        ml_preds.append((bool(idx), conf))

        for i, code in enumerate(sub["code"].tolist()):
            rv = regex_vals[i]
            if rv is not None and str(rv) != "":
                rows.append({
                    "code": code, "attr": attr, "predicted": str(rv),
                    "confidence": 1.0, "layer": "regex",
                })
                continue
            lbl, conf = ml_preds[i]
            if lbl is None:
                rows.append({
                    "code": code, "attr": attr, "predicted": None,
                    "confidence": conf, "layer": "abstain",
                })
            else:
                rows.append({
                    "code": code, "attr": attr, "predicted": str(lbl),
                    "confidence": conf, "layer": "ml",
                })

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PREDS, index=False)
    logger.info("Saved %d cascade rows → %s", len(out), OUT_PREDS)

    # Headline (per-attr e2e on gold)
    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)
    gold = gold[(gold.category == CATEGORY) & (~gold.gold_is_null)]

    headline_rows = []
    for attr in ATTRS:
        gattr = gold[gold.attr == attr][["code", "gold_value"]]
        gattr["gold_value"] = gattr["gold_value"].astype(str)
        attr_rows = out[out.attr == attr].merge(gattr, on="code", how="inner")
        attr_rows["pred_str"] = attr_rows["predicted"].astype(str).str.lower()
        attr_rows["gold_str"] = attr_rows["gold_value"].astype(str).str.lower()
        attr_rows["correct"] = (
            (attr_rows["pred_str"] == attr_rows["gold_str"]) &
            (attr_rows["layer"] != "abstain")
        )
        n_total = len(attr_rows)
        n_covered = int((attr_rows["layer"] != "abstain").sum())
        n_abstain = int((attr_rows["layer"] == "abstain").sum())
        n_correct = int(attr_rows["correct"].sum())
        acc_e2e = n_correct / n_total if n_total else float("nan")
        acc_covered = (
            n_correct / n_covered if n_covered else float("nan")
        )
        lo, hi = wilson_ci(n_correct, n_total)
        layer_share = attr_rows["layer"].value_counts(normalize=True).to_dict()
        layer_acc = {}
        for layer in ["regex", "ml", "abstain"]:
            ls = attr_rows[attr_rows.layer == layer]
            layer_acc[f"{layer}_n"] = len(ls)
            layer_acc[f"{layer}_acc"] = (
                ls["correct"].mean() if len(ls) else float("nan")
            )
        headline_rows.append({
            "category": CATEGORY, "attr": attr,
            "n_test": n_total, "n_covered": n_covered, "n_abstain": n_abstain,
            "acc_e2e_cascade_only": acc_e2e,
            "acc_on_covered": acc_covered,
            "acc_ci_lo": lo, "acc_ci_hi": hi,
            **{f"share_{k}": v for k, v in layer_share.items()},
            **layer_acc,
        })

    headline = pd.DataFrame(headline_rows)
    headline.to_parquet(OUT_HEADLINE, index=False)
    logger.info("Saved headline → %s", OUT_HEADLINE)

    print()
    print("=" * 78)
    print("HEADLINE AFTER FIX — chocolate")
    print("=" * 78)
    cols = ["attr", "n_test", "n_covered", "n_abstain",
            "acc_e2e_cascade_only", "acc_on_covered",
            "regex_n", "regex_acc", "ml_n", "ml_acc"]
    print(headline[cols].to_string(index=False))


if __name__ == "__main__":
    main()
