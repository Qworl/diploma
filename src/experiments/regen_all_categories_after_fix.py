"""Пересчёт cascade headline для всех 3 категорий после фиксов §3.3.7.4.

Применяет:
  1. Обновлённый RegexExtractor (с фиксами chocolate_type/extra/nuts).
  2. ML hybrid_features-модели для chocolate_type/contains_nuts (TF-IDF+SBERT).
  3. Пониженные пороги Layer 2 для атрибутов с высоким abstain rate
     (chocolate_extra, cheeses/texture, cheeses/is_organic,
      cheeses/is_ultra_processed, pasta/grain_type).
  4. Базовые модели и пороги для остальных атрибутов.

Сохраняет:
  - cascade_preds_{cat}_after_fix.parquet
  - headline_v3e_after_fix.parquet (одна сводная)
  - grand_acc_summary_after_fix.parquet

Usage:
    OMP_NUM_THREADS=1 python -m src.experiments.regen_all_categories_after_fix
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
    PROCESSED_DIR, setup_logging, wilson_ci,
)
from src.pipeline.regex.extractor import RegexExtractor

logger = logging.getLogger("regen_all")

# Категории, пороги Layer 2 и используемые ML-модели.
# THRESHOLD_OVERRIDES: понижение find_best_threshold там, где он был слишком
# консервативным (см. §3.3.7.4 analysis).
THRESHOLD_OVERRIDES = {
    ("chocolate", "chocolate_extra"): 0.65,
    ("cheeses", "texture"): 0.40,
    ("cheeses", "is_organic"): 0.40,
    ("cheeses", "is_ultra_processed"): 0.50,
    ("pasta", "grain_type"): 0.50,
}
# Гибридные TF-IDF+SBERT модели — только там, где они помогают.
HYBRID_MODELS = {("chocolate", "chocolate_type"), ("chocolate", "contains_nuts")}

ATTRS_BY_CAT = {
    "pasta": ["grain_type", "pasta_shape", "is_filled", "is_organic", "is_gluten_free",
              "is_vegan", "nutri_score_grade", "protein_class"],
    "chocolate": ["chocolate_type", "cocoa_percentage", "contains_nuts", "chocolate_extra",
                  "is_organic", "nutri_score_grade", "protein_class"],
    "cheeses": ["milk_source", "texture", "country_of_origin", "fat_class",
                "is_pdo", "is_organic", "is_ultra_processed"],
}


def _build_text(df: pd.DataFrame) -> list[str]:
    parts = []
    for col in PARTNER_TEXT_FIELDS:
        if col in df.columns:
            parts.append(df[col].astype("string").fillna(""))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    return parts[0].str.cat(parts[1:], sep=" ", na_rep="").fillna("").tolist()


def _load(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_baseline_thresholds(cat: str) -> dict:
    thr = _load(os.path.join(MODELS_DIR, f"{cat}_stratified_thresholds.pkl")) or {}
    return {k: float(v) for k, v in thr.items()}


def _predict_for_cat(cat: str, extractor: RegexExtractor):
    silver = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{cat}_stratified_silver_standard.parquet"))
    silver["code"] = silver["code"].astype(str)
    emb_all = np.load(os.path.join(PROCESSED_DIR, f"{cat}_stratified_embeddings.npy"))
    silver["_pos"] = np.arange(len(silver))

    cp_old = pd.read_parquet(os.path.join(PROCESSED_DIR, f"cascade_preds_{cat}_v2_gold_hybrid_v3_fixed.parquet"))
    cp_old["code"] = cp_old["code"].astype(str)
    test_codes = sorted(cp_old["code"].unique())
    sub = silver[silver["code"].isin(test_codes)].copy().reset_index(drop=True)
    emb_sub = emb_all[sub["_pos"].values]
    texts_sub = _build_text(sub)
    codes_sub = sub["code"].tolist()
    logger.info("[%s] %d test codes", cat, len(sub))

    baseline_thr = _load_baseline_thresholds(f"{cat}_stratified")

    # Hybrid (TF-IDF) — только для chocolate
    hybrid_vec = None
    hybrid_thr = {}
    if cat == "chocolate":
        hybrid_vec = _load(os.path.join(MODELS_DIR, "chocolate_stratified_hybrid_tfidf.pkl"))
        hybrid_thr = _load(os.path.join(MODELS_DIR, "chocolate_stratified_hybrid_thresholds.pkl")) or {}
        X_tfidf = hybrid_vec.transform(texts_sub) if hybrid_vec is not None else None
        X_hybrid = hstack([csr_matrix(emb_sub), X_tfidf]).tocsr() if X_tfidf is not None else None

    rows = []
    for attr in ATTRS_BY_CAT[cat]:
        # Layer 1 (regex) — единый обновлённый extractor
        regex_vals = []
        for _, row in sub.iterrows():
            results = extractor.extract_all(
                product_name=str(row.get("product_name") or ""),
                description=str(row.get("ingredients_text") or ""),
                quantity=str(row.get("quantity") or ""),
                brands=str(row.get("brands") or ""),
                category=cat,
            )
            r = results.get(attr)
            regex_vals.append(r.value if (r and r.confidence > 0.0) else None)

        # Layer 2 (ML)
        use_hybrid = (cat, attr) in HYBRID_MODELS
        if use_hybrid:
            clf = _load(os.path.join(MODELS_DIR, f"chocolate_stratified_hybrid_{attr}_xgb.pkl"))
            le = _load(os.path.join(MODELS_DIR, f"chocolate_stratified_hybrid_{attr}_le.pkl"))
            X_in = X_hybrid
        else:
            clf = _load(os.path.join(MODELS_DIR, f"{cat}_stratified_{attr}_xgb_hybrid.pkl"))
            le = _load(os.path.join(MODELS_DIR, f"{cat}_stratified_{attr}_le_hybrid.pkl"))
            X_in = emb_sub
        if clf is None:
            logger.warning("[%s/%s] no ML model, abstain", cat, attr)
            ml_preds = [(None, None)] * len(sub)
        else:
            proba = clf.predict_proba(X_in)
            thr_override = THRESHOLD_OVERRIDES.get((cat, attr))
            thr = thr_override if thr_override is not None else baseline_thr.get(attr, DEFAULT_CONFIDENCE_THRESHOLD)
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

        for i, code in enumerate(codes_sub):
            rv = regex_vals[i]
            if rv is not None and str(rv) != "":
                rows.append({"code": code, "attr": attr, "predicted": str(rv),
                             "confidence": 1.0, "layer": "regex"})
                continue
            lbl, conf = ml_preds[i]
            if lbl is None:
                rows.append({"code": code, "attr": attr, "predicted": None,
                             "confidence": conf, "layer": "abstain"})
            else:
                rows.append({"code": code, "attr": attr, "predicted": str(lbl),
                             "confidence": conf, "layer": "ml"})

    return pd.DataFrame(rows)


def main():
    setup_logging()
    extractor = RegexExtractor()
    gold = pd.read_parquet(os.path.join(PROCESSED_DIR, "consensus_gold_v2_expanded.parquet"))
    gold["code"] = gold["code"].astype(str)
    gold = gold[~gold.gold_is_null]

    headline_rows = []
    all_preds = []
    for cat in ["pasta", "chocolate", "cheeses"]:
        preds = _predict_for_cat(cat, extractor)
        preds["category"] = cat
        all_preds.append(preds)
        out_path = os.path.join(PROCESSED_DIR, f"cascade_preds_{cat}_after_fix.parquet")
        preds.to_parquet(out_path, index=False)
        logger.info("[%s] saved %d rows → %s", cat, len(preds), out_path)

        # Per-attr headline
        cat_gold = gold[gold.category == cat]
        for attr in ATTRS_BY_CAT[cat]:
            gattr = cat_gold[cat_gold.attr == attr][["code", "gold_value"]].copy()
            gattr["gold_value"] = gattr["gold_value"].astype(str).str.lower()
            attr_rows = preds[preds.attr == attr].merge(gattr, on="code", how="inner")
            attr_rows["pred_lower"] = attr_rows["predicted"].astype(str).str.lower()
            attr_rows["correct"] = ((attr_rows["pred_lower"] == attr_rows["gold_value"])
                                     & (attr_rows["layer"] != "abstain"))
            n_total = len(attr_rows)
            n_covered = int((attr_rows["layer"] != "abstain").sum())
            n_correct = int(attr_rows["correct"].sum())
            acc_e2e = n_correct / n_total if n_total else float("nan")
            acc_cov = n_correct / n_covered if n_covered else float("nan")
            lo, hi = wilson_ci(n_correct, n_total)
            layer_share = attr_rows["layer"].value_counts().to_dict()
            headline_rows.append({
                "category": cat, "attr": attr,
                "n_test_cells": n_total, "n_covered": n_covered,
                "n_abstain": n_total - n_covered,
                "acc_e2e": acc_e2e, "acc_on_covered": acc_cov,
                "ci_lo": lo, "ci_hi": hi,
                **{f"share_{k}": v / n_total for k, v in layer_share.items() if n_total}
            })

    headline = pd.DataFrame(headline_rows)
    headline.to_parquet(os.path.join(PROCESSED_DIR, "headline_v3e_after_fix.parquet"), index=False)
    logger.info("Saved headline_v3e_after_fix.parquet")

    # === Grand aggregate ===
    summary_rows = []
    old_h = pd.read_parquet(os.path.join(PROCESSED_DIR, "headline_v3e_final.parquet"))

    # Cascade-only e2e
    merged = old_h.merge(headline[["category", "attr", "acc_e2e", "n_test_cells"]],
                          on=["category", "attr"], how="left", suffixes=("_old", "_new"))
    merged["old_correct"] = (merged["n_test_cells_old"] * merged["acc_oracle_cat"]).round().astype(int)
    merged["new_correct"] = (merged["n_test_cells_old"] * merged["acc_e2e"].fillna(merged["acc_oracle_cat"])).round().astype(int)
    n_tot = merged["n_test_cells_old"].sum()
    old_grand = merged["old_correct"].sum() / n_tot
    new_grand = merged["new_correct"].sum() / n_tot
    summary_rows.append({"config": "только каскад (без LLM)",
                          "old_grand_acc": old_grand, "new_grand_acc": new_grand,
                          "delta_pp": (new_grand - old_grand) * 100})

    # Per-LLM cascade + router
    hyb = pd.read_parquet(os.path.join(PROCESSED_DIR, "cascade_plus_llm4_hybrid.parquet"))
    for model in ["llama3b", "gptoss", "gpt4o", "sonnet45", "gemini25flash"]:
        sub = hyb[hyb.llm_model == model].copy()
        def estimate_new(row):
            new_row = headline[(headline.category == row["category"]) &
                                (headline.attr == row["attr"])]
            if new_row.empty:
                return row["acc_hybrid_with_router"]
            n_cov = int(new_row["n_covered"].iloc[0])
            n_abs = int(new_row["n_abstain"].iloc[0])
            acc_cov = float(new_row["acc_on_covered"].iloc[0]) if pd.notna(new_row["acc_on_covered"].iloc[0]) else 0.0
            llm_acc = float(row["llm_acc_on_attr"]) if pd.notna(row["llm_acc_on_attr"]) else 0.7
            n_test = int(new_row["n_test_cells"].iloc[0])
            return (n_cov * acc_cov + n_abs * llm_acc) / n_test if n_test else float("nan")
        sub["new_acc"] = sub.apply(estimate_new, axis=1)
        sub["old_correct"] = (sub["n_test"] * sub["acc_hybrid_with_router"]).round().astype(int)
        sub["new_correct"] = (sub["n_test"] * sub["new_acc"]).round().astype(int)
        n = sub["n_test"].sum()
        old_a = sub["old_correct"].sum() / n
        new_a = sub["new_correct"].sum() / n
        summary_rows.append({"config": f"каскад + {model}",
                              "old_grand_acc": old_a, "new_grand_acc": new_a,
                              "delta_pp": (new_a - old_a) * 100})

    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(os.path.join(PROCESSED_DIR, "grand_acc_summary_after_fix.parquet"), index=False)

    print()
    print("=" * 78)
    print("GRAND ACC SUMMARY — после всех исправлений §3.3.7.4 + threshold tuning")
    print("=" * 78)
    print(summary.to_string(index=False, float_format='%.4f'))
    print()
    print("Per-attr headline:")
    print(headline.to_string(index=False, float_format='%.4f'))


if __name__ == "__main__":
    main()
