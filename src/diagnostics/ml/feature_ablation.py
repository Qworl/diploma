"""
Feature ablation: насколько каждое поле partner-input двигает ML accuracy.

Сейчас embeddings = SentenceTransformer(product_name + brands + ingredients_text + quantity).
Если убрать brands — на сколько падает accuracy для is_organic? Если падает много →
classifier полагается на brand-prior (Carrefour BIO → organic), а не на семантику ингредиентов.
Это объясняет, почему high accuracy на is_organic не гарантирует quality на products
с unseen brands.

Усиливает Phase 11 thesis: для food brand-information в name+brand часто избыточен
(дублирует apply_off_labels logic), а ML ценен для product_name+ingredients семантики
там где OFF tags отсутствуют.

Usage:
    python -m src.diagnostics.ml.feature_ablation --category chocolate
    python -m src.diagnostics.ml.feature_ablation --category beverages --attr is_organic
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.common import (
    EMBEDDING_MODEL,
    PROCESSED_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    setup_logging,
    wilson_ci,
)

logger = logging.getLogger(__name__)

# Ablation conditions: какие поля включаем в embedding
ABLATIONS = {
    "all_fields":        ("product_name", "brands", "ingredients_text", "quantity"),
    "no_brand":          ("product_name", "ingredients_text", "quantity"),
    "no_ingredients":    ("product_name", "brands", "quantity"),
    "name_only":         ("product_name",),
    "brand_only":        ("brands",),
    "ingredients_only":  ("ingredients_text",),
}

CATEGORIES_ATTRS = {
    "pasta_stratified": [
        "grain_type", "pasta_shape", "is_filled", "is_organic",
        "is_gluten_free", "is_vegan", "nutri_score_grade", "protein_class",
    ],
    "chocolate_stratified": [
        "chocolate_type", "cocoa_percentage", "contains_nuts", "chocolate_extra",
        "is_organic", "nutri_score_grade", "protein_class",
    ],
    "beverages_stratified": [
        "beverage_type", "sugar_class", "is_organic", "is_carbonated",
        "is_vegan", "nutri_score_grade", "nova_group", "protein_class",
    ],
    "cheeses_stratified": [
        "milk_source", "texture", "country_of_origin", "fat_class",
        "is_pdo", "is_organic", "is_ultra_processed",
    ],
    "cereals_stratified": [
        "cereal_type", "grain_type", "is_low_sugar", "is_high_fibre",
        "is_whole_grain", "is_vegan", "is_organic", "nova_class",
    ],
    "cosmetics_stratified": [
        "product_type", "form_factor", "body_area", "has_sulfates",
        "has_silicones", "is_organic",
    ],
}


def build_text_subset(df: pd.DataFrame, fields: tuple) -> list[str]:
    out = []
    for _, row in df.iterrows():
        parts = []
        for f in fields:
            v = row.get(f)
            if pd.notna(v) and str(v).strip():
                parts.append(str(v))
        out.append(" ".join(parts) if parts else " ")  # avoid empty input
    return out


def encode(texts: list[str], model) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False, batch_size=64)


def train_eval_xgb(X_tr, y_tr, X_te, y_te, *, multiclass: bool):
    """Train XGB with reasonable defaults, return accuracy and macro_f1."""
    if multiclass:
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
            objective="multi:softprob", num_class=len(np.unique(y_tr)),
            eval_metric="mlogloss", verbosity=0,
        )
    else:
        pos = (y_tr == 1).sum()
        neg = (y_tr == 0).sum()
        spw = max(neg / max(pos, 1), 0.5)
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
            scale_pos_weight=spw,
            objective="binary:logistic", eval_metric="logloss", verbosity=0,
        )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro", zero_division=0)
    n_correct = int((pred == y_te).sum())
    ci_lo, ci_hi = wilson_ci(n_correct, len(y_te))
    return {"accuracy": float(acc), "macro_f1": float(f1),
            "n": int(len(y_te)), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}


def run_ablation(category: str, attrs: list[str]):
    from sentence_transformers import SentenceTransformer

    ss_path = os.path.join(PROCESSED_DIR, f"{category}_silver_standard.parquet")
    if not os.path.exists(ss_path):
        logger.warning("%s: missing silver — skipping", category)
        return pd.DataFrame()
    df = pd.read_parquet(ss_path)
    logger.info("Loaded %s: %d rows", category, len(df))

    # Same global split for all ablations — only embedding содержимое меняется
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    rows = []
    for ablation_name, fields in ABLATIONS.items():
        # Skip ablation if all required fields are sparse for this category
        missing_cols = [f for f in fields if f not in df.columns]
        if missing_cols:
            logger.info("  skipping %s: missing %s", ablation_name, missing_cols)
            continue
        logger.info("Encoding [%s] (%s)", ablation_name, ",".join(fields))
        train_texts = build_text_subset(train_df, fields)
        test_texts = build_text_subset(test_df, fields)
        X_tr = encode(train_texts, model)
        X_te = encode(test_texts, model)

        for attr in attrs:
            if attr not in df.columns:
                continue
            tr_mask = train_df[attr].notna()
            te_mask = test_df[attr].notna()
            if tr_mask.sum() < 30 or te_mask.sum() < 20:
                continue

            y_tr_raw = train_df.loc[tr_mask, attr].astype(str)
            y_te_raw = test_df.loc[te_mask, attr].astype(str)
            classes = sorted(set(y_tr_raw) | set(y_te_raw))
            class_idx = {c: i for i, c in enumerate(classes)}
            y_tr = np.array([class_idx[c] for c in y_tr_raw])
            y_te = np.array([class_idx[c] for c in y_te_raw])
            multiclass = len(classes) > 2

            res = train_eval_xgb(X_tr[tr_mask.values], y_tr,
                                  X_te[te_mask.values], y_te,
                                  multiclass=multiclass)
            res.update({
                "category": category.replace("_stratified", ""),
                "attr": attr,
                "ablation": ablation_name,
                "fields": ",".join(fields),
                "n_classes": len(classes),
            })
            rows.append(res)
            logger.info("  %-22s acc=%.3f [%4.1f, %4.1f]  f1=%.3f  n=%d",
                        attr, res["accuracy"], res["ci_lo"]*100, res["ci_hi"]*100,
                        res["macro_f1"], res["n"])

    return pd.DataFrame(rows)


def report_delta(df: pd.DataFrame):
    """Печатает Δ accuracy относительно all_fields baseline."""
    logger.info("\n" + "=" * 78)
    logger.info("FEATURE ABLATION — Δ accuracy относительно all_fields baseline")
    logger.info("=" * 78)
    logger.info("Если no_brand >> all_fields → ML полезен без brand bias")
    logger.info("Если no_brand << all_fields → ML опирается на brand prior (leak risk)")
    logger.info("Если name_only ~~ all_fields → достаточно одного product_name")
    logger.info("")

    pivot = df.pivot_table(index=["category", "attr"], columns="ablation",
                            values="accuracy", aggfunc="first")
    if "all_fields" not in pivot.columns:
        logger.warning("No 'all_fields' baseline — Δ comparison skipped")
        return
    base = pivot["all_fields"]
    delta = (pivot.subtract(base, axis=0)) * 100  # percentage points

    logger.info("Δ accuracy (pp) vs all_fields baseline:")
    cols = ["no_brand", "no_ingredients", "name_only", "brand_only", "ingredients_only"]
    cols = [c for c in cols if c in delta.columns]
    delta_show = delta[cols].round(1)
    logger.info(delta_show.to_string())

    logger.info("\nAbsolute accuracy (%):")
    abs_show = (pivot[["all_fields"] + cols] * 100).round(1)
    logger.info(abs_show.to_string())


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--category", choices=list(CATEGORIES_ATTRS.keys()), default=None,
                   help="Один cat (default: все 6)")
    p.add_argument("--attr", default=None, help="Один атрибут (default: список из CATEGORIES_ATTRS)")
    args = p.parse_args()

    cats = [args.category] if args.category else list(CATEGORIES_ATTRS.keys())
    all_results = []
    for cat in cats:
        attrs = [args.attr] if args.attr else CATEGORIES_ATTRS[cat]
        res = run_ablation(cat, attrs)
        if not res.empty:
            all_results.append(res)

    if not all_results:
        logger.warning("No data for any category — exiting")
        return
    full = pd.concat(all_results, ignore_index=True)
    out = os.path.join(PROCESSED_DIR, "feature_ablation.parquet")
    full.to_parquet(out, index=False)
    logger.info("\nSaved -> %s (%d rows)", out, len(full))

    report_delta(full)


if __name__ == "__main__":
    main()
