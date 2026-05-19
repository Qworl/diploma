"""
Stratified vs random split — насколько меняется accuracy.

Текущий split: train_test_split(...) без stratify=y. Для skewed multiclass это
оставляет разный class balance в train vs test. Особенно бьёт rare classes —
они могут пропасть из test (или train) совсем.

Этот скрипт: для каждого attr делает 2 split'а — random vs stratified by attr,
тренирует XGBoost, сравнивает accuracy + macro_f1 + per-class recall.

Usage:
    python -m src.diagnostics.ml.stratified_split_ablation --category beverages
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from src.common import PROCESSED_DIR, RANDOM_STATE, TEST_SIZE, setup_logging, wilson_ci

logger = logging.getLogger(__name__)


_PASTA = ["grain_type", "pasta_shape", "is_organic", "is_gluten_free",
          "nutri_score_grade", "protein_class"]
_CHOCO = ["chocolate_type", "cocoa_percentage",
          "is_organic", "nutri_score_grade", "protein_class"]
_BEV   = ["beverage_type", "sugar_class", "is_organic", "nutri_score_grade",
          "nova_group", "protein_class"]
_CHEESES = ["milk_source", "texture", "country_of_origin", "fat_class",
            "is_pdo", "is_organic"]
_CEREALS = ["cereal_type", "grain_type", "is_low_sugar", "is_high_fibre",
            "is_whole_grain", "is_organic"]
_COSM    = ["product_type", "form_factor", "body_area",
            "has_sulfates", "has_silicones", "is_organic"]

CATEGORIES_ATTRS = {
    "pasta": _PASTA, "chocolate": _CHOCO, "beverages": _BEV,
    "pasta_stratified": _PASTA, "chocolate_stratified": _CHOCO, "beverages_stratified": _BEV,
    "cheeses_stratified": _CHEESES, "cereals_stratified": _CEREALS,
    "cosmetics_stratified": _COSM,
}


def train_eval(X_train, y_train, X_test, y_test, classes):
    """Re-index labels per split to handle case where random split misses a class."""
    train_classes = sorted(np.unique(y_train).tolist())
    if len(train_classes) < 2:
        return None, None, None, None  # cannot train
    # Re-map both train and test to contiguous train_classes; test labels not in
    # train_classes get -1 (will count as wrong predictions automatically).
    remap = {c: i for i, c in enumerate(train_classes)}
    y_tr_r = np.array([remap[c] for c in y_train])
    y_te_r = np.array([remap.get(c, -1) for c in y_test])

    n_classes = len(train_classes)
    if n_classes == 2:
        pos = (y_tr_r == 1).sum()
        neg = (y_tr_r == 0).sum()
        spw = max(neg / max(pos, 1), 0.5)
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
            scale_pos_weight=spw,
            objective="binary:logistic", eval_metric="logloss", verbosity=0,
        )
    else:
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", verbosity=0,
        )
    clf.fit(X_train, y_tr_r)
    pred = clf.predict(X_test)
    # Mismatched test classes (label=-1 in remap) will never equal pred → wrong → counts as miss
    acc = accuracy_score(y_te_r, pred)
    f1 = f1_score(y_te_r, pred, average="macro", zero_division=0,
                   labels=list(range(n_classes)))
    n_correct = int((pred == y_te_r).sum())
    ci_lo, ci_hi = wilson_ci(n_correct, len(y_te_r))
    return acc, f1, ci_lo, ci_hi


def run(category: str, attrs: list[str]):
    ss_path = os.path.join(PROCESSED_DIR, f"{category}_silver_standard.parquet")
    emb_path = os.path.join(PROCESSED_DIR, f"{category}_embeddings.npy")
    if not (os.path.exists(ss_path) and os.path.exists(emb_path)):
        logger.warning("%s: missing silver/embeddings — skipping", category)
        return pd.DataFrame()
    ss = pd.read_parquet(ss_path)
    emb = np.load(emb_path)
    logger.info("Loaded %s: %d rows", category, len(ss))

    rows = []
    for attr in attrs:
        if attr not in ss.columns:
            continue
        mask = ss[attr].notna()
        if mask.sum() < 50:
            continue
        df_a = ss[mask]
        emb_a = emb[mask.values]
        y_raw = df_a[attr].astype(str)
        # Filter rare classes (<2 samples) — иначе stratify падает,
        # и XGBoost class indexing breaks при non-contiguous labels.
        vc = y_raw.value_counts()
        keep_classes = set(vc[vc >= 2].index)
        keep_mask = y_raw.isin(keep_classes)
        if keep_mask.sum() < 50:
            continue
        emb_a = emb_a[keep_mask.values]
        y_raw = y_raw[keep_mask.values].reset_index(drop=True)
        classes = sorted(y_raw.unique())
        if len(classes) < 2:
            continue
        class_idx = {c: i for i, c in enumerate(classes)}
        y = np.array([class_idx[c] for c in y_raw])

        # Random split (current behavior)
        X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
            emb_a, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )
        random_train_classes = sorted(np.unique(y_tr_r).tolist())
        random_lost_classes = sorted(set(range(len(classes))) - set(random_train_classes))
        acc_r, f1_r, lo_r, hi_r = train_eval(X_tr_r, y_tr_r, X_te_r, y_te_r, classes)
        if acc_r is None:
            continue

        # Stratified split
        try:
            X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
                emb_a, y, test_size=TEST_SIZE,
                random_state=RANDOM_STATE, stratify=y,
            )
            acc_s, f1_s, lo_s, hi_s = train_eval(X_tr_s, y_tr_s, X_te_s, y_te_s, classes)
            stratified_ok = acc_s is not None
        except ValueError:
            acc_s = f1_s = lo_s = hi_s = None
            stratified_ok = False

        # Class distribution comparison
        train_dist_r = pd.Series(y_tr_r).value_counts(normalize=True).sort_index()
        test_dist_r = pd.Series(y_te_r).value_counts(normalize=True).sort_index()
        kl_r = abs(train_dist_r - test_dist_r).sum()  # L1 distance — proxy для divergence

        rows.append({
            "category": category.replace("_stratified", ""), "attr": attr, "n_total": int(mask.sum()),
            "n_classes": len(classes),
            "random_acc": float(acc_r), "random_f1": float(f1_r),
            "random_acc_ci": f"[{lo_r*100:.0f},{hi_r*100:.0f}]",
            "random_lost_classes": ",".join(classes[i] for i in random_lost_classes) or "—",
            "stratified_acc": float(acc_s) if stratified_ok else None,
            "stratified_f1": float(f1_s) if stratified_ok else None,
            "stratified_acc_ci": f"[{lo_s*100:.0f},{hi_s*100:.0f}]" if stratified_ok else "—",
            "delta_acc_pp": (acc_s - acc_r) * 100 if stratified_ok else None,
            "delta_f1_pp": (f1_s - f1_r) * 100 if stratified_ok else None,
            "train_test_class_dist_L1": float(kl_r),
        })

    df = pd.DataFrame(rows)
    return df


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--category", choices=list(CATEGORIES_ATTRS.keys()), default=None)
    args = p.parse_args()

    cats = [args.category] if args.category else list(CATEGORIES_ATTRS.keys())
    all_rows = []
    for cat in cats:
        df = run(cat, CATEGORIES_ATTRS[cat])
        if not df.empty:
            all_rows.append(df)
    if not all_rows:
        logger.warning("No data for any category — exiting")
        return
    full = pd.concat(all_rows, ignore_index=True)

    out = os.path.join(PROCESSED_DIR, "stratified_split_ablation.parquet")
    full.to_parquet(out, index=False)

    logger.info("\n" + "=" * 78)
    logger.info("RANDOM vs STRATIFIED SPLIT — Δ accuracy & F1")
    logger.info("=" * 78)
    show_cols = ["category", "attr", "n_total", "n_classes",
                 "random_acc", "stratified_acc", "delta_acc_pp",
                 "random_f1", "stratified_f1", "delta_f1_pp",
                 "train_test_class_dist_L1"]
    logger.info(full[show_cols].round(3).to_string(index=False))

    logger.info("\nINTERPRETATION:")
    logger.info("• delta_acc_pp > +1: stratified split существенно улучшает test accuracy")
    logger.info("• train_test_class_dist_L1 > 0.10: random split дал заметный class skew")
    logger.info("• Если delta стабильно положителен → train_classifiers стоит переписать на stratified")
    logger.info("\nSaved -> %s", out)


if __name__ == "__main__":
    main()
