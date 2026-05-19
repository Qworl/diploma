"""
5-fold cross-validation для main accuracy numbers.

Сейчас все cifry в notebook'ах — на одном split с RANDOM_STATE=42. CV даёт
mean ± std accuracy: если std > 3pp, single-split numbers cherry-picked.

Embeddings cached (.npy) → fold'и переиспользуют тот же encoder, переучивается
только XGBoost. ~30s × 5 folds × N attrs × 3 cats ≈ 15 min.

Использует **Stratified** K-fold (в отличие от global random split в train_classifiers).
Это сильнее: stratified предотвращает class missing per fold (см. notebooks/07 §2).

С --n-seeds 10: 5-fold CV × 10 random seeds = 50 samples per attribute.
mean±std становится статистически значимым (~150 min total).

Usage:
    python -m src.diagnostics.ml.cv_stability
    python -m src.diagnostics.ml.cv_stability --category beverages --n-splits 5
    python -m src.diagnostics.ml.cv_stability --n-seeds 10
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold

from src.common import PROCESSED_DIR, RANDOM_STATE, setup_logging, wilson_ci

logger = logging.getLogger(__name__)


_PASTA = ["grain_type", "pasta_shape", "is_organic", "is_filled",
          "is_gluten_free", "is_vegan", "nutri_score_grade", "protein_class"]
_CHOCO = ["chocolate_type", "cocoa_percentage", "contains_nuts",
          "chocolate_extra", "is_organic", "nutri_score_grade", "protein_class"]
_BEV   = ["beverage_type", "sugar_class", "is_organic", "is_carbonated",
          "nutri_score_grade", "nova_group", "protein_class", "is_vegan"]
_BABY  = ["milk_type", "minimal_age", "feeding_purpose", "format",
          "is_organic", "is_lactose_free", "is_gluten_free"]
_COSM  = ["product_type", "form_factor", "body_area",
          "has_sulfates", "has_silicones", "is_organic"]
_CHEESE = ["milk_source", "texture", "country_of_origin", "fat_class",
           "is_pdo", "is_organic", "is_ultra_processed"]
_CEREAL = ["cereal_type", "grain_type", "is_low_sugar", "is_high_fibre",
           "nova_class", "is_vegan", "is_whole_grain", "is_organic"]

CATEGORIES_ATTRS = {
    "pasta": _PASTA, "chocolate": _CHOCO, "beverages": _BEV,
    "pasta_stratified": _PASTA, "chocolate_stratified": _CHOCO, "beverages_stratified": _BEV,
    "cosmetics_stratified": _COSM,
    "cheeses_stratified": _CHEESE,
    "cereals_stratified": _CEREAL,
}


def fit_one_fold(X_tr, y_tr, X_te, y_te, *, multiclass: bool):
    """Train XGB на один fold + return metrics. Re-index labels per split
    чтобы XGB не падал на missing classes."""
    train_classes = sorted(np.unique(y_tr).tolist())
    if len(train_classes) < 2:
        return None
    remap = {c: i for i, c in enumerate(train_classes)}
    y_tr_r = np.array([remap[c] for c in y_tr])
    y_te_r = np.array([remap.get(c, -1) for c in y_te])  # unseen test classes → wrong

    n_classes = len(train_classes)
    if not multiclass or n_classes == 2:
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
    clf.fit(X_tr, y_tr_r)
    pred = clf.predict(X_te)
    acc = accuracy_score(y_te_r, pred)
    f1 = f1_score(y_te_r, pred, average="macro", zero_division=0,
                   labels=list(range(n_classes)))
    return {"accuracy": float(acc), "macro_f1": float(f1),
            "n_test": int(len(y_te_r))}


def cv_one_attr(X_full, y_full, classes, n_splits: int, multiclass: bool, seed: int):
    """Stratified K-fold CV для одного (X, y). Returns list of fold dicts."""
    fold_results = []
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(skf.split(X_full, y_full))
    except ValueError:
        # fallback to KFold if stratify impossible
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(kf.split(X_full))

    for fold_idx, (tr_i, te_i) in enumerate(splits):
        res = fit_one_fold(X_full[tr_i], y_full[tr_i], X_full[te_i], y_full[te_i],
                            multiclass=multiclass)
        if res is None:
            continue
        res["fold"] = fold_idx
        fold_results.append(res)
    return fold_results


def run(category: str, attrs: list[str], n_splits: int, seed: int, n_seeds: int = 1):
    ss_path = os.path.join(PROCESSED_DIR, f"{category}_silver_standard.parquet")
    emb_path = os.path.join(PROCESSED_DIR, f"{category}_embeddings.npy")
    if not (os.path.exists(ss_path) and os.path.exists(emb_path)):
        logger.warning("%s: missing silver/embeddings — skipping", category)
        return pd.DataFrame()
    ss = pd.read_parquet(ss_path)
    emb = np.load(emb_path)
    logger.info("=" * 70)
    logger.info("%s — %d products, %d-dim embeddings", category, len(ss), emb.shape[1])
    if n_seeds > 1:
        logger.info("Multi-seed mode: %d seeds × %d folds = %d samples per attr",
                    n_seeds, n_splits, n_seeds * n_splits)
    logger.info("=" * 70)

    rows = []
    for attr in attrs:
        if attr not in ss.columns:
            continue
        mask = ss[attr].notna()
        if mask.sum() < 50:
            logger.info("  %s: skipped (n=%d <50)", attr, mask.sum())
            continue

        y_raw = ss.loc[mask, attr].astype(str)
        # Drop rare classes <2 samples (CV stratify needs >=2)
        vc = y_raw.value_counts()
        keep = set(vc[vc >= n_splits].index)  # need >= n_splits for stratified split
        if not keep:
            continue
        keep_mask = y_raw.isin(keep)
        if keep_mask.sum() < 50:
            continue
        y_filt = y_raw[keep_mask.values].reset_index(drop=True)
        X_filt = emb[mask.values][keep_mask.values]
        classes = sorted(y_filt.unique())
        if len(classes) < 2:
            continue
        class_idx = {c: i for i, c in enumerate(classes)}
        y = np.array([class_idx[c] for c in y_filt])
        multiclass = len(classes) > 2

        # Multi-seed loop: iterate over seeds, collect all fold results
        all_folds = []
        for seed_idx in range(n_seeds):
            folds = cv_one_attr(X_filt, y, classes, n_splits=n_splits,
                                multiclass=multiclass, seed=seed + seed_idx)
            all_folds.extend(folds)

        if len(all_folds) == 0:
            continue
        accs = np.array([f["accuracy"] for f in all_folds])
        f1s = np.array([f["macro_f1"] for f in all_folds])
        # 95% CI via Wilson on total correct/total n across all folds and seeds
        total_correct = sum(f["accuracy"] * f["n_test"] for f in all_folds)
        total_n = sum(f["n_test"] for f in all_folds)
        ci_lo, ci_hi = wilson_ci(int(round(total_correct)), int(total_n))
        rows.append({
            "category": category.replace("_stratified", ""), "attr": attr,
            "n_per_fold_avg": float(total_n / len(all_folds)),
            "n_classes": len(classes),
            "n_folds": len(all_folds),
            "n_seeds": n_seeds,
            "acc_mean": float(accs.mean()),
            "acc_std": float(accs.std()),
            "acc_min": float(accs.min()),
            "acc_max": float(accs.max()),
            "acc_range_pp": float((accs.max() - accs.min()) * 100),
            "acc_ci_lo": float(ci_lo),
            "acc_ci_hi": float(ci_hi),
            "f1_mean": float(f1s.mean()),
            "f1_std": float(f1s.std()),
        })
        logger.info("  %-22s acc=%.3f±%.3f  range=%.1fpp  f1=%.3f±%.3f  n=%d×%d(s%d)",
                    attr, accs.mean(), accs.std(),
                    (accs.max() - accs.min()) * 100,
                    f1s.mean(), f1s.std(), total_n // len(all_folds), len(all_folds),
                    n_seeds)
    return pd.DataFrame(rows)


def main():
    setup_logging()
    p = argparse.ArgumentParser(
        description="Stratified K-fold CV stability check for XGBoost classifiers."
    )
    p.add_argument("--category", choices=list(CATEGORIES_ATTRS.keys()), default=None)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--n-seeds", type=int, default=1,
                   help="Number of random seeds × K folds (1=single seed; 10=robust statistics).")
    args = p.parse_args()

    cats = [args.category] if args.category else list(CATEGORIES_ATTRS.keys())
    all_rows = []
    for cat in cats:
        df = run(cat, CATEGORIES_ATTRS[cat], args.n_splits, args.seed, n_seeds=args.n_seeds)
        if not df.empty:
            all_rows.append(df)
    full = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if full.empty:
        logger.warning("No CV results — all silver/embeddings missing?")
        return

    # Distinct output filename when using multiple seeds
    if args.n_seeds > 1:
        suffix = f"_{args.n_seeds}seed"
    else:
        suffix = f"_{args.n_splits}fold"
    out = os.path.join(PROCESSED_DIR, f"cv_stability{suffix}.parquet")
    full.to_parquet(out, index=False)
    logger.info("\nSaved -> %s", out)

    # Final summary
    logger.info("\n" + "=" * 78)
    if args.n_seeds > 1:
        logger.info("CV STABILITY SUMMARY (%d-fold × %d seeds = %d samples/attr)",
                    args.n_splits, args.n_seeds, args.n_splits * args.n_seeds)
    else:
        logger.info("CV STABILITY SUMMARY (%d-fold stratified)", args.n_splits)
    logger.info("=" * 78)
    show = full[["category", "attr", "n_classes", "acc_mean", "acc_std",
                 "acc_range_pp", "f1_mean", "f1_std"]].copy()
    show["acc"] = show.apply(lambda r: f"{r.acc_mean*100:.1f}±{r.acc_std*100:.1f}%", axis=1)
    show["f1"] = show.apply(lambda r: f"{r.f1_mean*100:.1f}±{r.f1_std*100:.1f}", axis=1)
    show = show[["category", "attr", "n_classes", "acc", "acc_range_pp", "f1"]]
    show.columns = ["category", "attr", "K", "acc (mean±std)", "range (pp)", "macro_f1 (mean±std)"]
    logger.info(show.to_string(index=False))

    logger.info("\nINTERPRETATION:")
    logger.info("• std < 1pp: very stable, single-split numbers reliable")
    logger.info("• std 1-3pp: typical, cite mean as headline number")
    logger.info("• std > 3pp: high variance, single-seed numbers misleading")
    logger.info("• range > 5pp: worst-fold << best-fold, possible class-imbalance issue")


if __name__ == "__main__":
    main()
