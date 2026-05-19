"""P2-EXP5b: LightGBM Layer 1.5 — focused honest 80/20 evaluation.

Variant M: lightgbm_ml @tau=0.85
  - LightGBM on TF-IDF n-grams (1-2) over product_name+ingredients_text+brands
  - Fallback to hybrid_v2 ML (fresh-trained on 80% silver+gold weighted 5x)

Same 80/20 split (seed=42, per-cat gold codes) as EXP3 / layer1_5_honest_comparison.py.

Output: datasets/processed/lightgbm_layer1_5.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.common import PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)

CATEGORIES = ["pasta", "chocolate", "cheeses"]
SEED = 42
TEST_SIZE = 0.2
MIN_GOLD = 20      # skip attr if fewer non-null gold cells
LGBM_TAU = 0.85   # top-proba threshold; below => fallback to hybrid_v2 ML

GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet"
OUT_PATH = Path(PROCESSED_DIR) / "lightgbm_layer1_5.parquet"


# ---------------------------------------------------------------------------
# Text builder
# ---------------------------------------------------------------------------

def _build_text(row: pd.Series) -> str:
    parts = []
    for col in ["product_name", "ingredients_text", "brands"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Fresh hybrid XGBoost trainer (same as layer1_5_honest_comparison.py)
# ---------------------------------------------------------------------------

def _train_fresh_hybrid(
    X_silver: np.ndarray,
    y_silver: np.ndarray,
    X_gold: np.ndarray,
    y_gold: np.ndarray,
    gold_weight: float = 5.0,
) -> tuple[Optional[xgb.XGBClassifier], Optional[LabelEncoder]]:
    """Train XGB on (silver + gold-weighted). Returns (clf, le) or (None, None)."""
    X_combined = np.vstack([X_silver, X_gold])
    y_combined = np.concatenate([y_silver, y_gold])
    w_combined = np.concatenate([
        np.ones(len(y_silver)),
        gold_weight * np.ones(len(y_gold)),
    ])

    all_classes = sorted(set(y_combined.tolist()))
    if len(all_classes) < 2:
        return None, None

    le = LabelEncoder()
    le.fit(all_classes)
    y_enc = le.transform(y_combined)

    n_classes = len(all_classes)
    common_kwargs = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        tree_method="hist", verbosity=0,
    )
    if n_classes == 2:
        pos = int((y_enc == 1).sum())
        neg = int((y_enc == 0).sum())
        spw = max(neg / max(pos, 1), 0.5)
        clf = xgb.XGBClassifier(scale_pos_weight=spw, **common_kwargs)
    else:
        clf = xgb.XGBClassifier(
            objective="multi:softmax", num_class=n_classes, **common_kwargs
        )

    clf.fit(X_combined, y_enc, sample_weight=w_combined)
    return clf, le


# ---------------------------------------------------------------------------
# LightGBM trainer on TF-IDF features
# ---------------------------------------------------------------------------

def _train_lgbm(
    train_texts: list[str],
    y_train: list[str],
) -> tuple[Optional[lgb.LGBMClassifier], Optional[TfidfVectorizer], Optional[LabelEncoder]]:
    """Train LightGBM on TF-IDF features. Returns (clf, vectorizer, le) or (None, None, None)."""
    all_classes = sorted(set(y_train))
    if len(all_classes) < 2:
        return None, None, None

    le = LabelEncoder()
    le.fit(all_classes)
    y_enc = le.transform(y_train)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10000)
    X_tfidf = vec.fit_transform(train_texts)

    n_classes = len(all_classes)
    if n_classes == 2:
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            objective="binary",
            verbose=-1,
        )
    else:
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            objective="multiclass",
            num_class=n_classes,
            verbose=-1,
        )

    clf.fit(X_tfidf, y_enc)
    return clf, vec, le


def _lgbm_predict(
    clf: lgb.LGBMClassifier,
    vec: TfidfVectorizer,
    le: LabelEncoder,
    texts: list[str],
    tau: float,
) -> list[tuple[Optional[str], float]]:
    """Return (label_or_None, top_proba) for each text. None => below tau."""
    X = vec.transform(texts)
    probas = clf.predict_proba(X)
    results = []
    for row in probas:
        top_idx = int(np.argmax(row))
        top_p = float(row[top_idx])
        if top_p >= tau:
            label = le.inverse_transform([top_idx])[0]
            results.append((str(label), top_p))
        else:
            results.append((None, top_p))
    return results


# ---------------------------------------------------------------------------
# Per-(cat, attr) computation
# ---------------------------------------------------------------------------

def run_one_attr(
    cat: str,
    attr: str,
    gold: pd.DataFrame,
    silver: pd.DataFrame,
    emb_all: np.ndarray,
    code_to_idx: dict[str, int],
    train_codes_set: set[str],
    test_codes_set: set[str],
    test_products: pd.DataFrame,
) -> list[dict]:
    """Run all variants for one (cat, attr). Returns list of result dicts."""
    # Filter gold for this attr
    cat_gold = gold[
        (gold["category"] == cat) & (gold["attr"] == attr) & ~gold["gold_is_null"]
    ].copy()
    cat_gold["code"] = cat_gold["code"].astype(str)
    cat_gold = cat_gold[cat_gold["code"].isin(code_to_idx)]

    if len(cat_gold) < MIN_GOLD:
        logger.info("[%s/%s] only %d non-null gold cells, skipping", cat, attr, len(cat_gold))
        return []

    train_gold = cat_gold[cat_gold["code"].isin(train_codes_set)]
    test_gold = cat_gold[cat_gold["code"].isin(test_codes_set)]

    if len(train_gold) < 10 or len(test_gold) < 5:
        logger.info("[%s/%s] insufficient train/test split, skipping", cat, attr)
        return []

    # Build embedding arrays
    train_idx = np.array([code_to_idx[c] for c in train_gold["code"]])
    test_idx = np.array([code_to_idx[c] for c in test_gold["code"]])

    X_gold_train = emb_all[train_idx]
    y_gold_train = train_gold["gold_value"].astype(str).values
    X_test_emb = emb_all[test_idx]
    y_test = test_gold["gold_value"].astype(str).values
    test_codes_list = test_gold["code"].tolist()

    # Build silver training data (exclude test codes)
    X_silver: np.ndarray = np.empty((0, emb_all.shape[1]))
    y_silver: np.ndarray = np.array([], dtype=str)

    if attr in silver.columns:
        silver_attr = silver[silver[attr].notna()].copy()
        silver_attr["code"] = silver_attr["code"].astype(str)
        silver_attr = silver_attr[~silver_attr["code"].isin(test_codes_set)]
        silver_attr = silver_attr[~silver_attr["code"].isin(train_codes_set)]
        silver_attr = silver_attr[silver_attr["code"].isin(code_to_idx)]

        silver_idx = np.array([code_to_idx[c] for c in silver_attr["code"]])
        if len(silver_idx) > 0:
            X_silver = emb_all[silver_idx]
            y_silver = silver_attr[attr].astype(str).values

    # Train fresh hybrid ML (XGBoost fallback)
    if len(X_silver) > 0:
        clf_xgb, le_xgb = _train_fresh_hybrid(X_silver, y_silver, X_gold_train, y_gold_train)
    else:
        # Gold-only XGBoost
        all_classes = sorted(set(y_gold_train.tolist()))
        if len(all_classes) < 2:
            logger.info("[%s/%s] degenerate classes in gold, skipping", cat, attr)
            return []
        le_xgb = LabelEncoder()
        le_xgb.fit(all_classes)
        y_enc = le_xgb.transform(y_gold_train)
        n_classes = len(all_classes)
        kwargs = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
                      tree_method="hist", verbosity=0)
        if n_classes == 2:
            pos = int((y_enc == 1).sum())
            neg = int((y_enc == 0).sum())
            kwargs["scale_pos_weight"] = max(neg / max(pos, 1), 0.5)
            clf_xgb = xgb.XGBClassifier(**kwargs)
        else:
            clf_xgb = xgb.XGBClassifier(
                objective="multi:softmax", num_class=n_classes, **kwargs
            )
        clf_xgb.fit(X_gold_train, y_enc)

    if clf_xgb is None or le_xgb is None:
        logger.info("[%s/%s] could not train hybrid ML, skipping", cat, attr)
        return []

    # Compute hybrid ML predictions for all test
    enc_preds = clf_xgb.predict(X_test_emb)
    ml_labels_all = le_xgb.inverse_transform(enc_preds).tolist()

    # Build text features: train texts from gold train joined with silver product text
    train_gold_with_text = train_gold.merge(
        silver[["code", "product_name", "ingredients_text", "brands"]],
        on="code", how="left",
    )
    train_texts = [_build_text(row) for _, row in train_gold_with_text.iterrows()]
    y_train_texts = train_gold_with_text["gold_value"].astype(str).tolist()

    # Test texts from test_products
    test_products_idx = test_products.set_index("code")
    test_texts_list = []
    for c in test_codes_list:
        if c in test_products_idx.index:
            test_texts_list.append(_build_text(test_products_idx.loc[c]))
        else:
            test_texts_list.append("")

    # Train LightGBM on TF-IDF features
    lgbm_clf, lgbm_vec, lgbm_le = _train_lgbm(train_texts, y_train_texts)

    # --- Variant B: ml_only (hybrid XGBoost) ---
    preds_b = ml_labels_all[:]

    # --- Variant M: lightgbm_ml (LightGBM@tau + fallback to hybrid ML) ---
    preds_m = []
    lgbm_used = 0
    if lgbm_clf is not None and lgbm_vec is not None and lgbm_le is not None:
        lgbm_results = _lgbm_predict(lgbm_clf, lgbm_vec, lgbm_le, test_texts_list, tau=LGBM_TAU)
        for i, (label, conf) in enumerate(lgbm_results):
            if label is not None:
                preds_m.append(str(label))
                lgbm_used += 1
            else:
                preds_m.append(ml_labels_all[i])
    else:
        preds_m = ml_labels_all[:]

    # Compute accuracies
    n = len(y_test)
    acc_b = sum(1 for p, g in zip(preds_b, y_test) if p == g) / n if n > 0 else float("nan")
    acc_m = sum(1 for p, g in zip(preds_m, y_test) if p == g) / n if n > 0 else float("nan")
    lgbm_coverage = lgbm_used / n if n > 0 else 0.0

    logger.info(
        "[%s/%s] n_test=%d | B_ml_only=%.3f | M_lgbm_ml=%.3f | lgbm_cov=%.2f",
        cat, attr, n, acc_b, acc_m, lgbm_coverage,
    )

    rows = [
        {
            "category": cat,
            "attr": attr,
            "variant": "B_ml_only",
            "accuracy": acc_b,
            "n_test": n,
            "n_train_gold": len(train_gold),
            "n_silver": len(y_silver),
            "lgbm_coverage": 0.0,
        },
        {
            "category": cat,
            "attr": attr,
            "variant": "M_lightgbm_ml",
            "accuracy": acc_m,
            "n_test": n,
            "n_train_gold": len(train_gold),
            "n_silver": len(y_silver),
            "lgbm_coverage": lgbm_coverage,
        },
    ]
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)

    logger.info(
        "Loaded gold: %d rows, %d unique codes",
        len(gold), gold["code"].nunique(),
    )

    all_rows: list[dict] = []

    for cat in CATEGORIES:
        logger.info("=== Category: %s ===", cat)

        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        )
        silver["code"] = silver["code"].astype(str)

        emb_all = np.load(Path(PROCESSED_DIR) / f"{cat}_stratified_embeddings.npy")
        code_to_idx: dict[str, int] = {c: i for i, c in enumerate(silver["code"].tolist())}

        cat_gold = gold[gold["category"] == cat].copy()
        unique_codes = sorted(cat_gold["code"].unique().tolist())

        # Same 80/20 split as eval_v2_expanded.py (seed=42)
        train_codes, test_codes = train_test_split(
            unique_codes, test_size=TEST_SIZE, random_state=SEED
        )
        train_codes_set = set(train_codes)
        test_codes_set = set(test_codes)
        logger.info(
            "  Split: %d train codes, %d test codes",
            len(train_codes), len(test_codes),
        )

        # Build test products subset from silver (for text features)
        test_products = silver[silver["code"].isin(test_codes_set)].copy()

        attrs = sorted(cat_gold["attr"].unique().tolist())
        logger.info("  Attrs: %s", attrs)

        for attr in attrs:
            rows = run_one_attr(
                cat=cat,
                attr=attr,
                gold=cat_gold,
                silver=silver,
                emb_all=emb_all,
                code_to_idx=code_to_idx,
                train_codes_set=train_codes_set,
                test_codes_set=test_codes_set,
                test_products=test_products,
            )
            all_rows.extend(rows)

    result = pd.DataFrame(all_rows)
    result.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(result), OUT_PATH)

    _print_summary(result)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(result: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("P2-EXP5b: LightGBM Layer 1.5 — Grand Means")
    print("=" * 70)

    grand = result.groupby("variant")["accuracy"].mean()
    print(f"  B_ml_only      (baseline): {grand.get('B_ml_only', float('nan'))*100:.2f}%")
    print(f"  M_lightgbm_ml (@tau=0.85): {grand.get('M_lightgbm_ml', float('nan'))*100:.2f}%")
    print(f"  Reference: A_regex_ml ~83.58% (from EXP3)")

    b = grand.get("B_ml_only", float("nan"))
    m = grand.get("M_lightgbm_ml", float("nan"))
    delta_m_vs_b = (m - b) * 100
    regex_ref = 0.8358
    delta_m_vs_regex = (m - regex_ref) * 100
    print(f"\n  M vs B (pp):     {delta_m_vs_b:+.2f}")
    print(f"  M vs regex (pp): {delta_m_vs_regex:+.2f}")

    print("\n" + "=" * 70)
    print("Per-(category, attr) accuracy")
    print("=" * 70)
    pivot = result.pivot_table(
        index=["category", "attr"], columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [c for c in ["B_ml_only", "M_lightgbm_ml"] if c in pivot.columns]
    pivot = pivot[cols].copy()
    if "M_lightgbm_ml" in pivot.columns and "B_ml_only" in pivot.columns:
        pivot["delta_vs_B"] = (pivot["M_lightgbm_ml"] - pivot["B_ml_only"]) * 100
        pivot["vs_regex_ref"] = (pivot["M_lightgbm_ml"] - 0.8358) * 100
    pivot["winner"] = pivot[cols].idxmax(axis=1)
    print(pivot.round(4).to_string())

    print("\n" + "=" * 70)
    print("Winner counts")
    print("=" * 70)
    wc = pivot["winner"].value_counts()
    for v in cols:
        print(f"  {v:22s}: {wc.get(v, 0)} attrs")

    print("\n" + "=" * 70)
    print("Average LightGBM coverage (tau=0.85)")
    print("=" * 70)
    lgbm_rows = result[result["variant"] == "M_lightgbm_ml"]
    print(f"  Mean coverage: {lgbm_rows['lgbm_coverage'].mean()*100:.1f}%")
    pivot_cov = result[result["variant"] == "M_lightgbm_ml"].set_index(["category", "attr"])["lgbm_coverage"]
    for (cat, attr), cov in pivot_cov.items():
        print(f"    [{cat}/{attr}] {cov*100:.1f}%")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if m > regex_ref:
        print(f"  LightGBM ({m*100:.2f}%) BEATS regex baseline (83.58%) by {delta_m_vs_regex:+.2f} pp")
    else:
        print(f"  LightGBM ({m*100:.2f}%) does NOT beat regex baseline (83.58%), gap: {delta_m_vs_regex:+.2f} pp")


if __name__ == "__main__":
    main()
