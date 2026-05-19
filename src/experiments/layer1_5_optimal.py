"""P2-EXP4: Layer 1.5 optimal mix — per-attr best method + cascade variants.

Builds on P2-EXP3 (layer1_5_honest_comparison.py) which tested 6 variants (A-F)
on a fresh 80/20 split. Here we test 4 new variants:

  G  per_attr_best         — pick winning method per (cat, attr) using 5-fold CV on train
  H  regex_then_dt_then_ml — sequential: regex → DT@0.80 → ML
  I  dt_then_regex_then_ml — sequential: DT@0.80 → regex → ML
  J  all_vote_then_ml      — soft-vote: regex/DT/NB → if 2+ agree → use; else → ML

Output:
  datasets/processed/layer1_5_optimal.parquet
  models/per_attr_best_mapping.json
  docs/layer1_5_optimal_recommendation.md
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from src.common import MODELS_DIR, PROCESSED_DIR, setup_logging
from src.experiments.layer1_5_methods import (
    predict_centroid,
    predict_dt,
    predict_logreg,
    train_centroid,
    train_dt,
    train_logreg,
)
from src.experiments.nb_layer import predict_nb, train_nb_for_attr
from src.pipeline.regex.extractor import RegexExtractor

logger = logging.getLogger(__name__)

CATEGORIES = ["pasta", "chocolate", "cheeses"]
SEED = 42
TEST_SIZE = 0.2
MIN_GOLD = 20

# Layer-1.5 thresholds (same as EXP3)
NB_TAU = 0.95
DT_TAU = 0.80
LOGREG_TAU = 0.95
CENTROID_MARGIN = 0.10

# Cross-validation folds for per_attr_best selection
CV_FOLDS = 5

GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet"
OUT_PATH = Path(PROCESSED_DIR) / "layer1_5_optimal.parquet"
MAPPING_PATH = Path(MODELS_DIR) / "per_attr_best_mapping.json"
DOC_PATH = Path("docs") / "layer1_5_optimal_recommendation.md"

# EXP3 baselines from layer1_5_honest_comparison.parquet
EXP3_PATH = Path(PROCESSED_DIR) / "layer1_5_honest_comparison.parquet"

# Candidate method names for per_attr_best selection
CANDIDATE_METHODS = ["regex", "nb", "dt", "centroid", "ml"]


# ---------------------------------------------------------------------------
# Text builder
# ---------------------------------------------------------------------------

def _build_text(row: pd.Series) -> str:
    parts = []
    for col in ["product_name", "ingredients_text", "brands", "quantity"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Fresh hybrid XGBoost trainer (same as EXP3)
# ---------------------------------------------------------------------------

def _train_fresh_hybrid(
    X_silver: np.ndarray,
    y_silver: np.ndarray,
    X_gold: np.ndarray,
    y_gold: np.ndarray,
    gold_weight: float = 5.0,
) -> tuple[Optional[xgb.XGBClassifier], Optional[LabelEncoder]]:
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


def _train_gold_only_hybrid(
    X_gold: np.ndarray,
    y_gold: np.ndarray,
) -> tuple[Optional[xgb.XGBClassifier], Optional[LabelEncoder]]:
    """Train XGB on gold data only."""
    all_classes = sorted(set(y_gold.tolist()))
    if len(all_classes) < 2:
        return None, None

    le = LabelEncoder()
    le.fit(all_classes)
    y_enc = le.transform(y_gold)

    n_classes = len(all_classes)
    kwargs = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        tree_method="hist", verbosity=0,
    )
    if n_classes == 2:
        pos = int((y_enc == 1).sum())
        neg = int((y_enc == 0).sum())
        kwargs["scale_pos_weight"] = max(neg / max(pos, 1), 0.5)
        clf = xgb.XGBClassifier(**kwargs)
    else:
        clf = xgb.XGBClassifier(objective="multi:softmax", num_class=n_classes, **kwargs)

    clf.fit(X_gold, y_enc)
    return clf, le


# ---------------------------------------------------------------------------
# Regex helper
# ---------------------------------------------------------------------------

def _build_regex_preds(
    test_products: pd.DataFrame,
    domain: str,
) -> dict[str, dict[str, str]]:
    extractor = RegexExtractor()
    result: dict[str, dict[str, str]] = {}
    for _, row in test_products.iterrows():
        code = str(row["code"])
        extracted = extractor.extract_all(
            product_name=str(row.get("product_name") or ""),
            description=str(row.get("ingredients_text") or ""),
            quantity=str(row.get("quantity") or ""),
            category=domain,
        )
        attr_vals: dict[str, str] = {}
        for attr, res in extracted.items():
            if res.confidence > 0.0 and res.value is not None:
                attr_vals[attr] = str(res.value)
        result[code] = attr_vals
    return result


# ---------------------------------------------------------------------------
# CV-based method selection on training data
# ---------------------------------------------------------------------------

def _select_best_method_via_cv(
    cat: str,
    attr: str,
    gold_train: pd.DataFrame,
    silver: pd.DataFrame,
    emb_all: np.ndarray,
    code_to_idx: dict[str, int],
    regex_preds_train: dict[str, dict[str, str]],
    silver_lookup: pd.DataFrame,
    X_silver: np.ndarray,
    y_silver: np.ndarray,
    n_folds: int = CV_FOLDS,
) -> str:
    """Select best method using n-fold CV on training data.

    Returns one of: 'regex', 'nb', 'dt', 'centroid', 'ml'.
    Falls back to 'ml' if insufficient data for CV.
    """
    codes = gold_train["code"].values
    labels = gold_train["gold_value"].astype(str).values

    if len(codes) < n_folds * 2:
        return "ml"

    # Collect per-method fold accuracies
    method_scores: dict[str, list[float]] = {m: [] for m in CANDIDATE_METHODS}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    indices = np.arange(len(codes))

    for fold_train_idx, fold_val_idx in kf.split(indices):
        fold_train_codes = set(codes[fold_train_idx].tolist())
        fold_val_codes = codes[fold_val_idx].tolist()
        fold_val_labels = labels[fold_val_idx].tolist()

        if len(fold_train_codes) < 5 or len(fold_val_codes) < 2:
            continue

        fold_train_gold = gold_train[gold_train["code"].isin(fold_train_codes)]
        fold_val_gold = gold_train[gold_train["code"].isin(set(fold_val_codes))]

        # Get embeddings
        fold_train_idx_emb = np.array([
            code_to_idx[c] for c in fold_train_gold["code"] if c in code_to_idx
        ])
        fold_val_idx_emb = np.array([
            code_to_idx[c] for c in fold_val_codes if c in code_to_idx
        ])

        if len(fold_train_idx_emb) < 2 or len(fold_val_idx_emb) < 1:
            continue

        X_fold_train = emb_all[fold_train_idx_emb]
        y_fold_train = fold_train_gold[fold_train_gold["code"].isin(code_to_idx)]["gold_value"].astype(str).values
        X_fold_val = emb_all[fold_val_idx_emb]

        val_codes_in_idx = [c for c in fold_val_codes if c in code_to_idx]
        y_fold_val = [labels[i] for i, c in enumerate(fold_val_codes) if c in code_to_idx]

        if len(y_fold_train) < 2 or len(set(y_fold_train)) < 2:
            continue

        # Build texts for fold_train
        fold_train_merged = fold_train_gold.merge(
            silver_lookup, on="code", how="left"
        )
        fold_train_texts = [_build_text(r) for _, r in fold_train_merged.iterrows()]
        fold_train_labels_text = fold_train_merged["gold_value"].astype(str).tolist()

        # Build texts for fold_val
        fold_val_texts = []
        for c in val_codes_in_idx:
            if c in silver_lookup.index:
                fold_val_texts.append(_build_text(silver_lookup.loc[c]))
            else:
                fold_val_texts.append("")

        # ---- ML ----
        try:
            if len(X_silver) > 0:
                # Exclude fold_val codes from silver to prevent leakage
                val_set = set(val_codes_in_idx)
                silver_mask = np.ones(len(X_silver), dtype=bool)
                # (simplified: silver is already test-excluded, fold val is small)
                clf_ml, le_ml = _train_fresh_hybrid(X_silver, y_silver, X_fold_train, y_fold_train)
            else:
                clf_ml, le_ml = _train_gold_only_hybrid(X_fold_train, y_fold_train)

            if clf_ml is not None:
                enc_preds = clf_ml.predict(X_fold_val)
                ml_preds = le_ml.inverse_transform(enc_preds).tolist()
                acc_ml = accuracy_score(y_fold_val, ml_preds)
            else:
                acc_ml = 0.0
        except Exception as e:
            logger.debug("ML fold error: %s", e)
            acc_ml = 0.0
        method_scores["ml"].append(acc_ml)

        # ---- DT ----
        try:
            dt_vec, dt_clf = train_dt(fold_train_texts, fold_train_labels_text)
            if dt_vec is not None:
                dt_results = predict_dt(dt_vec, dt_clf, fold_val_texts, tau=DT_TAU)
                dt_preds = [
                    label if label is not None else (ml_preds[i] if acc_ml > 0 else "unknown")
                    for i, (label, _) in enumerate(dt_results)
                ]
                acc_dt = accuracy_score(y_fold_val, dt_preds)
            else:
                acc_dt = acc_ml
        except Exception as e:
            logger.debug("DT fold error: %s", e)
            acc_dt = acc_ml
        method_scores["dt"].append(acc_dt)

        # ---- NB ----
        try:
            # Build a minimal gold df for train_nb_for_attr
            fold_gold_for_nb = fold_train_gold.copy()
            fold_gold_for_nb["category"] = cat
            fold_gold_for_nb["attr"] = attr
            nb_clf, nb_vec = train_nb_for_attr(
                cat, attr,
                gold=fold_gold_for_nb,
                silver=silver,
                train_codes=list(fold_train_codes),
            )
            if nb_clf is not None:
                nb_results = predict_nb(nb_clf, nb_vec, fold_val_texts, tau=NB_TAU)
                nb_preds = [
                    label if label is not None else (ml_preds[i] if acc_ml > 0 else "unknown")
                    for i, (label, _) in enumerate(nb_results)
                ]
                acc_nb = accuracy_score(y_fold_val, nb_preds)
            else:
                acc_nb = acc_ml
        except Exception as e:
            logger.debug("NB fold error: %s", e)
            acc_nb = acc_ml
        method_scores["nb"].append(acc_nb)

        # ---- Centroid ----
        try:
            centroid_dict = train_centroid(X_fold_train, y_fold_train.tolist())
            if centroid_dict is not None:
                c_results = predict_centroid(centroid_dict, X_fold_val, margin_tau=CENTROID_MARGIN)
                c_preds = [
                    label if label is not None else (ml_preds[i] if acc_ml > 0 else "unknown")
                    for i, (label, _) in enumerate(c_results)
                ]
                acc_centroid = accuracy_score(y_fold_val, c_preds)
            else:
                acc_centroid = acc_ml
        except Exception as e:
            logger.debug("Centroid fold error: %s", e)
            acc_centroid = acc_ml
        method_scores["centroid"].append(acc_centroid)

        # ---- Regex ----
        try:
            regex_val_preds = []
            for i, c in enumerate(val_codes_in_idx):
                rval = regex_preds_train.get(c, {}).get(attr)
                if rval is not None:
                    regex_val_preds.append(str(rval))
                else:
                    regex_val_preds.append(ml_preds[i] if acc_ml > 0 else "unknown")
            acc_regex = accuracy_score(y_fold_val, regex_val_preds)
        except Exception as e:
            logger.debug("Regex fold error: %s", e)
            acc_regex = acc_ml
        method_scores["regex"].append(acc_regex)

    # Average folds
    avg_scores = {}
    for m, scores in method_scores.items():
        avg_scores[m] = float(np.mean(scores)) if scores else 0.0

    if not avg_scores:
        return "ml"

    best = max(avg_scores, key=lambda m: avg_scores[m])
    logger.debug(
        "[%s/%s] CV method scores: %s → best=%s",
        cat, attr,
        {m: f"{v:.3f}" for m, v in avg_scores.items()},
        best,
    )
    return best


# ---------------------------------------------------------------------------
# Main per-(cat, attr) runner
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
    regex_preds_test: dict[str, dict[str, str]],
    regex_preds_train: dict[str, dict[str, str]],
    test_products: pd.DataFrame,
    silver_lookup: pd.DataFrame,
) -> tuple[list[dict], Optional[str]]:
    """Run all 4 new variants (G/H/I/J) for one (cat, attr) pair.

    Returns (list of result rows, best_method_from_cv).
    """
    # ---- Filter gold ----
    cat_gold = gold[(gold["category"] == cat) & (gold["attr"] == attr) & ~gold["gold_is_null"]].copy()
    cat_gold["code"] = cat_gold["code"].astype(str)
    cat_gold = cat_gold[cat_gold["code"].isin(code_to_idx)]

    if len(cat_gold) < MIN_GOLD:
        logger.info("[%s/%s] only %d non-null gold cells, skipping", cat, attr, len(cat_gold))
        return [], None

    train_gold = cat_gold[cat_gold["code"].isin(train_codes_set)]
    test_gold = cat_gold[cat_gold["code"].isin(test_codes_set)]

    if len(train_gold) < 10 or len(test_gold) < 5:
        logger.info("[%s/%s] insufficient train/test split, skipping", cat, attr)
        return [], None

    # ---- Embeddings ----
    train_idx = np.array([code_to_idx[c] for c in train_gold["code"]])
    test_idx = np.array([code_to_idx[c] for c in test_gold["code"]])

    X_gold_train = emb_all[train_idx]
    y_gold_train = train_gold["gold_value"].astype(str).values
    X_test_emb = emb_all[test_idx]
    y_test = test_gold["gold_value"].astype(str).values
    test_codes_list = test_gold["code"].tolist()

    # ---- Silver data (exclude test codes + train gold codes) ----
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

    # ---- Train fresh hybrid ML ----
    if len(X_silver) > 0:
        clf, le = _train_fresh_hybrid(X_silver, y_silver, X_gold_train, y_gold_train)
    else:
        clf, le = _train_gold_only_hybrid(X_gold_train, y_gold_train)

    if clf is None or le is None:
        logger.info("[%s/%s] could not train hybrid ML, skipping", cat, attr)
        return [], None

    # ---- Build text features ----
    train_gold_with_text = train_gold.merge(
        silver[["code", "product_name", "ingredients_text", "brands", "quantity"]],
        on="code", how="left",
    )
    train_texts = [_build_text(row) for _, row in train_gold_with_text.iterrows()]
    y_train_texts = train_gold_with_text["gold_value"].astype(str).tolist()

    test_products_idx = test_products.set_index("code")
    test_texts_list = []
    for c in test_codes_list:
        if c in test_products_idx.index:
            test_texts_list.append(_build_text(test_products_idx.loc[c]))
        else:
            test_texts_list.append("")

    # ---- Train Layer-1.5 methods ----
    nb_clf, nb_vec = train_nb_for_attr(
        cat, attr,
        gold=gold[(gold["category"] == cat) & (gold["attr"] == attr)],
        silver=silver,
        train_codes=list(train_codes_set),
    )
    dt_vec, dt_clf = train_dt(train_texts, y_train_texts)
    centroid_dict = train_centroid(X_gold_train, y_gold_train.tolist())
    lr_vec, lr_clf = train_logreg(train_texts, y_train_texts)

    # ---- ML predictions (base fallback) ----
    enc_preds = clf.predict(X_test_emb)
    ml_labels_all = le.inverse_transform(enc_preds).tolist()

    # ---- Pre-compute each method's raw predictions ----
    # Regex
    regex_preds_list = [regex_preds_test.get(c, {}).get(attr) for c in test_codes_list]

    # DT
    if dt_vec is not None and dt_clf is not None:
        dt_results = predict_dt(dt_vec, dt_clf, test_texts_list, tau=DT_TAU)
        dt_preds_raw = [label for label, _ in dt_results]  # None or str
    else:
        dt_preds_raw = [None] * len(test_codes_list)

    # NB
    if nb_clf is not None and nb_vec is not None:
        nb_results = predict_nb(nb_clf, nb_vec, test_texts_list, tau=NB_TAU)
        nb_preds_raw = [label for label, _ in nb_results]
    else:
        nb_preds_raw = [None] * len(test_codes_list)

    # Centroid
    if centroid_dict is not None:
        c_results = predict_centroid(centroid_dict, X_test_emb, margin_tau=CENTROID_MARGIN)
        centroid_preds_raw = [label for label, _ in c_results]
    else:
        centroid_preds_raw = [None] * len(test_codes_list)

    # ---- CV-based best method selection (on train data only) ----
    logger.info("[%s/%s] Running CV for per_attr_best selection...", cat, attr)
    best_method = _select_best_method_via_cv(
        cat=cat,
        attr=attr,
        gold_train=train_gold,
        silver=silver,
        emb_all=emb_all,
        code_to_idx=code_to_idx,
        regex_preds_train=regex_preds_train,
        silver_lookup=silver_lookup,
        X_silver=X_silver,
        y_silver=y_silver,
    )
    logger.info("[%s/%s] CV winner: %s", cat, attr, best_method)

    # ---- Variant G: per_attr_best ----
    preds_g = []
    for i in range(len(test_codes_list)):
        if best_method == "regex":
            pred = regex_preds_list[i]
            preds_g.append(pred if pred is not None else ml_labels_all[i])
        elif best_method == "dt":
            pred = dt_preds_raw[i]
            preds_g.append(pred if pred is not None else ml_labels_all[i])
        elif best_method == "nb":
            pred = nb_preds_raw[i]
            preds_g.append(pred if pred is not None else ml_labels_all[i])
        elif best_method == "centroid":
            pred = centroid_preds_raw[i]
            preds_g.append(pred if pred is not None else ml_labels_all[i])
        else:  # ml
            preds_g.append(ml_labels_all[i])

    # ---- Variant H: regex_then_dt_then_ml ----
    preds_h = []
    for i in range(len(test_codes_list)):
        regex_val = regex_preds_list[i]
        if regex_val is not None:
            preds_h.append(str(regex_val))
        elif dt_preds_raw[i] is not None:
            preds_h.append(dt_preds_raw[i])
        else:
            preds_h.append(ml_labels_all[i])

    # ---- Variant I: dt_then_regex_then_ml ----
    preds_i = []
    for i in range(len(test_codes_list)):
        dt_val = dt_preds_raw[i]
        if dt_val is not None:
            preds_i.append(dt_val)
        elif regex_preds_list[i] is not None:
            preds_i.append(str(regex_preds_list[i]))
        else:
            preds_i.append(ml_labels_all[i])

    # ---- Variant J: all_vote_then_ml ----
    # Vote across regex, DT, NB — if 2+ agree → use; else → ML
    preds_j = []
    for i in range(len(test_codes_list)):
        votes: list[str] = []
        if regex_preds_list[i] is not None:
            votes.append(str(regex_preds_list[i]))
        if dt_preds_raw[i] is not None:
            votes.append(dt_preds_raw[i])
        if nb_preds_raw[i] is not None:
            votes.append(nb_preds_raw[i])

        if len(votes) >= 2:
            # Check if any label has majority (≥ 2/3 votes)
            counts = Counter(votes)
            top_label, top_count = counts.most_common(1)[0]
            if top_count >= 2:
                preds_j.append(top_label)
            else:
                preds_j.append(ml_labels_all[i])
        elif len(votes) == 1:
            # Only one method fired — not enough for majority
            preds_j.append(ml_labels_all[i])
        else:
            preds_j.append(ml_labels_all[i])

    # ---- Compute accuracies ----
    n = len(y_test)
    variant_preds = {
        "G_per_attr_best": preds_g,
        "H_regex_dt_ml": preds_h,
        "I_dt_regex_ml": preds_i,
        "J_vote_then_ml": preds_j,
    }

    rows = []
    for vname, preds in variant_preds.items():
        correct = sum(1 for p, g in zip(preds, y_test) if p == g)
        acc = correct / n if n > 0 else float("nan")
        rows.append({
            "category": cat,
            "attr": attr,
            "variant": vname,
            "accuracy": acc,
            "n_test": n,
            "n_train_gold": len(train_gold),
            "n_silver": len(y_silver),
            "cv_best_method": best_method,
        })

    logger.info(
        "[%s/%s] n_test=%d | G=%.3f H=%.3f I=%.3f J=%.3f | cv_best=%s",
        cat, attr, n,
        *[rows[i]["accuracy"] for i in range(4)],
        best_method,
    )
    return rows, best_method


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)
    logger.info("Loaded gold: %d rows, %d unique codes", len(gold), gold["code"].nunique())

    all_rows: list[dict] = []
    per_attr_best_mapping: dict[str, str] = {}

    for cat in CATEGORIES:
        logger.info("=== Category: %s ===", cat)

        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        )
        silver["code"] = silver["code"].astype(str)

        emb_all = np.load(Path(PROCESSED_DIR) / f"{cat}_stratified_embeddings.npy")
        code_to_idx: dict[str, int] = {c: i for i, c in enumerate(silver["code"].tolist())}

        # Silver lookup by code (for CV text access)
        silver_lookup = silver[["code", "product_name", "ingredients_text", "brands", "quantity"]].copy()
        silver_lookup = silver_lookup.set_index("code")

        cat_gold = gold[gold["category"] == cat].copy()
        unique_codes = sorted(cat_gold["code"].unique().tolist())

        # Same 80/20 split as EXP3 (seed=42, per-category codes)
        train_codes, test_codes = train_test_split(
            unique_codes, test_size=TEST_SIZE, random_state=SEED
        )
        train_codes_set = set(train_codes)
        test_codes_set = set(test_codes)
        logger.info(
            "  Split: %d train codes, %d test codes",
            len(train_codes), len(test_codes),
        )

        test_products = silver[silver["code"].isin(test_codes_set)].copy()
        train_products = silver[silver["code"].isin(train_codes_set)].copy()

        # Pre-compute regex preds for test products
        logger.info("  Building regex predictions for test products...")
        regex_preds_test = _build_regex_preds(test_products, cat)
        logger.info("  Regex test hits: %d (code, attr) pairs",
                    sum(len(v) for v in regex_preds_test.values()))

        # Pre-compute regex preds for train products (for CV method selection)
        logger.info("  Building regex predictions for train products (for CV)...")
        regex_preds_train = _build_regex_preds(train_products, cat)
        logger.info("  Regex train hits: %d (code, attr) pairs",
                    sum(len(v) for v in regex_preds_train.values()))

        attrs = sorted(cat_gold["attr"].unique().tolist())
        logger.info("  Attrs: %s", attrs)

        for attr in attrs:
            rows, best_method = run_one_attr(
                cat=cat,
                attr=attr,
                gold=cat_gold,
                silver=silver,
                emb_all=emb_all,
                code_to_idx=code_to_idx,
                train_codes_set=train_codes_set,
                test_codes_set=test_codes_set,
                regex_preds_test=regex_preds_test,
                regex_preds_train=regex_preds_train,
                test_products=test_products,
                silver_lookup=silver_lookup,
            )
            all_rows.extend(rows)
            if best_method is not None:
                key = f"{cat}__{attr}"
                per_attr_best_mapping[key] = best_method

    result = pd.DataFrame(all_rows)
    result.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(result), OUT_PATH)

    # Save per_attr_best mapping
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    MAPPING_PATH.write_text(json.dumps(per_attr_best_mapping, indent=2, ensure_ascii=False))
    logger.info("Wrote per_attr_best mapping (%d entries) to %s",
                len(per_attr_best_mapping), MAPPING_PATH)

    _print_summary(result, per_attr_best_mapping)
    _write_doc(result, per_attr_best_mapping)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

NEW_VARIANTS = ["G_per_attr_best", "H_regex_dt_ml", "I_dt_regex_ml", "J_vote_then_ml"]

# EXP3 baselines for context (loaded from parquet if available)
EXP3_BASELINES = {
    "A_regex_ml": 0.835778,
    "B_ml_only": 0.821951,
    "C_nb_ml": 0.821561,
    "D_dt_ml": 0.829785,
    "E_centroid_ml": 0.822230,
    "F_logreg_ml": 0.822287,
}


def _print_summary(result: pd.DataFrame, mapping: dict[str, str]) -> None:
    print("\n" + "=" * 70)
    print("P2-EXP4: Layer 1.5 Optimal Mix — Grand Means")
    print("=" * 70)
    grand = result.groupby("variant")["accuracy"].mean()
    for v in NEW_VARIANTS:
        if v in grand.index:
            print(f"  {v:25s}: {grand[v] * 100:.2f}%")

    b_mean = EXP3_BASELINES["B_ml_only"]
    a_mean = EXP3_BASELINES["A_regex_ml"]

    print("\nLifts vs B_ml_only (EXP3 baseline = 82.20%):")
    for v in NEW_VARIANTS:
        if v in grand.index:
            delta = (grand[v] - b_mean) * 100
            print(f"  {v:25s}: {delta:+.2f} pp")

    print(f"\n  EXP3 best (A_regex_ml):    {a_mean * 100:.2f}%")
    print(f"  EXP3 B_ml_only baseline:   {b_mean * 100:.2f}%")

    print("\n" + "=" * 70)
    print("Per-category mean accuracy")
    print("=" * 70)
    pivot_cat = result.pivot_table(
        index="category", columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [v for v in NEW_VARIANTS if v in pivot_cat.columns]
    print(pivot_cat[cols].round(4).to_string())

    print("\n" + "=" * 70)
    print("Per-attr winner counts (CV method selection for G)")
    print("=" * 70)
    method_counts = Counter(mapping.values())
    for m in CANDIDATE_METHODS:
        print(f"  {m:12s}: {method_counts.get(m, 0)} attrs")


# ---------------------------------------------------------------------------
# Doc writer
# ---------------------------------------------------------------------------

def _write_doc(result: pd.DataFrame, mapping: dict[str, str]) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# P2-EXP4: Layer 1.5 Optimal Mix — Final Recommendation")
    lines.append("")
    lines.append("**Methodology:** Fresh 80/20 split (seed=42) identical to EXP3.")
    lines.append("Four new variants evaluated on same held-out test set.")
    lines.append("Variant G uses 5-fold CV *on train data only* to select the winning method")
    lines.append("per (cat, attr) — no test leakage.")
    lines.append("")

    grand = result.groupby("variant")["accuracy"].mean()
    b_mean = EXP3_BASELINES["B_ml_only"]
    a_mean = EXP3_BASELINES["A_regex_ml"]

    # Load EXP3 data for comparison
    exp3_grand = EXP3_BASELINES

    lines.append("## Grand Mean Accuracy — All Variants (EXP3 + EXP4)")
    lines.append("")
    lines.append("| Variant | Source | Accuracy | vs B (pp) |")
    lines.append("|---------|--------|----------|-----------|")
    for v, acc in sorted(EXP3_BASELINES.items(), key=lambda x: -x[1]):
        delta = (acc - b_mean) * 100
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {v} | EXP3 | {acc * 100:.2f}% | {sign}{delta:.2f} |")
    for v in NEW_VARIANTS:
        if v in grand.index:
            acc = grand[v]
            delta = (acc - b_mean) * 100
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {v} | **EXP4** | {acc * 100:.2f}% | {sign}{delta:.2f} |")
    lines.append("")

    # Per-category
    if len(result) > 0:
        pivot_cat = result.pivot_table(
            index="category", columns="variant", values="accuracy", aggfunc="mean"
        )
        cols = [v for v in NEW_VARIANTS if v in pivot_cat.columns]
        if cols:
            lines.append("## Per-Category Mean Accuracy (EXP4 variants)")
            lines.append("")
            header = "| Category | " + " | ".join(cols) + " |"
            sep = "|----------|" + "---------|" * len(cols)
            lines.append(header)
            lines.append(sep)
            for cat, row in pivot_cat[cols].round(4).iterrows():
                vals = " | ".join(f"{v*100:.2f}%" for v in row)
                lines.append(f"| {cat} | {vals} |")
            lines.append("")

        # Per-(cat, attr)
        pivot_attr = result.pivot_table(
            index=["category", "attr"], columns="variant", values="accuracy", aggfunc="mean"
        )
        cols_attr = [v for v in NEW_VARIANTS if v in pivot_attr.columns]
        if cols_attr:
            pivot_attr = pivot_attr[cols_attr].round(4)
            pivot_attr["winner_exp4"] = pivot_attr.idxmax(axis=1)

            lines.append("## Per-(Category, Attr) Accuracy (EXP4 variants)")
            lines.append("")
            header2 = "| Category | Attr | " + " | ".join(cols_attr) + " | Winner (EXP4) | CV Best Method |"
            sep2 = "|----------|------|" + "---------|" * len(cols_attr) + "---------------|----------------|"
            lines.append(header2)
            lines.append(sep2)

            cv_best_map = result[result["variant"] == "G_per_attr_best"][["category", "attr", "cv_best_method"]].drop_duplicates()
            cv_best_lookup = {(r["category"], r["attr"]): r["cv_best_method"] for _, r in cv_best_map.iterrows()}

            for (cat, attr), row in pivot_attr.iterrows():
                acc_vals = " | ".join(f"{v*100:.2f}%" for v in row[cols_attr])
                winner = row["winner_exp4"]
                cv_m = cv_best_lookup.get((cat, attr), "N/A")
                lines.append(f"| {cat} | {attr} | {acc_vals} | {winner} | {cv_m} |")
            lines.append("")

    # per_attr_best mapping
    lines.append("## Per-Attr Best Method Mapping (CV-Selected on Train)")
    lines.append("")
    lines.append("| Category_Attr | CV Winner |")
    lines.append("|---------------|-----------|")
    for key, method in sorted(mapping.items()):
        lines.append(f"| {key} | {method} |")
    lines.append("")

    # Winner counts
    method_counts = Counter(mapping.values())
    lines.append("### Method win counts (CV selection on train):")
    lines.append("")
    lines.append("| Method | # Attrs |")
    lines.append("|--------|---------|")
    for m in CANDIDATE_METHODS:
        lines.append(f"| {m} | {method_counts.get(m, 0)} |")
    lines.append("")

    # Production recommendation
    if len(grand) > 0:
        best_exp4 = grand.idxmax()
        best_acc = grand[best_exp4]
        all_accs = dict(exp3_grand)
        all_accs.update({v: grand[v] for v in NEW_VARIANTS if v in grand.index})
        overall_best = max(all_accs, key=lambda k: all_accs[k])
        overall_best_acc = all_accs[overall_best]

        lines.append("## Production Recommendation")
        lines.append("")
        lines.append(
            f"Overall best variant across EXP3+EXP4: **{overall_best}** "
            f"({overall_best_acc * 100:.2f}% grand mean accuracy, "
            f"{(overall_best_acc - b_mean) * 100:+.2f} pp vs ML-only)."
        )
        lines.append("")
        lines.append(
            f"Best EXP4 variant: **{best_exp4}** ({best_acc * 100:.2f}%)."
        )
        lines.append("")

        g_acc = grand.get("G_per_attr_best", 0.0)
        h_acc = grand.get("H_regex_dt_ml", 0.0)
        i_acc = grand.get("I_dt_regex_ml", 0.0)
        j_acc = grand.get("J_vote_then_ml", 0.0)

        lines.append("Key findings:")
        lines.append(f"- G (per_attr_best CV): {g_acc * 100:.2f}% ({(g_acc - b_mean) * 100:+.2f} pp vs ML-only)")
        lines.append(f"- H (regex→DT→ML): {h_acc * 100:.2f}% ({(h_acc - b_mean) * 100:+.2f} pp vs ML-only)")
        lines.append(f"- I (DT→regex→ML): {i_acc * 100:.2f}% ({(i_acc - b_mean) * 100:+.2f} pp vs ML-only)")
        lines.append(f"- J (vote→ML): {j_acc * 100:.2f}% ({(j_acc - b_mean) * 100:+.2f} pp vs ML-only)")
        lines.append("")
        lines.append(
            "**Final recommendation:** Deploy the cascade variant with highest lift. "
            "If G wins, use the per-attr mapping (saved in `models/per_attr_best_mapping.json`). "
            "If H or I wins, use the simpler sequential cascade (no per-attr lookup required). "
            "J (soft vote) has lowest operational overhead but may not outperform regex-only cascade."
        )

    doc_text = "\n".join(lines) + "\n"
    DOC_PATH.write_text(doc_text, encoding="utf-8")
    logger.info("Wrote doc to %s", DOC_PATH)


if __name__ == "__main__":
    main()
