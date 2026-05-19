"""P2-EXP8: Bayes revival — train DiscreteBayesianNetwork on v2 gold + use in ensemble.

Hypothesis: Bayesian network trained on CLEAN gold data (not silver tags) can leverage
inter-attribute correlations and improve accuracy when used as an ensemble component
(not as a cascade layer).

3 variants compared:
  P_2          - (LightGBM + XGB) / 2  [baseline from EXP7 = 86.87%]
  P_3          - (LightGBM + XGB + Bayes) / 3
  P_3_weighted - (LightGBM + XGB + 0.5*Bayes) / 2.5  [Bayes downweighted]

Same 80/20 split (seed=42) as EXP7.

Output:
  datasets/processed/bayes_ensemble.parquet
  docs/bayes_revival_findings.md
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from pgmpy.models import DiscreteBayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork as DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.inference import VariableElimination

from src.common import PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)

CATEGORIES = ["pasta", "chocolate", "cheeses"]
SEED = 42
TEST_SIZE = 0.2
MIN_GOLD = 20
BAYES_TOP_BRANDS = 15   # top N brands to keep (rest → "other")
BAYES_MIN_ROWS = 50     # skip BN learning if train set too small

GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet"
OUT_PARQUET = Path(PROCESSED_DIR) / "bayes_ensemble.parquet"
OUT_MD = Path("docs") / "bayes_revival_findings.md"


# ---------------------------------------------------------------------------
# Text builder (same as EXP7)
# ---------------------------------------------------------------------------

def _build_text(row: pd.Series) -> str:
    parts = []
    for col in ["product_name", "ingredients_text", "brands"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Fresh hybrid XGBoost trainer (same as EXP7)
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


# ---------------------------------------------------------------------------
# LightGBM trainer (same as EXP7)
# ---------------------------------------------------------------------------

def _train_lgbm(
    train_texts: list[str],
    y_train: list[str],
) -> tuple[Optional[lgb.LGBMClassifier], Optional[TfidfVectorizer], Optional[LabelEncoder]]:
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
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=5,
            objective="binary", verbose=-1,
        )
    else:
        clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=5,
            objective="multiclass", num_class=n_classes, verbose=-1,
        )

    clf.fit(X_tfidf, y_enc)
    return clf, vec, le


def _lgbm_probas(
    clf: lgb.LGBMClassifier,
    vec: TfidfVectorizer,
    texts: list[str],
) -> np.ndarray:
    X = vec.transform(texts)
    return clf.predict_proba(X)


# ---------------------------------------------------------------------------
# Label space alignment helper (same as EXP7 _align_probas)
# ---------------------------------------------------------------------------

def _align_probas(
    a_probas: np.ndarray,
    a_le: LabelEncoder,
    b_probas: np.ndarray,
    b_le: LabelEncoder,
    c_probas: Optional[np.ndarray] = None,
    c_classes: Optional[list[str]] = None,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, list[str]]:
    """Average probas over shared label space with optional third component."""
    a_classes = list(a_le.classes_)
    b_classes = list(b_le.classes_)

    if c_probas is not None and c_classes is not None:
        all_classes = sorted(set(a_classes) | set(b_classes) | set(c_classes))
    else:
        all_classes = sorted(set(a_classes) | set(b_classes))

    n = a_probas.shape[0]
    k = len(all_classes)
    wa, wb, wc = weights
    total_w = wa + wb + (wc if c_probas is not None else 0.0)

    a_full = np.zeros((n, k))
    for j, cls in enumerate(a_classes):
        col = all_classes.index(cls)
        a_full[:, col] = a_probas[:, j]

    b_full = np.zeros((n, k))
    for j, cls in enumerate(b_classes):
        col = all_classes.index(cls)
        b_full[:, col] = b_probas[:, j]

    avg = (wa * a_full + wb * b_full)

    if c_probas is not None and c_classes is not None:
        c_full = np.zeros((n, k))
        for j, cls in enumerate(c_classes):
            col = all_classes.index(cls)
            c_full[:, col] = c_probas[:, j]
        avg = avg + wc * c_full

    avg = avg / total_w
    return avg, all_classes


# ---------------------------------------------------------------------------
# Bayesian Network training on GOLD data
# ---------------------------------------------------------------------------

def _normalize_brand(series: pd.Series, top_n: int = BAYES_TOP_BRANDS) -> pd.Series:
    """Normalize brand to top N, collapse rest to 'other'."""
    s = series.fillna("unknown").astype(str)
    s = s.str.split(",").str[0].str.strip().str.lower().replace("", "unknown")
    top = s.value_counts().head(top_n).index
    return s.where(s.isin(top), other="other")


def _discretize_col(series: pd.Series, n_bins: int = 5) -> pd.Series:
    """Discretize a numeric-looking column into bins."""
    try:
        num = pd.to_numeric(series, errors="coerce")
        if num.notna().sum() < 10:
            return series.fillna("unknown").astype(str)
        labels = [f"bin{i}" for i in range(n_bins)]
        binned = pd.cut(num, bins=n_bins, labels=labels, duplicates="drop")
        return binned.astype(str).where(num.notna(), "unknown")
    except Exception:
        return series.fillna("unknown").astype(str)


def _build_bayes_train_data(
    cat: str,
    cat_gold_wide: pd.DataFrame,  # code x attrs (non-null pivot)
    silver: pd.DataFrame,
) -> pd.DataFrame:
    """Build discrete DataFrame for Bayes structure learning.

    Columns: brand_norm + all attrs that are present.
    """
    # Add brand from silver
    merged = cat_gold_wide.reset_index().merge(
        silver[["code", "brands"]], on="code", how="left"
    )
    data = pd.DataFrame()
    data["brand_norm"] = _normalize_brand(merged["brands"], top_n=BAYES_TOP_BRANDS)

    # Collect all attr columns
    attrs_in_data = [c for c in cat_gold_wide.columns]
    for attr in attrs_in_data:
        vals = merged[attr].copy()
        # Detect numeric-like columns (nutri_score_grade is letter A-E so string is fine)
        num_test = pd.to_numeric(vals.dropna(), errors="coerce")
        if num_test.notna().sum() > 5 and num_test.notna().sum() / max(len(vals.dropna()), 1) > 0.5:
            data[attr] = _discretize_col(vals, n_bins=5)
        else:
            # Categorical — keep top values, collapse rest
            s = vals.fillna("unknown").astype(str)
            top_vals = s.value_counts().head(10).index
            data[attr] = s.where(s.isin(top_vals), other="other")

    return data.dropna()


def _train_bayes(data: pd.DataFrame, prefix: str) -> Optional[tuple]:
    """Train DiscreteBayesianNetwork via HillClimb + BIC.

    Returns (model, inference, col_list) or None if training fails.
    """
    if len(data) < BAYES_MIN_ROWS:
        logger.warning("[%s] Bayes: too few rows (%d), using fallback manual structure", prefix, len(data))
        return _train_bayes_fallback(data, prefix)

    logger.info("[%s] Bayes: running HillClimb on %d rows, %d nodes", prefix, len(data), len(data.columns))

    try:
        hc = HillClimbSearch(data)
        best_model = hc.estimate(scoring_method="bic-d", max_indegree=3)
        edges = list(best_model.edges())

        if not edges:
            logger.warning("[%s] Bayes: no edges discovered, using fallback", prefix)
            return _train_bayes_fallback(data, prefix)

        logger.info("[%s] Bayes: %d edges discovered", prefix, len(edges))
        for src, dst in edges:
            logger.info("[%s]   %s -> %s", prefix, src, dst)

        model = DiscreteBayesianNetwork(edges)
        est = BayesianEstimator(model, data)
        for node in model.nodes():
            cpd = est.estimate_cpd(node, prior_type="BDeu", equivalent_sample_size=10)
            model.add_cpds(cpd)
        model.check_model()

        inference_engine = VariableElimination(model)
        logger.info("[%s] Bayes: model trained OK with %d nodes", prefix, len(list(model.nodes())))
        return model, inference_engine, list(data.columns)

    except Exception as e:
        logger.warning("[%s] Bayes: HillClimb failed (%s), using fallback", prefix, e)
        return _train_bayes_fallback(data, prefix)


def _train_bayes_fallback(data: pd.DataFrame, prefix: str) -> Optional[tuple]:
    """Fallback: simple star structure brand_norm -> each attr."""
    non_brand_cols = [c for c in data.columns if c != "brand_norm"]
    if not non_brand_cols:
        return None

    edges = [("brand_norm", col) for col in non_brand_cols]
    try:
        model = DiscreteBayesianNetwork(edges)
        est = BayesianEstimator(model, data)
        for node in model.nodes():
            cpd = est.estimate_cpd(node, prior_type="BDeu", equivalent_sample_size=10)
            model.add_cpds(cpd)
        model.check_model()
        inference_engine = VariableElimination(model)
        logger.info("[%s] Bayes fallback: star model trained OK", prefix)
        return model, inference_engine, list(data.columns)
    except Exception as e:
        logger.error("[%s] Bayes fallback also failed: %s", prefix, e)
        return None


def _bayes_predict_proba_for_attr(
    model,
    inference_engine: VariableElimination,
    target_attr: str,
    brand_norm: str,
    other_attr_preds: dict[str, str],  # attr -> best_label from LightGBM
) -> tuple[Optional[np.ndarray], Optional[list[str]]]:
    """Compute P(target_attr | brand_norm, other_attrs).

    Returns (proba_array, class_list) aligned to model's state names.
    """
    model_nodes = set(model.nodes())

    if target_attr not in model_nodes:
        return None, None

    evidence: dict[str, str] = {}

    # Add brand evidence
    if "brand_norm" in model_nodes:
        try:
            cpd = model.get_cpds("brand_norm")
            known_brands = list(cpd.state_names["brand_norm"])
            bn = brand_norm if brand_norm in known_brands else "other"
            evidence["brand_norm"] = bn
        except Exception:
            pass

    # Add other attr predictions as evidence
    for attr, val in other_attr_preds.items():
        if attr == target_attr or attr not in model_nodes:
            continue
        try:
            cpd = model.get_cpds(attr)
            known_states = list(cpd.state_names[attr])
            str_val = str(val)
            if str_val in known_states:
                evidence[attr] = str_val
            # else skip this evidence (unknown state)
        except Exception:
            pass

    try:
        result = inference_engine.query([target_attr], evidence=evidence, show_progress=False)
        class_list = list(result.state_names[target_attr])
        proba_array = np.array([float(result.values[i]) for i in range(len(class_list))])
        return proba_array, class_list
    except Exception as e:
        logger.debug("[%s] Bayes query failed for evidence=%s: %s", target_attr, evidence, e)
        return None, None


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
    # Bayes model (trained on all train codes × all attrs)
    bayes_model,
    bayes_inference,
    # LightGBM top preds for test codes (for Bayes evidence), keyed by code->attr->label
    lgbm_top_preds_test: dict[str, dict[str, str]],
) -> list[dict]:
    """Run 3 ensemble variants for one (cat, attr)."""
    cat_gold = gold[
        (gold["category"] == cat) & (gold["attr"] == attr) & ~gold["gold_is_null"]
    ].copy()
    cat_gold["code"] = cat_gold["code"].astype(str)
    cat_gold = cat_gold[cat_gold["code"].isin(code_to_idx)]

    if len(cat_gold) < MIN_GOLD:
        return []

    train_gold = cat_gold[cat_gold["code"].isin(train_codes_set)]
    test_gold = cat_gold[cat_gold["code"].isin(test_codes_set)]

    if len(train_gold) < 10 or len(test_gold) < 5:
        return []

    # Embedding arrays
    train_idx = np.array([code_to_idx[c] for c in train_gold["code"]])
    test_idx = np.array([code_to_idx[c] for c in test_gold["code"]])

    X_gold_train = emb_all[train_idx]
    y_gold_train = train_gold["gold_value"].astype(str).values
    X_test_emb = emb_all[test_idx]
    y_test = test_gold["gold_value"].astype(str).values
    test_codes_list = test_gold["code"].tolist()

    # Silver training data (exclude test + train codes)
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

    # Train fresh hybrid ML (XGBoost)
    if len(X_silver) > 0:
        clf_xgb, le_xgb = _train_fresh_hybrid(X_silver, y_silver, X_gold_train, y_gold_train)
    else:
        all_classes = sorted(set(y_gold_train.tolist()))
        if len(all_classes) < 2:
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
        return []

    # XGBoost probas
    ml_probas = clf_xgb.predict_proba(X_test_emb)  # (n, k_xgb)

    # Build text features for LightGBM
    train_gold_with_text = train_gold.merge(
        silver[["code", "product_name", "ingredients_text", "brands"]],
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

    lgbm_clf, lgbm_vec, lgbm_le = _train_lgbm(train_texts, y_train_texts)

    n = len(y_test)

    def _acc(preds: list[str]) -> float:
        return sum(1 for p, g in zip(preds, y_test) if p == g) / n if n > 0 else float("nan")

    # --- Variant P_2: LightGBM + XGBoost ensemble (baseline = EXP7 P_ensemble) ---
    preds_p2: list[str] = []
    lgbm_pr: Optional[np.ndarray] = None

    if lgbm_clf is not None and lgbm_vec is not None and lgbm_le is not None:
        lgbm_pr = _lgbm_probas(lgbm_clf, lgbm_vec, test_texts_list)  # (n, k_lgbm)
        avg_p2, merged_classes_p2 = _align_probas(lgbm_pr, lgbm_le, ml_probas, le_xgb,
                                                    weights=(1.0, 1.0, 1.0))
        top_idx_p2 = np.argmax(avg_p2, axis=1)
        preds_p2 = [merged_classes_p2[i] for i in top_idx_p2]
    else:
        # Fallback: XGBoost only
        enc_preds = np.argmax(ml_probas, axis=1)
        preds_p2 = le_xgb.inverse_transform(enc_preds).tolist()

    acc_p2 = _acc(preds_p2)

    # --- Compute LightGBM top-1 labels for Bayes evidence ---
    # For each test code, store LightGBM's top-1 prediction for this attr
    # (to be used as evidence for OTHER attrs' Bayes queries)
    if lgbm_pr is not None:
        for i, code in enumerate(test_codes_list):
            top_idx = int(np.argmax(lgbm_pr[i]))
            top_label = str(lgbm_le.inverse_transform([top_idx])[0])
            if code not in lgbm_top_preds_test:
                lgbm_top_preds_test[code] = {}
            lgbm_top_preds_test[code][attr] = top_label

    # --- Variant P_3: LightGBM + XGB + Bayes (equal weight 1/3 each) ---
    # --- Variant P_3_weighted: LightGBM + XGB + 0.5*Bayes ---
    preds_p3: list[str] = preds_p2[:]
    preds_p3w: list[str] = preds_p2[:]
    bayes_success_count = 0

    if bayes_model is not None and bayes_inference is not None and lgbm_clf is not None:
        # We need brands for the test codes
        test_brands_map = {}
        if "brands" in silver.columns:
            silver_brands = silver[silver["code"].isin(set(test_codes_list))][["code", "brands"]]
            silver_brands["code"] = silver_brands["code"].astype(str)
            for _, row in silver_brands.iterrows():
                brand_raw = str(row.get("brands", "unknown"))
                brand_norm = brand_raw.split(",")[0].strip().lower() or "unknown"
                test_brands_map[row["code"]] = brand_norm

        merged_classes_p3 = None
        merged_classes_p3w = None

        for i, code in enumerate(test_codes_list):
            brand_norm = test_brands_map.get(code, "other")
            other_preds = lgbm_top_preds_test.get(code, {})

            bayes_pr, bayes_classes = _bayes_predict_proba_for_attr(
                bayes_model, bayes_inference, attr, brand_norm, other_preds
            )

            if bayes_pr is None or bayes_classes is None:
                preds_p3[i] = preds_p2[i]
                preds_p3w[i] = preds_p2[i]
                continue

            bayes_success_count += 1

            # Build per-sample arrays for alignment
            lgbm_pr_i = lgbm_pr[i:i+1] if lgbm_pr is not None else None
            ml_pr_i = ml_probas[i:i+1]
            bayes_pr_i = bayes_pr.reshape(1, -1)

            if lgbm_pr_i is not None:
                # P_3: equal weight
                avg_p3, cls_p3 = _align_probas(
                    lgbm_pr_i, lgbm_le, ml_pr_i, le_xgb,
                    c_probas=bayes_pr_i, c_classes=bayes_classes,
                    weights=(1.0, 1.0, 1.0),
                )
                top_p3 = int(np.argmax(avg_p3[0]))
                preds_p3[i] = cls_p3[top_p3]

                # P_3_weighted: bayes at 0.5
                avg_p3w, cls_p3w = _align_probas(
                    lgbm_pr_i, lgbm_le, ml_pr_i, le_xgb,
                    c_probas=bayes_pr_i, c_classes=bayes_classes,
                    weights=(1.0, 1.0, 0.5),
                )
                top_p3w = int(np.argmax(avg_p3w[0]))
                preds_p3w[i] = cls_p3w[top_p3w]
            else:
                preds_p3[i] = preds_p2[i]
                preds_p3w[i] = preds_p2[i]

    acc_p3 = _acc(preds_p3)
    acc_p3w = _acc(preds_p3w)
    bayes_cov = bayes_success_count / n if n > 0 else 0.0

    logger.info(
        "[%s/%s] n=%d | P_2=%.3f P_3=%.3f P_3w=%.3f | bayes_cov=%.2f",
        cat, attr, n, acc_p2, acc_p3, acc_p3w, bayes_cov,
    )

    base_row = dict(
        category=cat, attr=attr, n_test=n,
        n_train_gold=len(train_gold), n_silver=len(y_silver),
    )
    return [
        {**base_row, "variant": "P_2",           "accuracy": acc_p2,  "bayes_coverage": 0.0},
        {**base_row, "variant": "P_3",            "accuracy": acc_p3,  "bayes_coverage": bayes_cov},
        {**base_row, "variant": "P_3_weighted",   "accuracy": acc_p3w, "bayes_coverage": bayes_cov},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)
    logger.info("Loaded gold: %d rows, %d unique codes", len(gold), gold["code"].nunique())

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

        # Same 80/20 split as EXP7 (seed=42)
        train_codes, test_codes = train_test_split(
            unique_codes, test_size=TEST_SIZE, random_state=SEED
        )
        train_codes_set = set(train_codes)
        test_codes_set = set(test_codes)
        logger.info("  Split: %d train codes, %d test codes", len(train_codes), len(test_codes))

        test_products = silver[silver["code"].isin(test_codes_set)].copy()

        # ---------------------------------------------------------------
        # Train ONE Bayes model per category on 80% gold (all attrs wide)
        # ---------------------------------------------------------------
        # Build wide pivot: code x attrs for train codes
        train_gold_all = cat_gold[
            cat_gold["code"].isin(train_codes_set) & ~cat_gold["gold_is_null"]
        ]
        gold_wide_train = train_gold_all.pivot_table(
            index="code", columns="attr", values="gold_value", aggfunc="first"
        )
        # Only keep codes that have embeddings
        valid_train_codes = [c for c in gold_wide_train.index if c in code_to_idx]
        gold_wide_train = gold_wide_train.loc[valid_train_codes]

        bayes_data = _build_bayes_train_data(cat, gold_wide_train, silver)
        logger.info("[%s] Bayes train data: %d rows", cat, len(bayes_data))

        bayes_result = _train_bayes(bayes_data, prefix=cat)
        if bayes_result is not None:
            bayes_model, bayes_inference, _bayes_cols = bayes_result
        else:
            bayes_model, bayes_inference = None, None
            logger.warning("[%s] Bayes training failed entirely, P_3 will equal P_2", cat)

        # Dict to accumulate LightGBM top-1 preds across attrs for Bayes evidence
        lgbm_top_preds_test: dict[str, dict[str, str]] = {}

        attrs = sorted(cat_gold["attr"].unique().tolist())
        logger.info("  Attrs: %s", attrs)

        # First pass: populate lgbm_top_preds_test by running through all attrs
        # We need this for Bayes evidence, but it's built incrementally in run_one_attr
        # Process attrs in alphabetical order — earlier attrs become evidence for later ones
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
                bayes_model=bayes_model,
                bayes_inference=bayes_inference,
                lgbm_top_preds_test=lgbm_top_preds_test,
            )
            all_rows.extend(rows)

    result = pd.DataFrame(all_rows)
    result.to_parquet(OUT_PARQUET, index=False)
    logger.info("Wrote %d rows to %s", len(result), OUT_PARQUET)

    summary = _build_summary(result)
    _print_summary(summary, result)
    _write_md(summary, result)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

VARIANT_ORDER = ["P_2", "P_3", "P_3_weighted"]


def _build_summary(result: pd.DataFrame) -> pd.DataFrame:
    grand = result.groupby("variant").agg(
        mean_acc=("accuracy", "mean"),
        mean_bayes_cov=("bayes_coverage", "mean"),
    ).reindex(VARIANT_ORDER)
    return grand


def _print_summary(summary: pd.DataFrame, result: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("P2-EXP8: Bayes Revival as Ensemble Component — Grand Means")
    print("=" * 70)
    p2_acc = summary.loc["P_2", "mean_acc"]
    for v in VARIANT_ORDER:
        acc = summary.loc[v, "mean_acc"]
        delta = (acc - p2_acc) * 100
        cov = summary.loc[v, "mean_bayes_cov"]
        cov_str = f"  bayes_cov={cov*100:.1f}%" if v != "P_2" else ""
        print(f"  {v:20s}: {acc*100:.2f}%  ({delta:+.2f} pp vs P_2){cov_str}")

    print("\n" + "=" * 70)
    print("Per-(category, attr) accuracy")
    print("=" * 70)
    pivot = result.pivot_table(
        index=["category", "attr"], columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [c for c in VARIANT_ORDER if c in pivot.columns]
    pivot = pivot[cols].copy()
    if "P_3" in pivot.columns and "P_2" in pivot.columns:
        pivot["P3_vs_P2_pp"] = (pivot["P_3"] - pivot["P_2"]) * 100
    print(pivot.round(4).to_string())

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    p3_acc = summary.loc["P_3", "mean_acc"]
    p3w_acc = summary.loc["P_3_weighted", "mean_acc"]
    delta_p3 = (p3_acc - p2_acc) * 100
    delta_p3w = (p3w_acc - p2_acc) * 100
    if delta_p3 >= 0.5:
        verdict = f"Bayes adds +{delta_p3:.2f} pp — REVIVE Bayes as ensemble component."
    elif delta_p3 >= 0.0:
        verdict = f"Bayes adds +{delta_p3:.2f} pp — marginal gain, not worth added complexity."
    else:
        verdict = f"Bayes hurts {delta_p3:.2f} pp — keep DEPRECATED."
    print(f"  P_2 (LightGBM+XGB):          {p2_acc*100:.2f}%")
    print(f"  P_3 (LightGBM+XGB+Bayes):    {p3_acc*100:.2f}%  ({delta_p3:+.2f} pp)")
    print(f"  P_3w (Bayes 0.5x weight):    {p3w_acc*100:.2f}%  ({delta_p3w:+.2f} pp)")
    print(f"\n  {verdict}")


def _write_md(summary: pd.DataFrame, result: pd.DataFrame) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    p2_acc = summary.loc["P_2", "mean_acc"]
    p3_acc = summary.loc["P_3", "mean_acc"]
    p3w_acc = summary.loc["P_3_weighted", "mean_acc"]
    delta_p3 = (p3_acc - p2_acc) * 100
    delta_p3w = (p3w_acc - p2_acc) * 100

    pivot = result.pivot_table(
        index=["category", "attr"], columns="variant", values="accuracy", aggfunc="mean"
    )

    if delta_p3 >= 0.5:
        verdict = f"Bayes adds **+{delta_p3:.2f} pp** — consider reviving as ensemble component."
        verdict_short = "REVIVE"
    elif delta_p3 >= 0.0:
        verdict = f"Bayes adds **+{delta_p3:.2f} pp** — marginal; not worth added complexity."
        verdict_short = "MARGINAL"
    else:
        verdict = f"Bayes hurts **{delta_p3:.2f} pp** — keep deprecated."
        verdict_short = "DEPRECATED"

    lines = [
        "# P2-EXP8: Bayes Revival on Gold as Ensemble Component",
        "",
        "**Date:** 2026-05-17  ",
        "**Script:** `src/experiments/bayes_ensemble.py`  ",
        "**Output:** `datasets/processed/bayes_ensemble.parquet`  ",
        "",
        "## Setup",
        "",
        "- DiscreteBayesianNetwork trained on 80% v2 gold codes (HillClimb + BIC, max_indegree=3)",
        "- Same 80/20 split (seed=42) as EXP7 on `consensus_gold_v2_expanded.parquet`",
        "- 3 categories: pasta, chocolate, cheeses",
        "- Ensemble: soft-vote average over shared label space",
        "- Bayes evidence: brand_norm + LightGBM top-1 predictions for other attrs",
        "",
        "## Grand Mean Results",
        "",
        "| Variant | Accuracy | vs P_2 (pp) | Description |",
        "|---------|----------|-------------|-------------|",
        f"| P_2 (baseline)   | {p2_acc*100:.2f}% | +0.00 | LightGBM + XGB soft-vote (EXP7 P_ensemble) |",
        f"| P_3              | {p3_acc*100:.2f}% | {delta_p3:+.2f} | (LightGBM + XGB + Bayes) / 3 |",
        f"| P_3_weighted     | {p3w_acc*100:.2f}% | {delta_p3w:+.2f} | (LightGBM + XGB + 0.5×Bayes) / 2.5 |",
        "",
        "## Per-(Category, Attr) Detail",
        "",
        "```",
        pivot.round(4).to_string(),
        "```",
        "",
        "## Verdict",
        "",
        f"**{verdict_short}**: {verdict}",
        "",
        "- P_2 (LightGBM + XGB): **{:.2f}%**".format(p2_acc * 100),
        "- P_3 (+Bayes 1/3):     **{:.2f}%** ({:+.2f} pp)".format(p3_acc * 100, delta_p3),
        "- P_3w (+Bayes 0.5x):   **{:.2f}%** ({:+.2f} pp)".format(p3w_acc * 100, delta_p3w),
    ]

    # Per-cat breakdown
    lines += ["", "## Per-Category Delta (P_3 vs P_2)"]
    lines += ["", "| Category | P_2 | P_3 | Delta (pp) |", "|----------|-----|-----|------------|"]
    for cat in CATEGORIES:
        cat_rows = result[result["category"] == cat]
        cat_p2 = cat_rows[cat_rows["variant"] == "P_2"]["accuracy"].mean()
        cat_p3 = cat_rows[cat_rows["variant"] == "P_3"]["accuracy"].mean()
        d = (cat_p3 - cat_p2) * 100
        lines.append(f"| {cat:10s} | {cat_p2*100:.2f}% | {cat_p3*100:.2f}% | {d:+.2f} |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    logger.info("Wrote findings to %s", OUT_MD)


if __name__ == "__main__":
    main()
