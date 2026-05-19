"""P2-EXP7: Is ML hybrid_v2 layer redundant when LightGBM Layer 1.5 is in place?

5-variant comparison on honest 80/20 (seed=42, same as EXP5b):

  B  ml_only          — hybrid_v2 XGBoost only (baseline 82.20%)
  M  lightgbm_τ_ml   — LightGBM @τ=0.85 → ML fallback (current 86.24%)
  N  lightgbm_only   — LightGBM always predict, no τ, no fallback
  O  lightgbm_lowτ   — LightGBM @τ=0.5 → ML fallback
  P  ensemble        — Soft-vote average of LightGBM proba + ML proba

Also reports per-(cat, attr) LightGBM fires-rate at τ=0.85 (% predictions
made by LightGBM itself, not fallback).

Output:
  datasets/processed/lightgbm_alone_check.parquet
  docs/lightgbm_alone_findings.md
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.common import PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)

CATEGORIES = ["pasta", "chocolate", "cheeses"]
SEED = 42
TEST_SIZE = 0.2
MIN_GOLD = 20
LGBM_TAU_HIGH = 0.85   # Variant M
LGBM_TAU_LOW = 0.50    # Variant O

GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet"
OUT_PARQUET = Path(PROCESSED_DIR) / "lightgbm_alone_check.parquet"
OUT_MD = Path("docs") / "lightgbm_alone_findings.md"


# ---------------------------------------------------------------------------
# Text builder (same as lightgbm_only.py)
# ---------------------------------------------------------------------------

def _build_text(row: pd.Series) -> str:
    parts = []
    for col in ["product_name", "ingredients_text", "brands"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Fresh hybrid XGBoost trainer (same as lightgbm_only.py)
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
# LightGBM trainer on TF-IDF features (same as lightgbm_only.py)
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
    """Return raw proba matrix (n_samples, n_classes)."""
    X = vec.transform(texts)
    return clf.predict_proba(X)


# ---------------------------------------------------------------------------
# Label space alignment helper (for ensemble)
# ---------------------------------------------------------------------------

def _align_probas(
    lgbm_probas: np.ndarray,
    lgbm_le: LabelEncoder,
    ml_probas: np.ndarray,
    ml_le: LabelEncoder,
) -> tuple[np.ndarray, list[str]]:
    """Average LightGBM and ML probas over a shared label space.

    Returns (averaged proba matrix, merged class list).
    Classes present in only one model get zero weight from the other.
    """
    lgbm_classes = list(lgbm_le.classes_)
    ml_classes = list(ml_le.classes_)
    all_classes = sorted(set(lgbm_classes) | set(ml_classes))
    n = lgbm_probas.shape[0]
    k = len(all_classes)

    lgbm_full = np.zeros((n, k))
    for j, cls in enumerate(lgbm_classes):
        col = all_classes.index(cls)
        lgbm_full[:, col] = lgbm_probas[:, j]

    ml_full = np.zeros((n, k))
    for j, cls in enumerate(ml_classes):
        col = all_classes.index(cls)
        ml_full[:, col] = ml_probas[:, j]

    avg = 0.5 * lgbm_full + 0.5 * ml_full
    return avg, all_classes


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
    """Run all 5 variants for one (cat, attr). Returns list of result dicts."""
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

    # Build embedding arrays
    train_idx = np.array([code_to_idx[c] for c in train_gold["code"]])
    test_idx = np.array([code_to_idx[c] for c in test_gold["code"]])

    X_gold_train = emb_all[train_idx]
    y_gold_train = train_gold["gold_value"].astype(str).values
    X_test_emb = emb_all[test_idx]
    y_test = test_gold["gold_value"].astype(str).values

    # Build silver training data (exclude test + train codes)
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

    # --- Train fresh hybrid ML (XGBoost) ---
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

    # --- ML (XGBoost) predictions and probas ---
    ml_probas = clf_xgb.predict_proba(X_test_emb)  # shape (n, k_ml)
    enc_preds = np.argmax(ml_probas, axis=1)
    ml_labels_all = le_xgb.inverse_transform(enc_preds).tolist()

    # --- Build text features for LightGBM ---
    train_gold_with_text = train_gold.merge(
        silver[["code", "product_name", "ingredients_text", "brands"]],
        on="code", how="left",
    )
    train_texts = [_build_text(row) for _, row in train_gold_with_text.iterrows()]
    y_train_texts = train_gold_with_text["gold_value"].astype(str).tolist()

    test_products_idx = test_products.set_index("code")
    test_texts_list = []
    for c in test_gold["code"].tolist():
        if c in test_products_idx.index:
            test_texts_list.append(_build_text(test_products_idx.loc[c]))
        else:
            test_texts_list.append("")

    # --- Train LightGBM ---
    lgbm_clf, lgbm_vec, lgbm_le = _train_lgbm(train_texts, y_train_texts)

    n = len(y_test)

    def _acc(preds: list[str]) -> float:
        return sum(1 for p, g in zip(preds, y_test) if p == g) / n if n > 0 else float("nan")

    # --- Variant B: ml_only ---
    preds_b = ml_labels_all[:]
    acc_b = _acc(preds_b)

    # --- Variants M, N, O, P require LightGBM ---
    preds_m: list[str] = ml_labels_all[:]
    preds_n: list[str] = ml_labels_all[:]
    preds_o: list[str] = ml_labels_all[:]
    preds_p: list[str] = ml_labels_all[:]
    lgbm_fires_m = 0
    lgbm_fires_o = 0

    if lgbm_clf is not None and lgbm_vec is not None and lgbm_le is not None:
        lgbm_pr = _lgbm_probas(lgbm_clf, lgbm_vec, test_texts_list)  # (n, k_lgbm)

        # Variant N: LightGBM always (no τ)
        top_idx_n = np.argmax(lgbm_pr, axis=1)
        preds_n = lgbm_le.inverse_transform(top_idx_n).tolist()

        # Variants M and O: τ-gated fallback
        for i, row_pr in enumerate(lgbm_pr):
            top_idx = int(np.argmax(row_pr))
            top_p = float(row_pr[top_idx])
            lbl = str(lgbm_le.inverse_transform([top_idx])[0])

            if top_p >= LGBM_TAU_HIGH:
                preds_m[i] = lbl
                lgbm_fires_m += 1

            if top_p >= LGBM_TAU_LOW:
                preds_o[i] = lbl
                lgbm_fires_o += 1

        # Variant P: Soft-vote ensemble
        avg_pr, merged_classes = _align_probas(lgbm_pr, lgbm_le, ml_probas, le_xgb)
        top_idx_p = np.argmax(avg_pr, axis=1)
        preds_p = [merged_classes[i] for i in top_idx_p]

    acc_m = _acc(preds_m)
    acc_n = _acc(preds_n)
    acc_o = _acc(preds_o)
    acc_p = _acc(preds_p)

    lgbm_fire_rate_m = lgbm_fires_m / n if n > 0 else 0.0
    lgbm_fire_rate_o = lgbm_fires_o / n if n > 0 else 0.0

    logger.info(
        "[%s/%s] n=%d | B=%.3f M=%.3f N=%.3f O=%.3f P=%.3f | fires_M=%.2f fires_O=%.2f",
        cat, attr, n, acc_b, acc_m, acc_n, acc_o, acc_p, lgbm_fire_rate_m, lgbm_fire_rate_o,
    )

    base_row = dict(
        category=cat, attr=attr, n_test=n,
        n_train_gold=len(train_gold), n_silver=len(y_silver),
    )
    rows = [
        {**base_row, "variant": "B_ml_only",         "accuracy": acc_b, "lgbm_fire_rate": 0.0},
        {**base_row, "variant": "M_lightgbm_τ_ml",   "accuracy": acc_m, "lgbm_fire_rate": lgbm_fire_rate_m},
        {**base_row, "variant": "N_lightgbm_only",   "accuracy": acc_n, "lgbm_fire_rate": 1.0},
        {**base_row, "variant": "O_lightgbm_lowτ_ml","accuracy": acc_o, "lgbm_fire_rate": lgbm_fire_rate_o},
        {**base_row, "variant": "P_ensemble",        "accuracy": acc_p, "lgbm_fire_rate": float("nan")},
    ]
    return rows


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

        train_codes, test_codes = train_test_split(
            unique_codes, test_size=TEST_SIZE, random_state=SEED
        )
        train_codes_set = set(train_codes)
        test_codes_set = set(test_codes)
        logger.info("  Split: %d train codes, %d test codes", len(train_codes), len(test_codes))

        test_products = silver[silver["code"].isin(test_codes_set)].copy()
        attrs = sorted(cat_gold["attr"].unique().tolist())

        for attr in attrs:
            rows = run_one_attr(
                cat=cat, attr=attr, gold=cat_gold, silver=silver,
                emb_all=emb_all, code_to_idx=code_to_idx,
                train_codes_set=train_codes_set, test_codes_set=test_codes_set,
                test_products=test_products,
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

VARIANT_ORDER = [
    "B_ml_only",
    "M_lightgbm_τ_ml",
    "N_lightgbm_only",
    "O_lightgbm_lowτ_ml",
    "P_ensemble",
]


def _build_summary(result: pd.DataFrame) -> pd.DataFrame:
    grand = result.groupby("variant").agg(
        mean_acc=("accuracy", "mean"),
        mean_fire_rate=("lgbm_fire_rate", lambda x: x[~x.isna()].mean()),
    ).reindex(VARIANT_ORDER)
    return grand


def _print_summary(summary: pd.DataFrame, result: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("P2-EXP7: LightGBM-alone vs LightGBM+ML cascade — Grand Means")
    print("=" * 70)
    b_acc = summary.loc["B_ml_only", "mean_acc"]
    for v in VARIANT_ORDER:
        acc = summary.loc[v, "mean_acc"]
        fire = summary.loc[v, "mean_fire_rate"]
        delta = (acc - b_acc) * 100
        fire_str = f"  fires={fire*100:.1f}%" if not np.isnan(fire) else ""
        print(f"  {v:26s}: {acc*100:.2f}%  ({delta:+.2f} pp vs B){fire_str}")

    print("\n" + "=" * 70)
    print("LightGBM fires-rate at τ=0.85 per (cat, attr)")
    print("=" * 70)
    m_rows = result[result["variant"] == "M_lightgbm_τ_ml"].copy()
    for _, row in m_rows.sort_values(["category", "attr"]).iterrows():
        print(f"  [{row['category']}/{row['attr']}]  {row['lgbm_fire_rate']*100:.1f}%")

    mean_fires = m_rows["lgbm_fire_rate"].mean()
    print(f"\n  Overall mean fires-rate @τ=0.85: {mean_fires*100:.1f}%")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    acc_m = summary.loc["M_lightgbm_τ_ml", "mean_acc"] * 100
    acc_n = summary.loc["N_lightgbm_only", "mean_acc"] * 100
    delta_m_n = acc_m - acc_n
    fires = mean_fires * 100
    print(f"  M (τ=0.85+ML fallback): {acc_m:.2f}%")
    print(f"  N (LightGBM-only):      {acc_n:.2f}%  (delta M-N: {delta_m_n:+.2f} pp)")
    print(f"  Mean fires-rate @τ=0.85: {fires:.1f}%  "
          f"=> ML fallback used {100-fires:.1f}% of predictions")
    if fires >= 95:
        verdict = "ML LAYER REDUNDANT — LightGBM fires 95%+ of the time; ML fallback barely used."
    elif fires >= 80:
        verdict = "ML LAYER MARGINAL — LightGBM fires 80-95%; ML fallback provides small safety net."
    else:
        verdict = "ML LAYER MEANINGFUL — LightGBM fires <80%; ML fallback does real work."
    print(f"\n  {verdict}")
    if delta_m_n >= 1.0:
        print(f"  ML fallback adds {delta_m_n:+.2f} pp — KEEP 3-layer (LightGBM + ML + LLM).")
    elif delta_m_n >= 0.0:
        verdict2 = f"ML fallback adds {delta_m_n:+.2f} pp — borderline, keep for safety."
        print(f"  {verdict2}")
    else:
        print(f"  ML fallback hurts {delta_m_n:+.2f} pp — consider 2-layer (LightGBM + LLM).")


def _write_md(summary: pd.DataFrame, result: pd.DataFrame) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    b_acc = summary.loc["B_ml_only", "mean_acc"]

    m_rows = result[result["variant"] == "M_lightgbm_τ_ml"]
    mean_fires = m_rows["lgbm_fire_rate"].mean()

    lines = [
        "# P2-EXP7: LightGBM-Alone vs LightGBM+ML Cascade",
        "",
        "**Date:** 2026-05-17  ",
        "**Script:** `src/experiments/exp7_lightgbm_alone.py`  ",
        "**Output:** `datasets/processed/lightgbm_alone_check.parquet`  ",
        "",
        "## Setup",
        "",
        "- Same 80/20 split (seed=42) on `consensus_gold_v2_expanded.parquet`",
        "- 3 categories: pasta, chocolate, cheeses — 22 attrs",
        "- LightGBM: TF-IDF ngrams(1–2), max_features=10000",
        "- ML: fresh hybrid_v2 XGBoost (silver + gold × 5 weight)",
        "",
        "## Grand Mean Results",
        "",
        "| # | Variant | Accuracy | vs B (pp) | Notes |",
        "|---|---------|----------|-----------|-------|",
    ]

    descriptions = {
        "B_ml_only":           "hybrid_v2 XGBoost only (baseline)",
        "M_lightgbm_τ_ml":    "LightGBM @τ=0.85 → ML fallback",
        "N_lightgbm_only":    "LightGBM always predict, no fallback",
        "O_lightgbm_lowτ_ml": "LightGBM @τ=0.50 → ML fallback",
        "P_ensemble":          "Soft-vote: 0.5 × LightGBM + 0.5 × ML",
    }
    letters = {"B_ml_only": "B", "M_lightgbm_τ_ml": "M", "N_lightgbm_only": "N",
               "O_lightgbm_lowτ_ml": "O", "P_ensemble": "P"}

    for v in VARIANT_ORDER:
        acc = summary.loc[v, "mean_acc"]
        delta = (acc - b_acc) * 100
        fire = summary.loc[v, "mean_fire_rate"]
        fire_str = f"fires={fire*100:.1f}%" if not np.isnan(fire) else "—"
        lines.append(f"| {letters[v]} | {v} | {acc*100:.2f}% | {delta:+.2f} | {descriptions[v]} |")

    lines += [
        "",
        "## LightGBM Fires-Rate at τ=0.85 (per cat/attr)",
        "",
        f"**Overall mean: {mean_fires*100:.1f}%** — ML fallback used for "
        f"{(1-mean_fires)*100:.1f}% of predictions.",
        "",
        "| Category | Attr | Fires-rate |",
        "|----------|------|-----------|",
    ]
    for _, row in m_rows.sort_values(["category", "attr"]).iterrows():
        lines.append(f"| {row['category']} | {row['attr']} | {row['lgbm_fire_rate']*100:.1f}% |")

    acc_m = summary.loc["M_lightgbm_τ_ml", "mean_acc"] * 100
    acc_n = summary.loc["N_lightgbm_only", "mean_acc"] * 100
    delta_m_n = acc_m - acc_n

    if mean_fires >= 0.95:
        fires_verdict = "LightGBM fires 95%+ — ML layer effectively unused."
        arch_rec = "2-layer (LightGBM + LLM) is sufficient."
    elif mean_fires >= 0.80:
        fires_verdict = "LightGBM fires 80–95% — ML layer is a minor safety net."
        arch_rec = "Keep 3-layer for safety; ML adds marginal value."
    else:
        fires_verdict = "LightGBM fires <80% — ML layer does meaningful work."
        arch_rec = "Keep 3-layer (LightGBM + ML + LLM)."

    if delta_m_n >= 1.0:
        ml_verdict = f"ML fallback adds +{delta_m_n:.2f} pp — KEEP ML layer."
    elif delta_m_n >= 0.0:
        ml_verdict = f"ML fallback adds +{delta_m_n:.2f} pp — borderline."
    else:
        ml_verdict = f"ML fallback costs {delta_m_n:.2f} pp — consider dropping."

    lines += [
        "",
        "## Verdict",
        "",
        f"- M (τ=0.85 + ML fallback): **{acc_m:.2f}%**",
        f"- N (LightGBM-only):        **{acc_n:.2f}%**  (delta M−N: {delta_m_n:+.2f} pp)",
        f"- {fires_verdict}",
        f"- {ml_verdict}",
        f"- **Architecture recommendation:** {arch_rec}",
    ]

    OUT_MD.write_text("\n".join(lines) + "\n")
    logger.info("Wrote findings to %s", OUT_MD)


if __name__ == "__main__":
    main()
