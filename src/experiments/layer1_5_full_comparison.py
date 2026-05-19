"""P2-EXP2: Honest 80/20 comparison of 6 Layer 1.5 variants.

Variants:
  A  regex_ml    — hand-crafted regex + hybrid_v2 ML fallback  (current production)
  B  ml_only     — no Layer 1, hybrid_v2 ML directly
  C  nb_ml       — NaiveBayes (tau=0.95) + hybrid_v2 ML fallback
  D  dt_ml       — Decision Tree (tau=0.80) + hybrid_v2 ML fallback
  E  centroid_ml — nearest-centroid in embedding space (margin_tau=0.10) + ML
  F  logreg_ml   — Logistic Regression (tau=0.95) + hybrid_v2 ML fallback

Split: per-category code split, 80/20, seed=42 (same methodology as layer1_5_ablation).
Evaluation: accuracy on non-null gold cells in the 20% test split.

Output: datasets/processed/layer1_5_full_comparison.parquet
        columns: category, attr, variant, accuracy, n_test, l15_fire_rate
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import PROCESSED_DIR, setup_logging
from src.eval.cascade_predict import predict_cascade
from src.experiments.layer1_5_methods import (
    predict_centroid,
    predict_dt,
    predict_logreg,
    train_centroid,
    train_dt,
    train_logreg,
)
from src.experiments.nb_layer import load_nb_model, predict_nb

logger = logging.getLogger(__name__)

CATEGORIES = ["pasta", "chocolate", "cheeses"]
SEED = 42
TEST_SIZE = 0.2

NB_TAU = 0.95
DT_TAU = 0.80
LOGREG_TAU = 0.95
CENTROID_MARGIN = 0.05

GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet"
OUT_PATH = Path(PROCESSED_DIR) / "layer1_5_full_comparison.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_text(row: pd.Series) -> str:
    parts = []
    for col in ["product_name", "ingredients_text", "brands", "quantity"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


def _compute_accuracy(
    preds: pd.DataFrame,
    gold: pd.DataFrame,
) -> dict[str, dict]:
    """Return {attr: {accuracy, n_test}} on non-null gold cells."""
    gold = gold[~gold["gold_is_null"]].copy()
    gold["code"] = gold["code"].astype(str)
    preds = preds.copy()
    preds["code"] = preds["code"].astype(str)

    m = gold.merge(preds[["code", "attr", "predicted"]], on=["code", "attr"], how="left")
    m["correct"] = (
        m["predicted"].astype(object) == m["gold_value"].astype(object)
    ).fillna(False)

    result = {}
    for attr, grp in m.groupby("attr"):
        result[attr] = {
            "accuracy": float(grp["correct"].mean()),
            "n_test": len(grp),
        }
    return result


def _nb_fire_rate(preds: pd.DataFrame, layer_name: str = "nb") -> float:
    if len(preds) == 0:
        return 0.0
    return float((preds.get("layer", pd.Series(dtype=str)) == layer_name).sum() / len(preds))


def _l15_fire_rate(preds: pd.DataFrame) -> float:
    """Fraction of predictions from Layer 1.5 (not ML/abstain)."""
    if "layer" not in preds.columns or len(preds) == 0:
        return 0.0
    l15 = preds["layer"].isin(["nb", "dt", "centroid", "logreg"])
    return float(l15.sum() / len(preds))


# ---------------------------------------------------------------------------
# Variant A: regex + ml-hybrid (current production)
# Variant B: ml-hybrid only (no regex)
# ---------------------------------------------------------------------------

def run_variant_ab(
    products: pd.DataFrame,
    cat: str,
    include_regex: bool,
) -> pd.DataFrame:
    """Run cascade_predict with or without regex. Returns long-format preds."""
    preds = predict_cascade(
        products,
        category=f"{cat}_stratified",
        use_hybrid=True,
        include_regex=include_regex,
    )
    preds["code"] = preds["code"].astype(str)
    return preds


# ---------------------------------------------------------------------------
# Variant C: NaiveBayes + ml-hybrid
# ---------------------------------------------------------------------------

def run_variant_c(
    products: pd.DataFrame,
    cat: str,
    texts: list[str],
    codes: list[str],
    attrs: list[str],
    tau: float = NB_TAU,
) -> pd.DataFrame:
    """NB layer then ML fallback.

    Loads NB models trained during P2-EXP1 (nb_layer.py train_all).
    """
    nb_rows: list[dict] = []
    nb_fired_set: set = set()

    for attr in attrs:
        nb, vec = load_nb_model(cat, attr)
        if nb is None:
            continue
        preds = predict_nb(nb, vec, texts, tau=tau)
        for code, (label, proba) in zip(codes, preds):
            if label is not None:
                nb_rows.append({
                    "code": code, "attr": attr,
                    "predicted": label, "confidence": proba,
                    "layer": "nb",
                })
                nb_fired_set.add((code, attr))

    nb_df = pd.DataFrame(nb_rows) if nb_rows else pd.DataFrame(
        columns=["code", "attr", "predicted", "confidence", "layer"]
    )

    # ML fallback for everything NB didn't fire on
    ml_preds = predict_cascade(
        products,
        category=f"{cat}_stratified",
        use_hybrid=True,
        include_regex=False,
    )
    ml_preds["code"] = ml_preds["code"].astype(str)
    ml_fallback = ml_preds[
        ~ml_preds.apply(lambda r: (r["code"], r["attr"]) in nb_fired_set, axis=1)
    ].copy()

    return pd.concat([nb_df, ml_fallback], ignore_index=True)


# ---------------------------------------------------------------------------
# Variant D: Decision Tree + ml-hybrid
# ---------------------------------------------------------------------------

def run_variant_d(
    products: pd.DataFrame,
    cat: str,
    texts: list[str],
    codes: list[str],
    attrs: list[str],
    dt_models: dict,  # attr -> (vec, clf) | (None, None)
    tau: float = DT_TAU,
) -> pd.DataFrame:
    dt_rows: list[dict] = []
    dt_fired_set: set = set()

    for attr in attrs:
        vec, clf = dt_models.get(attr, (None, None))
        if clf is None:
            continue
        preds = predict_dt(vec, clf, texts, tau=tau)
        for code, (label, proba) in zip(codes, preds):
            if label is not None:
                dt_rows.append({
                    "code": code, "attr": attr,
                    "predicted": label, "confidence": proba,
                    "layer": "dt",
                })
                dt_fired_set.add((code, attr))

    dt_df = pd.DataFrame(dt_rows) if dt_rows else pd.DataFrame(
        columns=["code", "attr", "predicted", "confidence", "layer"]
    )

    ml_preds = predict_cascade(
        products,
        category=f"{cat}_stratified",
        use_hybrid=True,
        include_regex=False,
    )
    ml_preds["code"] = ml_preds["code"].astype(str)
    ml_fallback = ml_preds[
        ~ml_preds.apply(lambda r: (r["code"], r["attr"]) in dt_fired_set, axis=1)
    ].copy()

    return pd.concat([dt_df, ml_fallback], ignore_index=True)


# ---------------------------------------------------------------------------
# Variant E: Centroid + ml-hybrid
# ---------------------------------------------------------------------------

def run_variant_e(
    products: pd.DataFrame,
    cat: str,
    codes: list[str],
    attrs: list[str],
    test_embs: np.ndarray,           # (N, D) — test product embeddings
    centroid_models: dict,            # attr -> centroids dict | None
    margin_tau: float = CENTROID_MARGIN,
) -> pd.DataFrame:
    c_rows: list[dict] = []
    c_fired_set: set = set()

    for attr in attrs:
        centroids = centroid_models.get(attr)
        if centroids is None:
            continue
        preds = predict_centroid(centroids, test_embs, margin_tau=margin_tau)
        for code, (label, sim) in zip(codes, preds):
            if label is not None:
                c_rows.append({
                    "code": code, "attr": attr,
                    "predicted": label, "confidence": sim,
                    "layer": "centroid",
                })
                c_fired_set.add((code, attr))

    c_df = pd.DataFrame(c_rows) if c_rows else pd.DataFrame(
        columns=["code", "attr", "predicted", "confidence", "layer"]
    )

    ml_preds = predict_cascade(
        products,
        category=f"{cat}_stratified",
        use_hybrid=True,
        include_regex=False,
    )
    ml_preds["code"] = ml_preds["code"].astype(str)
    ml_fallback = ml_preds[
        ~ml_preds.apply(lambda r: (r["code"], r["attr"]) in c_fired_set, axis=1)
    ].copy()

    return pd.concat([c_df, ml_fallback], ignore_index=True)


# ---------------------------------------------------------------------------
# Variant F: LogReg + ml-hybrid
# ---------------------------------------------------------------------------

def run_variant_f(
    products: pd.DataFrame,
    cat: str,
    texts: list[str],
    codes: list[str],
    attrs: list[str],
    logreg_models: dict,  # attr -> (vec, clf) | (None, None)
    tau: float = LOGREG_TAU,
) -> pd.DataFrame:
    lr_rows: list[dict] = []
    lr_fired_set: set = set()

    for attr in attrs:
        vec, clf = logreg_models.get(attr, (None, None))
        if clf is None:
            continue
        preds = predict_logreg(vec, clf, texts, tau=tau)
        for code, (label, proba) in zip(codes, preds):
            if label is not None:
                lr_rows.append({
                    "code": code, "attr": attr,
                    "predicted": label, "confidence": proba,
                    "layer": "logreg",
                })
                lr_fired_set.add((code, attr))

    lr_df = pd.DataFrame(lr_rows) if lr_rows else pd.DataFrame(
        columns=["code", "attr", "predicted", "confidence", "layer"]
    )

    ml_preds = predict_cascade(
        products,
        category=f"{cat}_stratified",
        use_hybrid=True,
        include_regex=False,
    )
    ml_preds["code"] = ml_preds["code"].astype(str)
    ml_fallback = ml_preds[
        ~ml_preds.apply(lambda r: (r["code"], r["attr"]) in lr_fired_set, axis=1)
    ].copy()

    return pd.concat([lr_df, ml_fallback], ignore_index=True)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)

    logger.info(
        "Expanded gold: %d rows, %d unique codes",
        len(gold), gold["code"].nunique(),
    )

    all_rows: list[dict] = []

    for cat in CATEGORIES:
        logger.info("=== Category: %s ===", cat)

        cat_gold = gold[gold["category"] == cat].copy()
        unique_codes = sorted(cat_gold["code"].unique().tolist())

        # ---------- identical split to layer1_5_ablation ----------
        train_codes, test_codes = train_test_split(
            unique_codes, test_size=TEST_SIZE, random_state=SEED
        )
        train_codes_set = set(train_codes)
        test_codes_set = set(test_codes)
        logger.info("  %d train codes, %d test codes", len(train_codes), len(test_codes))

        test_gold = cat_gold[cat_gold["code"].isin(test_codes_set)].copy()
        test_gold["category"] = cat

        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        )
        silver["code"] = silver["code"].astype(str)

        emb_all = np.load(Path(PROCESSED_DIR) / f"{cat}_stratified_embeddings.npy")
        code_to_idx = {c: i for i, c in enumerate(silver["code"].tolist())}

        train_products = silver[silver["code"].isin(train_codes_set)].copy()
        test_products = silver[silver["code"].isin(test_codes_set)].copy()

        if len(test_products) == 0:
            logger.warning("  No test products for %s — skip", cat)
            continue

        logger.info("  %d train products, %d test products", len(train_products), len(test_products))

        attrs = sorted(cat_gold["attr"].unique().tolist())
        logger.info("  Attrs: %s", attrs)

        # ---------- build text features for L1.5 methods ----------
        # Training texts (from train silver)
        train_texts = [_build_text(row) for _, row in train_products.iterrows()]
        train_codes_list = train_products["code"].tolist()

        # Test texts and embeddings
        test_texts = [_build_text(row) for _, row in test_products.iterrows()]
        test_codes_list = test_products["code"].tolist()

        # Test embeddings from cached .npy (by code index)
        test_idx_list = [code_to_idx[c] for c in test_codes_list if c in code_to_idx]
        test_codes_with_emb = [c for c in test_codes_list if c in code_to_idx]
        test_embs = emb_all[test_idx_list]  # (N_test, 384)

        # ---------- Train DT / Centroid / LogReg per attr ----------
        dt_models: dict = {}
        centroid_models: dict = {}
        logreg_models: dict = {}

        for attr in attrs:
            # Training labels from gold (non-null, train split)
            train_attr_gold = cat_gold[
                (cat_gold["attr"] == attr)
                & ~cat_gold["gold_is_null"]
                & cat_gold["code"].isin(train_codes_set)
            ].copy()

            # Join product text from silver
            train_attr_gold = train_attr_gold.merge(
                train_products[["code", "product_name", "ingredients_text", "brands", "quantity"]],
                on="code", how="inner",
            )

            if len(train_attr_gold) == 0:
                dt_models[attr] = (None, None)
                centroid_models[attr] = None
                logreg_models[attr] = (None, None)
                continue

            X_texts = [_build_text(row) for _, row in train_attr_gold.iterrows()]
            y_labels = train_attr_gold["gold_value"].astype(str).tolist()

            # Build embeddings for centroid training
            train_attr_codes = train_attr_gold["code"].tolist()
            train_attr_idx = [code_to_idx[c] for c in train_attr_codes if c in code_to_idx]
            train_attr_codes_with_emb = [c for c in train_attr_codes if c in code_to_idx]

            # Align labels to codes with embeddings
            code_label_map = dict(zip(train_attr_gold["code"], y_labels))
            y_for_emb = [code_label_map[c] for c in train_attr_codes_with_emb]

            # Train methods
            dt_models[attr] = train_dt(X_texts, y_labels)

            if len(train_attr_idx) > 0 and len(y_for_emb) >= 10 and len(set(y_for_emb)) >= 2:
                X_emb = emb_all[train_attr_idx]
                centroid_models[attr] = train_centroid(X_emb, y_for_emb)
            else:
                centroid_models[attr] = None

            logreg_models[attr] = train_logreg(X_texts, y_labels)

            logger.debug(
                "  [%s/%s] trained: DT=%s, Centroid=%s, LogReg=%s  (n_train=%d)",
                cat, attr,
                "ok" if dt_models[attr][1] is not None else "skip",
                "ok" if centroid_models[attr] is not None else "skip",
                "ok" if logreg_models[attr][1] is not None else "skip",
                len(y_labels),
            )

        # ---------- Run all 6 variants on test products ----------

        def record_variant(
            variant_name: str,
            preds: pd.DataFrame,
            fire_rate: float = 0.0,
        ) -> None:
            acc_map = _compute_accuracy(preds, test_gold)
            for attr, stats in acc_map.items():
                all_rows.append({
                    "category": cat,
                    "attr": attr,
                    "variant": variant_name,
                    "accuracy": stats["accuracy"],
                    "n_test": stats["n_test"],
                    "l15_fire_rate": fire_rate,
                })
            logger.info(
                "  [%s/%s] mean_acc=%.4f",
                cat, variant_name,
                np.nanmean([s["accuracy"] for s in acc_map.values()]) if acc_map else float("nan"),
            )

        # A: regex + ml-hybrid
        logger.info("  Variant A: regex_ml")
        preds_a = run_variant_ab(test_products, cat, include_regex=True)
        preds_a["category"] = cat
        record_variant("A_regex_ml", preds_a, fire_rate=0.0)

        # B: ml only
        logger.info("  Variant B: ml_only")
        preds_b = run_variant_ab(test_products, cat, include_regex=False)
        preds_b["category"] = cat
        record_variant("B_ml_only", preds_b, fire_rate=0.0)

        # C: NB + ml
        logger.info("  Variant C: nb_ml (tau=%.2f)", NB_TAU)
        preds_c = run_variant_c(
            test_products, cat, test_texts, test_codes_list, attrs, tau=NB_TAU
        )
        preds_c["category"] = cat
        record_variant("C_nb_ml", preds_c, fire_rate=_l15_fire_rate(preds_c))

        # D: DT + ml
        logger.info("  Variant D: dt_ml (tau=%.2f)", DT_TAU)
        preds_d = run_variant_d(
            test_products, cat, test_texts, test_codes_list, attrs, dt_models, tau=DT_TAU
        )
        preds_d["category"] = cat
        record_variant("D_dt_ml", preds_d, fire_rate=_l15_fire_rate(preds_d))

        # E: Centroid + ml
        logger.info("  Variant E: centroid_ml (margin=%.2f)", CENTROID_MARGIN)
        preds_e = run_variant_e(
            test_products, cat, test_codes_with_emb, attrs, test_embs,
            centroid_models, margin_tau=CENTROID_MARGIN,
        )
        preds_e["category"] = cat
        record_variant("E_centroid_ml", preds_e, fire_rate=_l15_fire_rate(preds_e))

        # F: LogReg + ml
        logger.info("  Variant F: logreg_ml (tau=%.2f)", LOGREG_TAU)
        preds_f = run_variant_f(
            test_products, cat, test_texts, test_codes_list, attrs, logreg_models, tau=LOGREG_TAU
        )
        preds_f["category"] = cat
        record_variant("F_logreg_ml", preds_f, fire_rate=_l15_fire_rate(preds_f))

    # ---------- Save and print summary ----------
    result = pd.DataFrame(all_rows)
    result.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(result), OUT_PATH)

    _print_summary(result)


def _print_summary(result: pd.DataFrame) -> None:
    variants = ["A_regex_ml", "B_ml_only", "C_nb_ml", "D_dt_ml", "E_centroid_ml", "F_logreg_ml"]

    print("\n" + "=" * 70)
    print("P2-EXP2: Layer 1.5 Full Comparison — Grand Means")
    print("=" * 70)
    grand = result.groupby("variant")["accuracy"].mean()
    for v in variants:
        if v in grand.index:
            print(f"  {v:20s}: {grand[v] * 100:.2f}%")

    print("\n" + "=" * 70)
    print("Per-category mean accuracy")
    print("=" * 70)
    pivot_cat = result.pivot_table(
        index="category", columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [v for v in variants if v in pivot_cat.columns]
    print(pivot_cat[cols].round(4).to_string())

    print("\n" + "=" * 70)
    print("Per-(category, attr) — winner per row")
    print("=" * 70)
    pivot_attr = result.pivot_table(
        index=["category", "attr"], columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [v for v in variants if v in pivot_attr.columns]
    pivot_attr = pivot_attr[cols]
    pivot_attr["winner"] = pivot_attr.idxmax(axis=1)
    print(pivot_attr.round(4).to_string())

    print("\n" + "=" * 70)
    print("Problematic attrs: A vs B vs C vs D vs E vs F")
    print("=" * 70)
    problematic = [
        ("chocolate", "cocoa_percentage"),
        ("chocolate", "chocolate_type"),
        ("pasta", "grain_type"),
        ("pasta", "pasta_shape"),
    ]
    for p_cat, p_attr in problematic:
        row_vals = []
        sub = result[(result["category"] == p_cat) & (result["attr"] == p_attr)]
        for v in variants:
            v_sub = sub[sub["variant"] == v]["accuracy"]
            val = v_sub.values[0] * 100 if len(v_sub) > 0 else float("nan")
            row_vals.append(f"{v.split('_')[0]}={val:.1f}%")
        print(f"  {p_cat}/{p_attr}: " + "  ".join(row_vals))

    print("\n" + "=" * 70)
    print("Winner counts (how many attrs each variant wins)")
    print("=" * 70)
    winner_col = pivot_attr["winner"]
    winner_counts = winner_col.value_counts()
    for v in variants:
        cnt = winner_counts.get(v, 0)
        print(f"  {v:20s}: {cnt} attrs")


if __name__ == "__main__":
    main()
