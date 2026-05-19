"""P2-EXP3: Layer 1.5 honest comparison — fresh-train hybrid on 80%, evaluate on 20%.

Fixes the leakage in P2-EXP2 (layer1_5_full_comparison.py) where the production
hybrid_v2 model was trained on the full silver which overlaps with the test codes.

Here we fresh-train the hybrid ML on the SAME 80% split used to train the
Layer-1.5 methods (NB, DT, Centroid, LogReg), then evaluate all 6 variants
on the same 20% held-out test set.

Variants:
  A  regex_ml    — RegexExtractor + fresh hybrid_ml fallback
  B  ml_only     — fresh hybrid_ml directly (no regex)
  C  nb_ml       — NaiveBayes (tau=0.95) + hybrid_ml fallback
  D  dt_ml       — DT (tau=0.80) + hybrid_ml fallback
  E  centroid_ml — nearest-centroid (margin_gap=0.10) + hybrid_ml fallback
  F  logreg_ml   — LogReg (tau=0.95) + hybrid_ml fallback

Output: datasets/processed/layer1_5_honest_comparison.parquet
        docs/layer1_5_honest_comparison.md
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.common import PROCESSED_DIR, setup_logging
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
MIN_GOLD = 20  # skip attr if fewer non-null gold cells

# Layer-1.5 thresholds
NB_TAU = 0.95
DT_TAU = 0.80
LOGREG_TAU = 0.95
CENTROID_MARGIN = 0.10

GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet"
OUT_PATH = Path(PROCESSED_DIR) / "layer1_5_honest_comparison.parquet"
DOC_PATH = Path("docs") / "layer1_5_honest_comparison.md"


# ---------------------------------------------------------------------------
# Text builder (shared with nb_layer.py)
# ---------------------------------------------------------------------------

def _build_text(row: pd.Series) -> str:
    parts = []
    for col in ["product_name", "ingredients_text", "brands", "quantity"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Fresh hybrid XGBoost trainer — returns (clf, le) for inference
# ---------------------------------------------------------------------------

def _train_fresh_hybrid(
    X_silver: np.ndarray,
    y_silver: np.ndarray,
    X_gold: np.ndarray,
    y_gold: np.ndarray,
    gold_weight: float = 5.0,
) -> tuple[Optional[xgb.XGBClassifier], Optional[LabelEncoder]]:
    """Train XGB on (silver + gold-weighted) data. Returns (clf, le) for inference."""
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


def _hybrid_predict_label(
    clf: xgb.XGBClassifier,
    le: LabelEncoder,
    X: np.ndarray,
) -> list[str]:
    """Return predicted string labels for rows in X."""
    enc_preds = clf.predict(X)
    return le.inverse_transform(enc_preds).tolist()


# ---------------------------------------------------------------------------
# Regex helper — returns {code: {attr: value}} for test products
# ---------------------------------------------------------------------------

def _build_regex_preds(
    test_products: pd.DataFrame,
    domain: str,
) -> dict[str, dict[str, str]]:
    """Run regex extractor once per test product."""
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
    regex_preds: dict[str, dict[str, str]],
    test_products: pd.DataFrame,
) -> list[dict]:
    """Run all 6 variants for one (cat, attr) pair. Returns list of result rows."""
    # ---------------------------------------------------------------
    # Filter gold for this attr — non-null only
    # ---------------------------------------------------------------
    cat_gold = gold[(gold["category"] == cat) & (gold["attr"] == attr) & ~gold["gold_is_null"]].copy()
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

    # ---------------------------------------------------------------
    # Build arrays
    # ---------------------------------------------------------------
    train_idx = np.array([code_to_idx[c] for c in train_gold["code"]])
    test_idx = np.array([code_to_idx[c] for c in test_gold["code"]])

    X_gold_train = emb_all[train_idx]
    y_gold_train = train_gold["gold_value"].astype(str).values
    X_test_emb = emb_all[test_idx]
    y_test = test_gold["gold_value"].astype(str).values
    test_codes_list = test_gold["code"].tolist()

    # ---------------------------------------------------------------
    # Build silver training data (exclude test codes)
    # ---------------------------------------------------------------
    X_silver: np.ndarray = np.empty((0, emb_all.shape[1]))
    y_silver: np.ndarray = np.array([], dtype=str)

    if attr in silver.columns:
        silver_attr = silver[silver[attr].notna()].copy()
        silver_attr["code"] = silver_attr["code"].astype(str)
        # Exclude test-set codes to prevent leakage
        silver_attr = silver_attr[~silver_attr["code"].isin(test_codes_set)]
        # Also exclude train_gold codes to avoid double-counting (gold is weighted)
        silver_attr = silver_attr[~silver_attr["code"].isin(train_codes_set)]
        silver_attr = silver_attr[silver_attr["code"].isin(code_to_idx)]

        silver_idx = np.array([code_to_idx[c] for c in silver_attr["code"]])
        if len(silver_idx) > 0:
            X_silver = emb_all[silver_idx]
            y_silver = silver_attr[attr].astype(str).values

    # ---------------------------------------------------------------
    # Train fresh hybrid ML
    # ---------------------------------------------------------------
    if len(X_silver) > 0:
        clf, le = _train_fresh_hybrid(X_silver, y_silver, X_gold_train, y_gold_train)
    else:
        # No silver data for attr — train gold-only XGB
        all_classes = sorted(set(y_gold_train.tolist()))
        if len(all_classes) < 2:
            logger.info("[%s/%s] degenerate classes in gold, skipping", cat, attr)
            return []
        le = LabelEncoder()
        le.fit(all_classes)
        y_enc = le.transform(y_gold_train)
        n_classes = len(all_classes)
        kwargs = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
                      tree_method="hist", verbosity=0)
        if n_classes == 2:
            pos = int((y_enc == 1).sum())
            neg = int((y_enc == 0).sum())
            kwargs["scale_pos_weight"] = max(neg / max(pos, 1), 0.5)
            clf = xgb.XGBClassifier(**kwargs)
        else:
            clf = xgb.XGBClassifier(objective="multi:softmax", num_class=n_classes, **kwargs)
        clf.fit(X_gold_train, y_enc)

    if clf is None or le is None:
        logger.info("[%s/%s] could not train hybrid ML, skipping", cat, attr)
        return []

    # Test classes that are not in training — these will map to nearest le class
    # We handle this by checking unseen labels
    train_labels_set = set(le.classes_)

    # ---------------------------------------------------------------
    # Build text features for Layer-1.5 methods
    # ---------------------------------------------------------------
    # Training texts: from gold train rows joined with silver product text
    train_gold_with_text = train_gold.merge(
        silver[["code", "product_name", "ingredients_text", "brands", "quantity"]],
        on="code", how="left",
    )
    train_texts = [_build_text(row) for _, row in train_gold_with_text.iterrows()]
    y_train_texts = train_gold_with_text["gold_value"].astype(str).tolist()

    # Test texts: from test_products aligned to test_gold codes
    test_products_idx = test_products.set_index("code")
    test_texts_list = []
    for c in test_codes_list:
        if c in test_products_idx.index:
            test_texts_list.append(_build_text(test_products_idx.loc[c]))
        else:
            test_texts_list.append("")

    # ---------------------------------------------------------------
    # Train Layer-1.5 methods (on 80% gold only)
    # ---------------------------------------------------------------
    nb_clf, nb_vec = train_nb_for_attr(
        cat, attr,
        gold=gold[(gold["category"] == cat) & (gold["attr"] == attr)],
        silver=silver,
        train_codes=list(train_codes_set),
    )
    dt_vec, dt_clf = train_dt(train_texts, y_train_texts)
    centroid_dict = train_centroid(X_gold_train, y_gold_train.tolist())
    lr_vec, lr_clf = train_logreg(train_texts, y_train_texts)

    # ---------------------------------------------------------------
    # Helper: hybrid ML fallback prediction per sample
    # ---------------------------------------------------------------
    def ml_pred_for_row(i: int) -> str:
        """Return hybrid ML label for test sample i (by emb index)."""
        x = X_test_emb[i:i+1]
        enc = clf.predict(x)[0]
        return le.inverse_transform([enc])[0]

    def ml_preds_all() -> list[str]:
        enc_preds = clf.predict(X_test_emb)
        return le.inverse_transform(enc_preds).tolist()

    # ---------------------------------------------------------------
    # Variant A: regex → hybrid_ml fallback
    # ---------------------------------------------------------------
    ml_labels_all = ml_preds_all()
    preds_a = []
    for i, code in enumerate(test_codes_list):
        regex_val = regex_preds.get(code, {}).get(attr)
        if regex_val is not None:
            preds_a.append(str(regex_val))
        else:
            preds_a.append(ml_labels_all[i])

    # ---------------------------------------------------------------
    # Variant B: hybrid_ml only
    # ---------------------------------------------------------------
    preds_b = ml_labels_all[:]

    # ---------------------------------------------------------------
    # Variant C: NB@tau + hybrid_ml fallback
    # ---------------------------------------------------------------
    preds_c = []
    if nb_clf is not None and nb_vec is not None:
        nb_results = predict_nb(nb_clf, nb_vec, test_texts_list, tau=NB_TAU)
        for i, (label, _) in enumerate(nb_results):
            if label is not None:
                preds_c.append(str(label))
            else:
                preds_c.append(ml_labels_all[i])
    else:
        preds_c = ml_labels_all[:]

    # ---------------------------------------------------------------
    # Variant D: DT@tau + hybrid_ml fallback
    # ---------------------------------------------------------------
    preds_d = []
    if dt_vec is not None and dt_clf is not None:
        dt_results = predict_dt(dt_vec, dt_clf, test_texts_list, tau=DT_TAU)
        for i, (label, _) in enumerate(dt_results):
            if label is not None:
                preds_d.append(str(label))
            else:
                preds_d.append(ml_labels_all[i])
    else:
        preds_d = ml_labels_all[:]

    # ---------------------------------------------------------------
    # Variant E: Centroid + hybrid_ml fallback
    # ---------------------------------------------------------------
    preds_e = []
    if centroid_dict is not None:
        c_results = predict_centroid(centroid_dict, X_test_emb, margin_tau=CENTROID_MARGIN)
        for i, (label, _) in enumerate(c_results):
            if label is not None:
                preds_e.append(str(label))
            else:
                preds_e.append(ml_labels_all[i])
    else:
        preds_e = ml_labels_all[:]

    # ---------------------------------------------------------------
    # Variant F: LogReg@tau + hybrid_ml fallback
    # ---------------------------------------------------------------
    preds_f = []
    if lr_vec is not None and lr_clf is not None:
        lr_results = predict_logreg(lr_vec, lr_clf, test_texts_list, tau=LOGREG_TAU)
        for i, (label, _) in enumerate(lr_results):
            if label is not None:
                preds_f.append(str(label))
            else:
                preds_f.append(ml_labels_all[i])
    else:
        preds_f = ml_labels_all[:]

    # ---------------------------------------------------------------
    # Compute accuracies
    # ---------------------------------------------------------------
    n = len(y_test)
    variant_preds = {
        "A_regex_ml": preds_a,
        "B_ml_only": preds_b,
        "C_nb_ml": preds_c,
        "D_dt_ml": preds_d,
        "E_centroid_ml": preds_e,
        "F_logreg_ml": preds_f,
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
        })

    logger.info(
        "[%s/%s] n_test=%d | A=%.3f B=%.3f C=%.3f D=%.3f E=%.3f F=%.3f",
        cat, attr, n,
        *[rows[i]["accuracy"] for i in range(6)],
    )
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

        # Same 80/20 split as eval_v2_expanded.py (seed=42, per-category codes)
        train_codes, test_codes = train_test_split(
            unique_codes, test_size=TEST_SIZE, random_state=SEED
        )
        train_codes_set = set(train_codes)
        test_codes_set = set(test_codes)
        logger.info(
            "  Split: %d train codes, %d test codes",
            len(train_codes), len(test_codes),
        )

        # Build test products subset from silver (for text features + regex)
        test_products = silver[silver["code"].isin(test_codes_set)].copy()
        domain = cat  # e.g. "pasta"

        # Pre-compute regex preds once for all attrs (test products only)
        logger.info("  Building regex predictions for test products...")
        regex_preds = _build_regex_preds(test_products, domain)
        regex_hits = sum(len(v) for v in regex_preds.values())
        logger.info("  Regex hit %d (code, attr) pairs", regex_hits)

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
                regex_preds=regex_preds,
                test_products=test_products,
            )
            all_rows.extend(rows)

    result = pd.DataFrame(all_rows)
    result.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(result), OUT_PATH)

    _print_summary(result)
    _write_doc(result)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

VARIANT_ORDER = [
    "A_regex_ml", "B_ml_only", "C_nb_ml", "D_dt_ml", "E_centroid_ml", "F_logreg_ml"
]


def _print_summary(result: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("P2-EXP3: Layer 1.5 Honest Comparison — Grand Means")
    print("=" * 70)
    grand = result.groupby("variant")["accuracy"].mean()
    for v in VARIANT_ORDER:
        if v in grand.index:
            print(f"  {v:22s}: {grand[v] * 100:.2f}%")

    print("\nProduction lift (vs B_ml_only):")
    b_mean = grand.get("B_ml_only", float("nan"))
    for v in VARIANT_ORDER:
        if v in grand.index and v != "B_ml_only":
            delta = (grand[v] - b_mean) * 100
            print(f"  {v:22s}: {delta:+.2f} pp")

    print("\n" + "=" * 70)
    print("Per-category mean accuracy")
    print("=" * 70)
    pivot_cat = result.pivot_table(
        index="category", columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [v for v in VARIANT_ORDER if v in pivot_cat.columns]
    print(pivot_cat[cols].round(4).to_string())

    print("\n" + "=" * 70)
    print("Per-(category, attr) accuracy + winner")
    print("=" * 70)
    pivot_attr = result.pivot_table(
        index=["category", "attr"], columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [v for v in VARIANT_ORDER if v in pivot_attr.columns]
    pivot_attr = pivot_attr[cols]
    pivot_attr["winner"] = pivot_attr.idxmax(axis=1)
    print(pivot_attr.round(4).to_string())

    print("\n" + "=" * 70)
    print("Winner counts")
    print("=" * 70)
    winner_counts = pivot_attr["winner"].value_counts()
    for v in VARIANT_ORDER:
        cnt = winner_counts.get(v, 0)
        print(f"  {v:22s}: {cnt} attrs")


# ---------------------------------------------------------------------------
# Markdown doc writer
# ---------------------------------------------------------------------------

def _write_doc(result: pd.DataFrame) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# P2-EXP3: Layer 1.5 Honest Comparison — Fresh-Train Hybrid")
    lines.append("")
    lines.append("**Methodology:** Each variant is evaluated on the same 20% held-out test split")
    lines.append("(seed=42, per-category codes). The hybrid ML is fresh-trained on full silver")
    lines.append("(excluding test codes) + 80% gold (5× weight) for every (cat, attr). Layer-1.5")
    lines.append("methods (NB, DT, Centroid, LogReg) are also trained on 80% gold only.")
    lines.append("This eliminates the train==test leakage present in P2-EXP2.")
    lines.append("")

    # Grand means
    grand = result.groupby("variant")["accuracy"].mean()
    b_mean = grand.get("B_ml_only", float("nan"))

    lines.append("## Grand Mean Accuracy per Variant")
    lines.append("")
    lines.append("| Variant | Accuracy | vs B (pp) |")
    lines.append("|---------|----------|-----------|")
    for v in VARIANT_ORDER:
        if v in grand.index:
            acc = grand[v] * 100
            delta = (grand[v] - b_mean) * 100
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {v} | {acc:.2f}% | {sign}{delta:.2f} |")
    lines.append("")

    # Per-cat
    pivot_cat = result.pivot_table(
        index="category", columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [v for v in VARIANT_ORDER if v in pivot_cat.columns]
    pivot_cat = pivot_cat[cols].round(4)

    lines.append("## Per-Category Mean Accuracy")
    lines.append("")
    header = "| Category | " + " | ".join(cols) + " |"
    sep = "|----------|" + "---------|" * len(cols)
    lines.append(header)
    lines.append(sep)
    for cat, row in pivot_cat.iterrows():
        vals = " | ".join(f"{v*100:.2f}%" for v in row)
        lines.append(f"| {cat} | {vals} |")
    lines.append("")

    # Per-(cat, attr) pivot
    pivot_attr = result.pivot_table(
        index=["category", "attr"], columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [v for v in VARIANT_ORDER if v in pivot_attr.columns]
    pivot_attr = pivot_attr[cols].round(4)
    pivot_attr["winner"] = pivot_attr.idxmax(axis=1)

    lines.append("## Per-(Category, Attr) Accuracy")
    lines.append("")
    header2 = "| Category | Attr | " + " | ".join(cols) + " | Winner |"
    sep2 = "|----------|------|" + "---------|" * len(cols) + "--------|"
    lines.append(header2)
    lines.append(sep2)
    for (cat, attr), row in pivot_attr.iterrows():
        acc_vals = " | ".join(f"{v*100:.2f}%" for v in row[cols])
        winner = row["winner"]
        lines.append(f"| {cat} | {attr} | {acc_vals} | {winner} |")
    lines.append("")

    # Real production lift table
    lines.append("## Real Production Lift Table")
    lines.append("")
    lines.append("Comparison vs **B_ml_only** (honest ML baseline):")
    lines.append("")
    lines.append("| Category | Attr | B (%) | A–B (pp) | C–B (pp) | D–B (pp) | E–B (pp) | F–B (pp) |")
    lines.append("|----------|------|-------|----------|----------|----------|----------|----------|")
    b_vals = result[result["variant"] == "B_ml_only"].set_index(["category", "attr"])["accuracy"]
    for (cat, attr), row in pivot_attr.iterrows():
        b = b_vals.get((cat, attr), float("nan"))
        deltas = {}
        for v in ["A_regex_ml", "C_nb_ml", "D_dt_ml", "E_centroid_ml", "F_logreg_ml"]:
            if v in row.index:
                deltas[v] = (row[v] - b) * 100
            else:
                deltas[v] = float("nan")
        def fmt(d):
            if d != d:  # NaN
                return "N/A"
            return f"{'+' if d >= 0 else ''}{d:.1f}"
        lines.append(
            f"| {cat} | {attr} | {b*100:.2f}% "
            f"| {fmt(deltas['A_regex_ml'])} "
            f"| {fmt(deltas['C_nb_ml'])} "
            f"| {fmt(deltas['D_dt_ml'])} "
            f"| {fmt(deltas['E_centroid_ml'])} "
            f"| {fmt(deltas['F_logreg_ml'])} |"
        )
    lines.append("")

    # Winner counts
    winner_counts = pivot_attr["winner"].value_counts()
    lines.append("## Winner Counts (best variant per attr)")
    lines.append("")
    lines.append("| Variant | # Attrs Won |")
    lines.append("|---------|-------------|")
    for v in VARIANT_ORDER:
        cnt = winner_counts.get(v, 0)
        lines.append(f"| {v} | {cnt} |")
    lines.append("")

    # Production recommendation
    best_variant = grand.idxmax() if len(grand) > 0 else "unknown"
    lines.append("## Production Recommendation")
    lines.append("")
    lines.append(
        f"Based on grand mean accuracy on the fresh honest evaluation, "
        f"**{best_variant}** achieves the highest overall accuracy ({grand.get(best_variant, 0)*100:.2f}%)."
    )
    lines.append("")
    lines.append("Key findings:")
    a_over_b = (grand.get("A_regex_ml", 0) - b_mean) * 100
    c_over_b = (grand.get("C_nb_ml", 0) - b_mean) * 100
    f_over_b = (grand.get("F_logreg_ml", 0) - b_mean) * 100
    lines.append(f"- Regex adds **{a_over_b:+.2f} pp** over ML-only (A vs B)")
    lines.append(f"- NB adds **{c_over_b:+.2f} pp** over ML-only (C vs B)")
    lines.append(f"- LogReg adds **{f_over_b:+.2f} pp** over ML-only (F vs B)")
    lines.append("- Centroid method is embedding-based; useful for zero-shot new classes")
    lines.append("")
    lines.append(
        "**Recommendation:** Deploy the variant with the highest positive lift AND "
        "simplest implementation. If regex wins, it provides interpretable rules at "
        "zero training cost. If a learned L1.5 method wins, it should be pre-trained "
        "offline and cached per (cat, attr)."
    )

    doc_text = "\n".join(lines) + "\n"
    DOC_PATH.write_text(doc_text, encoding="utf-8")
    logger.info("Wrote doc to %s", DOC_PATH)


if __name__ == "__main__":
    main()
