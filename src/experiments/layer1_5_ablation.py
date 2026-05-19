"""Layer 1.5 Ablation: compare 4 variants on honest 80/20 split.

Variants:
  A. regex + ml-hybrid    — current production (with regex override)
  B. ml-hybrid only       — no Layer 1, only ML
  C. nb_rules + ml-hybrid — NaiveBayes "auto-regex" overrides ML when NB conf >= tau
  D. nb_rules only        — no ML, only NB

For each variant × cat × attr: accuracy on held-out 20% non-null gold cells.
Also sweeps tau ∈ {0.7, 0.8, 0.85, 0.9, 0.95} for variant C.

Output: datasets/processed/layer1_5_ablation.parquet
        columns: cat, attr, variant, tau, accuracy, n_test, nb_fire_rate

Run:
    python -m src.experiments.layer1_5_ablation
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import PROCESSED_DIR, setup_logging
from src.eval.cascade_predict import predict_cascade
from src.eval.predict_with_nb_layer import predict_cascade_with_nb
from src.experiments.nb_layer import train_all

logger = logging.getLogger(__name__)

CATEGORIES = ["pasta", "chocolate", "cheeses"]
SEED = 42
TEST_SIZE = 0.2
TAU_SWEEP = [0.70, 0.80, 0.85, 0.90, 0.95]
DEFAULT_TAU = 0.85


def _compute_accuracy(preds: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Per-(category, attr) accuracy on non-null gold cells."""
    gold = gold[~gold["gold_is_null"]].copy()
    gold["code"] = gold["code"].astype(str)
    preds = preds.copy()
    preds["code"] = preds["code"].astype(str)

    m = gold.merge(preds[["code", "attr", "predicted"]], on=["code", "attr"], how="left")
    m["correct"] = (
        m["predicted"].astype(object) == m["gold_value"].astype(object)
    ).fillna(False)

    return (
        m.groupby(["category", "attr"])
        .agg(n_test=("correct", "count"), n_correct=("correct", "sum"))
        .assign(accuracy=lambda x: x["n_correct"] / x["n_test"])
        .reset_index()
    )


def _nb_fire_rate(preds: pd.DataFrame) -> float:
    """Fraction of predictions made by NB (layer == 'nb')."""
    if len(preds) == 0:
        return 0.0
    return float((preds["layer"] == "nb").sum() / len(preds))


def main() -> None:
    setup_logging()

    gold = pd.read_parquet(Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet")
    gold["code"] = gold["code"].astype(str)

    # Step 0: train NB models (on 80% gold per cat)
    logger.info("Training NB models on 80%% gold split...")
    train_all()
    logger.info("NB training done.")

    all_rows: list[dict] = []

    for cat in CATEGORIES:
        logger.info("=== Category: %s ===", cat)
        cat_gold = gold[gold["category"] == cat].copy()
        unique_codes = sorted(cat_gold["code"].unique().tolist())

        # Identical split to regex_ablation.py and eval_v2_expanded.py
        train_codes, test_codes = train_test_split(
            unique_codes, test_size=TEST_SIZE, random_state=SEED
        )
        test_codes_set = set(test_codes)
        test_gold = cat_gold[cat_gold["code"].isin(test_codes_set)].copy()
        test_gold["category"] = cat

        logger.info("  %d train codes, %d test codes", len(train_codes), len(test_codes))

        # Load product rows from silver standard
        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        )
        silver["code"] = silver["code"].astype(str)
        products = silver[silver["code"].isin(test_codes_set)].copy()
        if len(products) == 0:
            logger.warning("  No products for %s test codes", cat)
            continue

        logger.info("  %d test products loaded from silver", len(products))

        # ------------------------------------------------------------------ #
        # Variant A: regex + ml-hybrid (current production)
        # ------------------------------------------------------------------ #
        logger.info("  Variant A: regex + ml-hybrid")
        preds_a = predict_cascade(
            products, category=f"{cat}_stratified",
            use_hybrid=True, include_regex=True,
        )
        preds_a["category"] = cat
        acc_a = _compute_accuracy(preds_a, test_gold)
        for _, row in acc_a.iterrows():
            all_rows.append({
                "category": cat, "attr": row["attr"],
                "variant": "A_regex_ml",
                "tau": None,
                "accuracy": row["accuracy"],
                "n_test": row["n_test"],
                "nb_fire_rate": 0.0,
            })

        # ------------------------------------------------------------------ #
        # Variant B: ml-hybrid only (no regex)
        # ------------------------------------------------------------------ #
        logger.info("  Variant B: ml-hybrid only")
        preds_b = predict_cascade(
            products, category=f"{cat}_stratified",
            use_hybrid=True, include_regex=False,
        )
        preds_b["category"] = cat
        acc_b = _compute_accuracy(preds_b, test_gold)
        for _, row in acc_b.iterrows():
            all_rows.append({
                "category": cat, "attr": row["attr"],
                "variant": "B_ml_only",
                "tau": None,
                "accuracy": row["accuracy"],
                "n_test": row["n_test"],
                "nb_fire_rate": 0.0,
            })

        # ------------------------------------------------------------------ #
        # Variant C: nb + ml-hybrid, tau sweep
        # ------------------------------------------------------------------ #
        for tau in TAU_SWEEP:
            logger.info("  Variant C: nb + ml-hybrid (tau=%.2f)", tau)
            preds_c = predict_cascade_with_nb(
                products, cat,
                nb_threshold=tau,
                use_hybrid=True,
                include_regex=False,  # NB replaces regex; no double-overrides
            )
            preds_c["category"] = cat
            fire_rate = _nb_fire_rate(preds_c)
            acc_c = _compute_accuracy(preds_c, test_gold)
            for _, row in acc_c.iterrows():
                all_rows.append({
                    "category": cat, "attr": row["attr"],
                    "variant": "C_nb_ml",
                    "tau": tau,
                    "accuracy": row["accuracy"],
                    "n_test": row["n_test"],
                    "nb_fire_rate": fire_rate,
                })

        # ------------------------------------------------------------------ #
        # Variant D: nb-only (no ML fallback), at default tau
        # ------------------------------------------------------------------ #
        logger.info("  Variant D: nb-only (tau=%.2f)", DEFAULT_TAU)
        preds_d = predict_cascade_with_nb(
            products, cat,
            nb_threshold=DEFAULT_TAU,
            use_hybrid=False,
            include_regex=False,
            nb_only=True,
        )
        preds_d["category"] = cat
        fire_rate_d = _nb_fire_rate(preds_d)
        acc_d = _compute_accuracy(preds_d, test_gold)
        for _, row in acc_d.iterrows():
            all_rows.append({
                "category": cat, "attr": row["attr"],
                "variant": "D_nb_only",
                "tau": DEFAULT_TAU,
                "accuracy": row["accuracy"],
                "n_test": row["n_test"],
                "nb_fire_rate": fire_rate_d,
            })

    result = pd.DataFrame(all_rows)
    out_path = Path(PROCESSED_DIR) / "layer1_5_ablation.parquet"
    result.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(result), out_path)

    # ------------------------------------------------------------------ #
    # Print summary tables
    # ------------------------------------------------------------------ #
    print("\n=== LAYER 1.5 ABLATION RESULTS ===")
    print("Variants: A=regex+ml, B=ml_only, C=nb+ml (default tau=0.85), D=nb_only\n")

    # Per-cat mean accuracy for A, B, D and C@0.85
    c_default = result[(result["variant"] == "C_nb_ml") & (result["tau"] == DEFAULT_TAU)]
    summary_parts = [
        result[result["variant"] == "A_regex_ml"].groupby("category")["accuracy"].mean().rename("A_regex_ml"),
        result[result["variant"] == "B_ml_only"].groupby("category")["accuracy"].mean().rename("B_ml_only"),
        c_default.groupby("category")["accuracy"].mean().rename("C_nb_ml@0.85"),
        result[result["variant"] == "D_nb_only"].groupby("category")["accuracy"].mean().rename("D_nb_only"),
    ]
    cat_summary = pd.concat(summary_parts, axis=1)
    cat_summary["C_vs_A"] = (cat_summary["C_nb_ml@0.85"] - cat_summary["A_regex_ml"]) * 100
    cat_summary["C_vs_B"] = (cat_summary["C_nb_ml@0.85"] - cat_summary["B_ml_only"]) * 100
    print("Per-category mean accuracy:")
    print(cat_summary.round(4).to_string())

    print("\nGrand means:")
    a_mean = result[result["variant"] == "A_regex_ml"]["accuracy"].mean()
    b_mean = result[result["variant"] == "B_ml_only"]["accuracy"].mean()
    c_mean = c_default["accuracy"].mean()
    d_mean = result[result["variant"] == "D_nb_only"]["accuracy"].mean()
    print(f"  A (regex+ml):  {a_mean:.4f}")
    print(f"  B (ml_only):   {b_mean:.4f}")
    print(f"  C (nb+ml@.85): {c_mean:.4f}")
    print(f"  D (nb_only):   {d_mean:.4f}")
    print(f"\n  C - A = {(c_mean - a_mean) * 100:+.2f} pp")
    print(f"  C - B = {(c_mean - b_mean) * 100:+.2f} pp")

    # Tau sweep for C
    print("\n=== TAU SWEEP (Variant C — nb+ml) ===")
    tau_sweep_df = result[result["variant"] == "C_nb_ml"].copy()
    tau_tbl = tau_sweep_df.groupby("tau").agg(
        mean_acc=("accuracy", "mean"),
        mean_fire_rate=("nb_fire_rate", "mean"),
    ).reset_index()
    tau_tbl["mean_acc_pp"] = (tau_tbl["mean_acc"] * 100).round(2)
    tau_tbl["fire_rate_pct"] = (tau_tbl["mean_fire_rate"] * 100).round(1)
    print(tau_tbl[["tau", "mean_acc_pp", "fire_rate_pct"]].to_string(index=False))
    best_tau_row = tau_tbl.loc[tau_tbl["mean_acc"].idxmax()]
    print(f"\nBest tau: {best_tau_row['tau']:.2f} @ mean_acc={best_tau_row['mean_acc_pp']:.2f}%")

    # Problematic attrs detail
    problematic = [
        ("chocolate", "cocoa_percentage"),
        ("pasta", "grain_type"),
        ("pasta", "pasta_shape"),
        ("chocolate", "chocolate_type"),
    ]
    print("\n=== PROBLEMATIC ATTRS: A vs B vs C vs D ===")
    for p_cat, p_attr in problematic:
        rows = result[result["category"] == p_cat]
        a = rows[(rows["variant"] == "A_regex_ml") & (rows["attr"] == p_attr)]["accuracy"]
        b = rows[(rows["variant"] == "B_ml_only") & (rows["attr"] == p_attr)]["accuracy"]
        c = result[
            (result["category"] == p_cat) & (result["attr"] == p_attr)
            & (result["variant"] == "C_nb_ml") & (result["tau"] == DEFAULT_TAU)
        ]["accuracy"]
        d = rows[(rows["variant"] == "D_nb_only") & (rows["attr"] == p_attr)]["accuracy"]
        a_v = a.values[0] * 100 if len(a) > 0 else float("nan")
        b_v = b.values[0] * 100 if len(b) > 0 else float("nan")
        c_v = c.values[0] * 100 if len(c) > 0 else float("nan")
        d_v = d.values[0] * 100 if len(d) > 0 else float("nan")
        print(
            f"  {p_cat}/{p_attr}: "
            f"A={a_v:.1f}%  B={b_v:.1f}%  C={c_v:.1f}%  D={d_v:.1f}%  "
            f"C-A={c_v - a_v:+.1f}pp"
        )

    # Best C tau for problematic attrs
    print("\n=== BEST TAU PER PROBLEMATIC ATTR (Variant C) ===")
    for p_cat, p_attr in problematic:
        sub = result[
            (result["category"] == p_cat) & (result["attr"] == p_attr)
            & (result["variant"] == "C_nb_ml")
        ].sort_values("accuracy", ascending=False)
        if len(sub) == 0:
            print(f"  {p_cat}/{p_attr}: no data")
        else:
            top = sub.iloc[0]
            print(
                f"  {p_cat}/{p_attr}: best tau={top['tau']:.2f}  "
                f"acc={top['accuracy'] * 100:.1f}%  nb_fire_rate={top['nb_fire_rate'] * 100:.1f}%"
            )


if __name__ == "__main__":
    main()
