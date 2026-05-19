"""E3: Per-attr ECE before/after isotonic calibration for production hybrid ML models.

For each (cat, attr) ∈ {pasta, chocolate, cheeses} × production attrs:
  - "before" = ECE of raw XGB confidence on v2 expanded gold test set.
  - "after"  = ECE after fitting isotonic regression on a brand-disjoint silver
               calibration split and applying the mapping to test confidences.

Output: datasets/processed/ece_calibration_table.parquet with columns
  [category, attr, n_test, ece_before, ece_after, ece_reduction_pp].

Entry: python -m src.diagnostics.ml.ece_table
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.common import MODELS_DIR, PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)

CATS = ["pasta", "chocolate", "cheeses"]
N_BINS = 10
ISO_SEED = 42
BRAND_HOLDOUT_FRAC = 0.25  # fraction of silver brands reserved for isotonic fit


def compute_ece_from_conf(confidences: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|."""
    if len(confidences) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidences[mask].mean())
        ece += (cnt / n) * abs(bin_acc - bin_conf)
    return float(ece)


def load_model(cat: str, attr: str):
    xgb_path = Path(MODELS_DIR) / f"{cat}_stratified_{attr}_xgb_hybrid.pkl"
    le_path = Path(MODELS_DIR) / f"{cat}_stratified_{attr}_le_hybrid.pkl"
    if not xgb_path.exists() or not le_path.exists():
        return None, None
    with open(xgb_path, "rb") as f:
        clf = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    return clf, le


def _primary_brand(brands_series: pd.Series) -> pd.Series:
    return brands_series.fillna("").astype(str).str.split(",").str[0].str.strip().str.lower()


def _brand_split(brands: np.ndarray, test_frac: float, seed: int) -> np.ndarray:
    """Return mask of rows whose brand is in the held-out test set."""
    uniq = np.unique(brands)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_brands = set(uniq[perm[:n_test]].tolist())
    return np.array([b in test_brands for b in brands])


def run_attr(
    cat: str,
    attr: str,
    silver: pd.DataFrame,
    embeddings: np.ndarray,
    gold_long: pd.DataFrame,
) -> dict | None:
    """Compute ECE before/after for one (cat, attr).

    Test set: v2 gold rows (consensus_gold_v2_expanded) joined back to silver
              embeddings (silver dump contains the gold codes since gold is a
              subset stratified-sampled out of OFF).
    Calib set: silver rows whose brand is in the calibration brand split AND
               whose code is NOT in test (to avoid trivial overlap).
    """
    clf, le = load_model(cat, attr)
    if clf is None:
        logger.warning("[%s/%s] model missing — skip", cat, attr)
        return None

    # ---- Test set: v2 gold for this attr ----
    g = gold_long[(gold_long["category"] == cat) & (gold_long["attr"] == attr)].copy()
    g = g[~g["gold_is_null"].astype(bool)]
    if len(g) == 0:
        logger.warning("[%s/%s] no v2 gold rows — skip", cat, attr)
        return None

    g["code"] = g["code"].astype(str)
    silver = silver.copy()
    silver["code"] = silver["code"].astype(str)
    # silver may have duplicate codes? assume unique. Build code→row_idx mapping.
    code_to_idx = {c: i for i, c in enumerate(silver["code"].values)}

    # Match gold codes to silver embeddings
    g["row_idx"] = g["code"].map(code_to_idx)
    g = g.dropna(subset=["row_idx"]).copy()
    if len(g) < 5:
        logger.warning("[%s/%s] only %d test rows after code-join — skip", cat, attr, len(g))
        return None
    g["row_idx"] = g["row_idx"].astype(int)

    # Filter to gold labels representable by LE classes
    valid_classes = set(map(str, le.classes_))
    g["gold_value"] = g["gold_value"].astype(str)
    g = g[g["gold_value"].isin(valid_classes)].copy()
    if len(g) < 5:
        logger.warning("[%s/%s] only %d test rows after class-filter — skip", cat, attr, len(g))
        return None

    test_idx = g["row_idx"].values
    X_test = embeddings[test_idx]
    y_test_labels = g["gold_value"].values

    # Predict raw proba
    proba_test = clf.predict_proba(X_test)
    pred_idx = proba_test.argmax(axis=1)
    pred_labels = np.array([str(c) for c in le.classes_])[pred_idx]
    conf_test_raw = proba_test.max(axis=1)
    correct_test = (pred_labels == y_test_labels).astype(float)

    ece_before = compute_ece_from_conf(conf_test_raw, correct_test)

    # ---- Calibration set: brand-disjoint slice of silver, excluding gold codes ----
    if attr not in silver.columns:
        logger.warning("[%s/%s] attr not in silver — cannot fit isotonic", cat, attr)
        # Report before only
        return {
            "category": cat,
            "attr": attr,
            "n_test": int(len(y_test_labels)),
            "ece_before": round(ece_before, 4),
            "ece_after": None,
            "ece_reduction_pp": None,
            "note": "no_silver_attr",
        }

    silver_valid = silver.dropna(subset=[attr]).copy()
    silver_valid = silver_valid[silver_valid[attr].astype(str).str.strip() != ""]
    # Exclude rows whose code appears in the test set
    test_codes = set(g["code"].values)
    silver_valid = silver_valid[~silver_valid["code"].isin(test_codes)].copy()
    silver_valid = silver_valid[silver_valid[attr].astype(str).isin(valid_classes)].copy()
    if len(silver_valid) < 50:
        logger.warning("[%s/%s] silver calib pool too small (%d) — skip after", cat, attr, len(silver_valid))
        return {
            "category": cat,
            "attr": attr,
            "n_test": int(len(y_test_labels)),
            "ece_before": round(ece_before, 4),
            "ece_after": None,
            "ece_reduction_pp": None,
            "note": "calib_too_small",
        }

    # Brand split on silver_valid: take the held-out brand subset as calibration data
    silver_valid["_brand"] = _primary_brand(silver_valid.get("brands", pd.Series([""] * len(silver_valid))))
    calib_mask = _brand_split(silver_valid["_brand"].values, BRAND_HOLDOUT_FRAC, ISO_SEED)
    calib_df = silver_valid[calib_mask]
    if len(calib_df) < 30:
        logger.warning("[%s/%s] calib subset tiny (%d) — fall back to all silver_valid", cat, attr, len(calib_df))
        calib_df = silver_valid

    calib_idx = calib_df.index.map(lambda r: code_to_idx.get(str(silver.iloc[r]["code"]), -1))
    # silver_valid has same index as silver subset rows — let me re-derive
    # Recompute: calib_df rows are slices of silver (since silver_valid is a filter of silver)
    # Use original silver row positions via a reindex
    silver_positions = silver_valid.index.values  # positional indices in silver
    # But silver_valid was filtered, so its .index is silver.index subset. silver.index is default 0..N-1, so it's positional.
    calib_positions = calib_df.index.values.astype(int)
    X_calib = embeddings[calib_positions]
    y_calib_labels = calib_df[attr].astype(str).values

    proba_calib = clf.predict_proba(X_calib)
    pred_calib_idx = proba_calib.argmax(axis=1)
    pred_calib_labels = np.array([str(c) for c in le.classes_])[pred_calib_idx]
    conf_calib_raw = proba_calib.max(axis=1)
    correct_calib = (pred_calib_labels == y_calib_labels).astype(float)

    if len(np.unique(correct_calib)) < 2:
        logger.warning("[%s/%s] calib set degenerate (all-correct or all-wrong) — skip isotonic", cat, attr)
        return {
            "category": cat,
            "attr": attr,
            "n_test": int(len(y_test_labels)),
            "ece_before": round(ece_before, 4),
            "ece_after": None,
            "ece_reduction_pp": None,
            "note": "calib_degenerate",
        }

    # Fit isotonic on calibration (confidence → empirical accuracy)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(conf_calib_raw, correct_calib)

    # Apply to test
    conf_test_iso = iso.predict(conf_test_raw)
    ece_after = compute_ece_from_conf(conf_test_iso, correct_test)

    reduction_pp = (ece_before - ece_after) * 100

    return {
        "category": cat,
        "attr": attr,
        "n_test": int(len(y_test_labels)),
        "n_calib": int(len(y_calib_labels)),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "ece_reduction_pp": round(reduction_pp, 2),
        "note": "",
    }


def main():
    setup_logging()
    gold = pd.read_parquet(Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet")
    gold["code"] = gold["code"].astype(str)

    results = []

    for cat in CATS:
        silver_path = Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        emb_path = Path(PROCESSED_DIR) / f"{cat}_stratified_embeddings.npy"
        if not silver_path.exists() or not emb_path.exists():
            logger.warning("missing silver or embeddings for %s — skip cat", cat)
            continue
        silver = pd.read_parquet(silver_path).reset_index(drop=True)
        embeddings = np.load(emb_path)
        if len(silver) != embeddings.shape[0]:
            logger.warning("silver(%d) vs emb(%d) mismatch for %s — skip cat",
                           len(silver), embeddings.shape[0], cat)
            continue

        # Discover attrs from model files
        model_glob = sorted(Path(MODELS_DIR).glob(f"{cat}_stratified_*_xgb_hybrid.pkl"))
        attrs = []
        for p in model_glob:
            name = p.name[len(f"{cat}_stratified_"):-len("_xgb_hybrid.pkl")]
            attrs.append(name)

        logger.info("[%s] %d attrs: %s", cat, len(attrs), attrs)

        for attr in attrs:
            try:
                r = run_attr(cat, attr, silver, embeddings, gold)
                if r is not None:
                    results.append(r)
                    logger.info(
                        "  %s/%s: n_test=%d before=%.4f after=%s",
                        cat, attr, r["n_test"], r["ece_before"], r["ece_after"],
                    )
            except Exception as e:
                logger.exception("[%s/%s] failed: %s", cat, attr, e)

    if not results:
        logger.error("no results")
        return

    df = pd.DataFrame(results)
    # Final column order
    keep_cols = ["category", "attr", "n_test", "ece_before", "ece_after", "ece_reduction_pp"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None
    extra = [c for c in df.columns if c not in keep_cols]
    df = df[keep_cols + extra]

    out_path = Path(PROCESSED_DIR) / "ece_calibration_table.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(df))

    # ---- Summary ----
    df_ok = df.dropna(subset=["ece_after"]).copy()
    print("\n=== ECE TABLE ===")
    print(df.to_string(index=False))

    print("\n=== SUMMARY ===")
    print(f"n attrs reported: {len(df)}")
    print(f"n with after-ECE computed: {len(df_ok)}")
    if len(df_ok) > 0:
        print(f"mean ECE before: {df['ece_before'].mean():.4f}")
        print(f"mean ECE after : {df_ok['ece_after'].mean():.4f}")
        print(f"mean reduction (pp): {df_ok['ece_reduction_pp'].mean():.2f}")

    problem = df[df["ece_before"] > 0.10].sort_values("ece_before", ascending=False)
    print(f"\nAttrs with ECE_before > 0.10 (production problem zone): {len(problem)}")
    if len(problem) > 0:
        print(problem[["category", "attr", "n_test", "ece_before", "ece_after"]].to_string(index=False))

    problem_after = df_ok[df_ok["ece_after"] > 0.10].sort_values("ece_after", ascending=False)
    print(f"\nAttrs with ECE_after > 0.10 (still bad after isotonic — need more work): {len(problem_after)}")
    if len(problem_after) > 0:
        print(problem_after[["category", "attr", "n_test", "ece_before", "ece_after"]].to_string(index=False))

    if len(df_ok) > 0:
        biggest = df_ok.sort_values("ece_reduction_pp", ascending=False).head(5)
        print("\nTop 5 ECE reductions (calibration wins):")
        print(biggest[["category", "attr", "n_test", "ece_before", "ece_after", "ece_reduction_pp"]].to_string(index=False))


if __name__ == "__main__":
    main()
