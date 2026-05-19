"""Train production hybrid cascade ML layer: silver + 5x v2 gold."""
from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from src.common import MODELS_DIR, PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)

OFF_CATS = ["pasta", "chocolate", "cheeses"]


def _xgb_classifier(n_classes: int, pos_weight: float | None = None):
    common = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        tree_method="hist", verbosity=0,
    )
    if n_classes == 2:
        return xgb.XGBClassifier(scale_pos_weight=pos_weight or 1.0, **common)
    return xgb.XGBClassifier(objective="multi:softmax", num_class=n_classes, **common)


def train_hybrid_for_attr(cat: str, attr: str,
                          silver: pd.DataFrame, emb: np.ndarray,
                          gold: pd.DataFrame,
                          code_to_idx: dict,
                          tier_weights: dict | None = None,
                          output_dir: Path | None = None,
                          holdout_codes: set | None = None) -> bool:
    """Train + save hybrid model for one (cat, attr). True if saved.

    tier_weights: {silver, tier1, tier2, tier3} → per-tier sample_weight.
    Default {silver:1, tier1:5, tier2:5, tier3:5} preserves original 5x-gold behavior.
    """
    if tier_weights is None:
        tier_weights = {"silver": 1.0, "tier0": 8.0, "tier1": 5.0, "tier2": 5.0, "tier3": 5.0}
    tier_weights.setdefault("tier0", 8.0)
    cat_gold = gold[(gold["category"] == cat) & (gold["attr"] == attr)
                    & ~gold["gold_is_null"]].copy()
    cat_gold["code"] = cat_gold["code"].astype(str)
    cat_gold = cat_gold[cat_gold["code"].isin(code_to_idx)]
    if len(cat_gold) < 10:
        logger.warning("[%s/%s] gold n=%d, skip", cat, attr, len(cat_gold))
        return False
    if attr not in silver.columns:
        logger.warning("[%s/%s] no silver col, skip", cat, attr)
        return False

    silver_keep = silver[silver[attr].notna()].copy()
    silver_keep["code"] = silver_keep["code"].astype(str)
    silver_keep = silver_keep[silver_keep["code"].isin(code_to_idx)]
    # Leakage fix: drop silver rows for any code held out from eval.
    # Without this, fold-k holdout codes leak into training via their silver labels.
    if holdout_codes:
        silver_keep = silver_keep[~silver_keep["code"].isin(holdout_codes)]
    sil_idx = np.array([code_to_idx[c] for c in silver_keep["code"]])
    X_sil = emb[sil_idx]
    y_sil = silver_keep[attr].astype(str).values

    gold_idx = np.array([code_to_idx[c] for c in cat_gold["code"]])
    X_gold = emb[gold_idx]
    y_gold = cat_gold["gold_value"].astype(str).values

    # Per-row gold weight derived from tier column (if present)
    def _row_w(t):
        if not isinstance(t, str):
            return tier_weights["tier2"]
        if t.startswith("tier0"): return tier_weights["tier0"]
        if t.startswith("tier1"): return tier_weights["tier1"]
        if t.startswith("tier2"): return tier_weights["tier2"]
        if t.startswith("tier3"): return tier_weights["tier3"]
        return tier_weights["tier2"]  # default for legacy rows without tier
    if "tier" in cat_gold.columns:
        w_gold = cat_gold["tier"].apply(_row_w).values.astype(float)
    else:
        w_gold = np.full(len(cat_gold), tier_weights["tier2"], dtype=float)

    # Drop silver rows that conflict with gold (same code, gold takes priority)
    gold_codes = set(cat_gold["code"])
    keep = ~np.isin(silver_keep["code"].values, list(gold_codes))
    X_sil, y_sil = X_sil[keep], y_sil[keep]

    X = np.vstack([X_sil, X_gold])
    y = np.concatenate([y_sil, y_gold])
    w = np.concatenate([tier_weights["silver"] * np.ones(len(y_sil)), w_gold])

    classes = sorted(set(y))
    if len(classes) < 2:
        logger.warning("[%s/%s] single class after merge, skip", cat, attr)
        return False
    remap = {c: i for i, c in enumerate(classes)}
    y_r = np.array([remap[c] for c in y])
    n_classes = len(classes)
    pos_weight = None
    if n_classes == 2:
        pos = int((y_r == 1).sum())
        neg = int((y_r == 0).sum())
        pos_weight = max(neg / max(pos, 1), 0.5)
    clf = _xgb_classifier(n_classes, pos_weight=pos_weight)
    clf.fit(X, y_r, sample_weight=w)

    le = LabelEncoder()
    le.fit(classes)
    # Ensure LE classes are in remap order
    assert list(le.classes_) == classes

    out_root = Path(output_dir) if output_dir else Path(MODELS_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
    out_xgb = out_root / f"{cat}_stratified_{attr}_xgb_hybrid.pkl"
    out_le = out_root / f"{cat}_stratified_{attr}_le_hybrid.pkl"
    with open(out_xgb, "wb") as f:
        pickle.dump(clf, f)
    with open(out_le, "wb") as f:
        pickle.dump(le, f)
    logger.info("[%s/%s] saved n_train=%d n_silver=%d n_gold=%d n_classes=%d",
                cat, attr, len(y), len(y_sil), len(y_gold), n_classes)
    return True


def main():
    setup_logging()
    p = argparse.ArgumentParser(description="Train production hybrid (silver + 5x v2 gold) XGB models.")
    p.add_argument("--cats", nargs="+", default=OFF_CATS)
    p.add_argument(
        "--gold-path",
        type=Path,
        default=Path(PROCESSED_DIR) / "consensus_gold_v2_off_grounded.parquet",
        help="Path to long-format gold parquet (default: consensus_gold_v2_off_grounded.parquet)",
    )
    p.add_argument("--w-silver", type=float, default=1.0)
    p.add_argument("--w-tier0", type=float, default=8.0, help="OFF-derived deterministic")
    p.add_argument("--w-tier1", type=float, default=5.0, help="Opus blind OFF-grounded")
    p.add_argument("--w-tier2", type=float, default=5.0, help="gpt-5.5 expansion")
    p.add_argument("--w-tier3", type=float, default=5.0, help="gemini-flash B3 expansion")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override MODELS_DIR (useful for CV folds)")
    p.add_argument("--holdout-codes", type=Path, default=None,
                   help="CSV with 'code' column; these codes excluded from silver+gold during train "
                        "(prevents silver-leakage during CV/holdout evals)")
    args = p.parse_args()
    holdout_codes_set: set[str] = set()
    if args.holdout_codes:
        holdout_codes_set = set(pd.read_csv(args.holdout_codes)["code"].astype(str))
        logger.info("Holdout codes loaded: %d (excluded from silver+gold)", len(holdout_codes_set))

    gold = pd.read_parquet(args.gold_path)
    gold["code"] = gold["code"].astype(str)
    if holdout_codes_set:
        before = len(gold)
        gold = gold[~gold["code"].isin(holdout_codes_set)]
        logger.info("Dropped %d gold rows for holdout codes (%d → %d)",
                    before - len(gold), before, len(gold))
    tw = {"silver": args.w_silver, "tier0": args.w_tier0, "tier1": args.w_tier1,
          "tier2": args.w_tier2, "tier3": args.w_tier3}
    logger.info("Tier weights: %s", tw)

    total_saved = 0
    for cat in args.cats:
        silver = pd.read_parquet(Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet")
        silver["code"] = silver["code"].astype(str)
        emb = np.load(Path(PROCESSED_DIR) / f"{cat}_stratified_embeddings.npy")
        code_to_idx = {c: i for i, c in enumerate(silver["code"].tolist())}
        attrs = sorted(gold[gold["category"] == cat]["attr"].unique())
        logger.info("[%s] training hybrid models for %d attrs", cat, len(attrs))
        saved = 0
        for attr in attrs:
            ok = train_hybrid_for_attr(cat, attr, silver, emb, gold, code_to_idx, tw,
                                       output_dir=args.output_dir,
                                       holdout_codes=holdout_codes_set or None)
            if ok:
                saved += 1
        logger.info("[%s] saved %d/%d hybrid models", cat, saved, len(attrs))
        total_saved += saved

    logger.info("Total hybrid models saved: %d", total_saved)


if __name__ == "__main__":
    main()
