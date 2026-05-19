"""
Cost-aware router training: XGBoost binary classifier + isotonic calibration.

Model output: calibrated P(cascade_correct | features) ∈ [0, 1].

Usage:
    python -m src.pipeline.router.train
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.common import MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, setup_logging
from src.pipeline.router.data import FOOD_CATS, build_training_dataset, by_product_split
from src.pipeline.router.features import (
    build_brand_set,
    build_brand_attr_acc_table,
    build_class_freq_table,
    featurize,
)

logger = logging.getLogger(__name__)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Standard binned ECE. p ∈ [0,1] probabilities, y ∈ {0,1} labels."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        weight = mask.mean()
        ece += weight * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def _calibrator_predict(calibrator, raw: np.ndarray) -> np.ndarray:
    """Unified calibrated probability prediction for both isotonic and Platt."""
    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]
    # IsotonicRegression (or other regressor-like)
    return calibrator.predict(np.asarray(raw))


@dataclass
class RouterArtefacts:
    model: xgb.XGBClassifier
    calibrator: object  # IsotonicRegression or LogisticRegression (Platt)
    feature_columns: list[str]
    brand_set: set[str]
    train_codes: set[str] = field(default_factory=set)
    val_codes: set[str] = field(default_factory=set)
    test_codes: set[str] = field(default_factory=set)
    val_metrics: dict = field(default_factory=dict)
    class_freq_table: dict = field(default_factory=dict)
    brand_attr_acc_table: dict = field(default_factory=dict)
    calibrator_choice: str = "isotonic"
    isotonic_ece: float | None = None
    platt_ece: float | None = None
    calibrator_ece: float | None = None

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns calibrated P(cascade_correct) ∈ [0, 1]."""
        X, _ = featurize(df, brand_set=self.brand_set,
                         class_freq_table=self.class_freq_table,
                         brand_attr_acc_table=self.brand_attr_acc_table)
        raw = self.model.predict_proba(X)[:, 1]
        return _calibrator_predict(self.calibrator, raw)


def _enrich_with_product_meta(df: pd.DataFrame, processed_dir: str | Path) -> pd.DataFrame:
    """Join product_name + brands from silver_standard files."""
    processed_dir = Path(processed_dir)
    cats = df["category"].unique()
    frames = []
    for cat in cats:
        s = pd.read_parquet(processed_dir / f"{cat}_stratified_silver_standard.parquet")
        s["code"] = s["code"].astype(str)
        frames.append(s[["code", "product_name", "brands"]])
    silver = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code"])
    df = df.copy()
    df["code"] = df["code"].astype(str)
    return df.merge(silver, on="code", how="left")


def train_router(
    df: pd.DataFrame,
    seed: int = RANDOM_STATE,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
) -> RouterArtefacts:
    """Train XGBoost binary classifier + isotonic calibration.

    Args:
        splits: Optional pre-computed (train, val, test) DataFrames. If None,
                falls back to `by_product_split(df, seed=seed)` (legacy behavior).
    """
    if "product_name" not in df.columns or "brands" not in df.columns:
        df = _enrich_with_product_meta(df, PROCESSED_DIR)

    if splits is not None:
        train, val, test = splits
    else:
        train, val, test = by_product_split(df, seed=seed)
    brand_set = build_brand_set(train)
    class_freq_table = build_class_freq_table(train)
    brand_attr_acc_table = build_brand_attr_acc_table(train)

    X_train, cols = featurize(train, brand_set=brand_set,
                              class_freq_table=class_freq_table,
                              brand_attr_acc_table=brand_attr_acc_table)
    X_val, _ = featurize(val, brand_set=brand_set,
                         class_freq_table=class_freq_table,
                         brand_attr_acc_table=brand_attr_acc_table)
    y_train = train["cascade_correct"].values
    y_val = val["cascade_correct"].values

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_raw = model.predict_proba(X_val)[:, 1]
    y_val_bin = y_val.astype(int)

    # In gold mode (splits provided externally), select calibrator by ECE on val.
    # In silver/legacy mode (splits=None), preserve historical behavior:
    # always use isotonic (no ECE selection, no Platt).
    gold_mode = splits is not None

    if gold_mode:
        # 2026-05-13: out-of-sample ECE selection (see D.2.1 review).
        # In-sample ECE biases toward isotonic (unbounded capacity → trivially ECE≈0).
        # Fit on train, evaluate ECE on held-out val.
        train_raw = model.predict_proba(X_train)[:, 1]
        y_train_bin = y_train.astype(int)

        iso = IsotonicRegression(out_of_bounds="clip").fit(train_raw, y_train_bin)
        iso_proba = iso.predict(val_raw)
        iso_ece = expected_calibration_error(iso_proba, y_val_bin, n_bins=10)

        platt = LogisticRegression().fit(train_raw.reshape(-1, 1), y_train_bin)
        platt_proba = platt.predict_proba(val_raw.reshape(-1, 1))[:, 1]
        platt_ece = expected_calibration_error(platt_proba, y_val_bin, n_bins=10)

        if iso_ece <= platt_ece:
            chosen_name = "isotonic"
            calibrator = iso
            chosen_ece = iso_ece
            val_calibrated = iso_proba
        else:
            chosen_name = "platt"
            calibrator = platt
            chosen_ece = platt_ece
            val_calibrated = platt_proba

        logger.info(
            "Calibrator ECE: isotonic=%.4f, platt=%.4f → chose %s",
            iso_ece, platt_ece, chosen_name,
        )
    else:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_raw, y_val_bin)
        val_calibrated = calibrator.predict(val_raw)
        chosen_name = "isotonic"
        iso_ece = None
        platt_ece = None
        chosen_ece = None

    val_metrics = {
        "val_n": int(len(y_val)),
        "auc_raw": float(roc_auc_score(y_val, val_raw)) if len(set(y_val)) > 1 else float("nan"),
        "brier_raw": float(brier_score_loss(y_val, val_raw)),
        "brier_calibrated": float(brier_score_loss(y_val, val_calibrated)),
        "logloss_raw": float(log_loss(y_val, val_raw, labels=[0, 1])),
        "logloss_calibrated": float(log_loss(y_val, val_calibrated, labels=[0, 1])),
        "positive_rate_val": float(y_val.mean()),
    }

    return RouterArtefacts(
        model=model,
        calibrator=calibrator,
        feature_columns=cols,
        brand_set=brand_set,
        train_codes=set(train["code"].astype(str)),
        val_codes=set(val["code"].astype(str)),
        test_codes=set(test["code"].astype(str)),
        val_metrics=val_metrics,
        class_freq_table=class_freq_table,
        brand_attr_acc_table=brand_attr_acc_table,
        calibrator_choice=chosen_name,
        isotonic_ece=iso_ece,
        platt_ece=platt_ece,
        calibrator_ece=chosen_ece,
    )


def save_artefacts(artefacts: RouterArtefacts, models_dir: str | Path = MODELS_DIR,
                   suffix: str = ""):
    """Save model + meta + calibration.

    Args:
        suffix: Optional infix for artefact filenames (e.g. "_gold").
                Default "" preserves legacy paths (router_xgb.pkl, ...).
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / f"router{suffix}_xgb.pkl", "wb") as f:
        pickle.dump({"model": artefacts.model, "calibrator": artefacts.calibrator}, f)

    # Standalone calibrator pkl for runtime loading.
    with open(models_dir / f"router{suffix}_calibrator.pkl", "wb") as f:
        pickle.dump(artefacts.calibrator, f)

    meta = {
        "feature_columns": artefacts.feature_columns,
        "brand_set": sorted(artefacts.brand_set),
        "train_codes": sorted(artefacts.train_codes),
        "val_codes": sorted(artefacts.val_codes),
        "test_codes": sorted(artefacts.test_codes),
        "class_freq_table": [{"key": list(k), "value": v}
                              for k, v in artefacts.class_freq_table.items()],
        "brand_attr_acc_table": [{"key": list(k), "value": v}
                                  for k, v in artefacts.brand_attr_acc_table.items()],
        "calibrator": artefacts.calibrator_choice,
        "calibrator_ece": artefacts.calibrator_ece,
        "isotonic_ece": artefacts.isotonic_ece,
        "platt_ece": artefacts.platt_ece,
    }
    with open(models_dir / f"router{suffix}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(models_dir / f"router{suffix}_calibration.json", "w") as f:
        json.dump(artefacts.val_metrics, f, indent=2)


def train_router_loco(
    df: pd.DataFrame,
    holdout_category: str,
    seed: int = RANDOM_STATE,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
) -> RouterArtefacts:
    """Train router on all categories EXCEPT holdout, evaluate on holdout.

    The `category` one-hot feature is DROPPED so model can't trivially encode
    domain membership. Test set = all rows of holdout category.
    """
    if "product_name" not in df.columns or "brands" not in df.columns:
        df = _enrich_with_product_meta(df, PROCESSED_DIR)

    train_df = df[df["category"] != holdout_category].copy()
    test_df = df[df["category"] == holdout_category].copy()

    # Inside train_df: by-product 80/20 train/val
    rng_codes = (
        train_df[["code", "category"]]
        .drop_duplicates(subset=["code"])
        .reset_index(drop=True)
    )
    from sklearn.model_selection import train_test_split
    train_codes, val_codes = train_test_split(
        rng_codes["code"], test_size=0.2, random_state=seed,
        stratify=rng_codes["category"] if rng_codes["category"].nunique() > 1 else None,
    )
    train_split = train_df[train_df["code"].isin(set(train_codes))]
    val_split = train_df[train_df["code"].isin(set(val_codes))]

    brand_set = build_brand_set(train_split)
    X_train, cols = featurize(train_split, brand_set=brand_set, drop_category=True)
    X_val, _ = featurize(val_split, brand_set=brand_set, drop_category=True)
    y_train = train_split["cascade_correct"].values
    y_val = val_split["cascade_correct"].values

    model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, gamma=0.1,
        eval_metric="logloss", random_state=seed, n_jobs=1, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_raw = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw, y_val)

    return RouterArtefacts(
        model=model,
        calibrator=calibrator,
        feature_columns=cols,
        brand_set=brand_set,
        train_codes=set(train_split["code"].astype(str)),
        val_codes=set(val_split["code"].astype(str)),
        test_codes=set(test_df["code"].astype(str)),
        val_metrics={
            "val_n": int(len(y_val)),
            "auc_raw": float(roc_auc_score(y_val, val_raw)) if len(set(y_val)) > 1 else float("nan"),
            "brier_raw": float(brier_score_loss(y_val, val_raw)),
            "logloss_raw": float(log_loss(y_val, val_raw, labels=[0, 1])),
            "holdout_category": holdout_category,
        },
    )


def _apply_gold_overrides(
    df: pd.DataFrame,
    processed_dir: str | Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Override `silver_gt` from consensus_gold_v1_emulated.parquet and use brand-disjoint
    {cat}_gold_split.parquet for train/val/test instead of random split.

    Returns (df_with_overridden_target, (train_df, val_df, test_df)).
    Rows w/o a gold split assignment for their (cat, code) are dropped.
    """
    processed_dir = Path(processed_dir)
    df = df.copy()
    df["code"] = df["code"].astype(str)

    # 1) Merge gt_consensus and override silver_gt where present.
    gold_path = processed_dir / "consensus_gold_v1_emulated.parquet"
    gold = pd.read_parquet(gold_path)
    gold = gold[gold["gt_consensus"].notna()].copy()
    gold["code"] = gold["code"].astype(str)
    gold["category"] = gold["category"].astype(str)
    gold["attr"] = gold["attr"].astype(str)

    before = len(df)
    df = df.merge(
        gold[["category", "code", "attr", "gt_consensus"]],
        on=["category", "code", "attr"], how="left",
    )
    n_overridden = df["gt_consensus"].notna().sum()
    df["silver_gt"] = df["gt_consensus"].combine_first(df["silver_gt"])
    df = df.drop(columns=["gt_consensus"])

    # Normalize case на всех value-колонках: gt_consensus в consensus_gold уже
    # lowercase (см. build_consensus_gold._normalize), а cascade_pred / llm_pred
    # сохраняют исходный case ("Wheat", "False"). Без приведения case-sensitive
    # equality «Wheat» != «wheat» ломает и cascade_correct, и downstream anchors.
    for col in ("cascade_pred", "llm_pred", "silver_gt"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()

    # Recompute cascade_correct against (now possibly overridden) silver_gt.
    df["cascade_correct"] = (df["cascade_pred"] == df["silver_gt"]).astype(int)
    logger.info("gold: merged %d gt_consensus overrides into %d rows",
                int(n_overridden), before)

    # 2) Per-category brand-disjoint splits.
    split_frames = []
    for cat in df["category"].unique():
        sp_path = processed_dir / f"{cat}_gold_split.parquet"
        sp = pd.read_parquet(sp_path)
        sp["code"] = sp["code"].astype(str)
        sp["category"] = cat
        split_frames.append(sp[["code", "category", "split"]])
    splits = pd.concat(split_frames, ignore_index=True)

    df = df.merge(splits, on=["code", "category"], how="inner")
    train_df = df[df["split"] == "train"].drop(columns=["split"]).copy()
    val_df = df[df["split"] == "val"].drop(columns=["split"]).copy()
    test_df = df[df["split"] == "test"].drop(columns=["split"]).copy()
    df = df.drop(columns=["split"])
    logger.info("gold: brand-disjoint splits — train=%d val=%d test=%d (rows)",
                len(train_df), len(val_df), len(test_df))
    return df, (train_df, val_df, test_df)


def main(gold: bool = False):
    setup_logging()
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "router_train.parquet"))
    logger.info("Loaded %d rows", len(df))
    df = _enrich_with_product_meta(df, PROCESSED_DIR)

    splits = None
    suffix = ""
    if gold:
        suffix = "_gold"
        df, splits = _apply_gold_overrides(df, PROCESSED_DIR)

    artefacts = train_router(df, splits=splits)
    save_artefacts(artefacts, suffix=suffix)
    logger.info("Val metrics: %s", artefacts.val_metrics)
    logger.info("Saved router%s_xgb.pkl, router%s_meta.json, router%s_calibration.json",
                suffix, suffix, suffix)


def main_loco():
    setup_logging()
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "router_train.parquet"))
    df = _enrich_with_product_meta(df, PROCESSED_DIR)
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    for cat in FOOD_CATS:
        logger.info("=== LOCO holdout: %s ===", cat)
        artefacts = train_router_loco(df, holdout_category=cat)
        out = models_dir / f"router_loco_{cat}.pkl"
        with open(out, "wb") as f:
            pickle.dump({
                "model": artefacts.model,
                "calibrator": artefacts.calibrator,
                "feature_columns": artefacts.feature_columns,
                "brand_set": sorted(artefacts.brand_set),
                "test_codes": sorted(artefacts.test_codes),
                "val_metrics": artefacts.val_metrics,
            }, f)
        logger.info("Saved %s. Val AUC=%.3f", out, artefacts.val_metrics["auc_raw"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loco", action="store_true", help="Train 6 LOCO models")
    parser.add_argument(
        "--gold", action="store_true",
        help="Train router against consensus_gold_v1_emulated.parquet (not silver). "
             "Uses {cat}_gold_split.parquet for brand-disjoint train/val/test. "
             "Saves artefacts as router_gold_xgb.pkl / router_gold_meta.json / "
             "router_gold_calibration.json to avoid clobbering silver-trained model.",
    )
    args = parser.parse_args()
    if args.loco:
        if args.gold:
            raise SystemExit("--loco and --gold are not currently supported together.")
        main_loco()
    else:
        main(gold=args.gold)
