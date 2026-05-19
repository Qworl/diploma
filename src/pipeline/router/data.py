"""
Router training data: join cascade per-product predictions with direct LLM
predictions and silver ground truth, produce binary target `cascade_correct`.

Inputs (per category):
    datasets/processed/experiment_per_product_{cat}_stratified.parquet
        cols: config, code, attr, gt, pred, conf, layer
    datasets/processed/direct_llm_eval_{cat}_stratified.parquet
        cols: category, code, attr, gt, pred, predicted_non_null, gt_non_null,
              correct_when_both_present

Output:
    datasets/processed/router_train.parquet
        cols: category, code, attr, silver_gt, cascade_pred, cascade_conf,
              cascade_layer, llm_pred, cascade_correct
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import PROCESSED_DIR

logger = logging.getLogger(__name__)

FOOD_CATS = ["pasta", "chocolate", "beverages", "cheeses", "cereals", "cosmetics"]


def load_joined_cat(category: str, processed_dir: str | Path,
                    llm_suffix: str = "") -> pd.DataFrame:
    """Join cascade per-product predictions with direct LLM for one category.

    Args:
        llm_suffix: If non-empty, reads direct_llm_eval_{category}_stratified_{llm_suffix}.parquet
                    instead of the default (no suffix) file.
    """
    processed_dir = Path(processed_dir)
    cascade_path = processed_dir / f"experiment_per_product_{category}_stratified.parquet"
    _sfx = f"_{llm_suffix}" if llm_suffix else ""
    direct_path = processed_dir / f"direct_llm_eval_{category}_stratified{_sfx}.parquet"

    cascade = pd.read_parquet(cascade_path)
    cascade = cascade[cascade["config"] == "regex_ml_bayes"].copy()
    cascade = cascade.rename(columns={
        "pred": "cascade_pred", "conf": "cascade_conf",
        "layer": "cascade_layer", "gt": "silver_gt_casc",
    })

    direct = pd.read_parquet(direct_path)
    direct = direct.rename(columns={"pred": "llm_pred", "gt": "silver_gt_llm"})

    merged = cascade.merge(direct[["code", "attr", "llm_pred", "silver_gt_llm"]],
                            on=["code", "attr"], how="inner")
    # silver_gt should agree between two sources; prefer cascade-side
    merged["silver_gt"] = merged["silver_gt_casc"].fillna(merged["silver_gt_llm"])
    merged["category"] = category

    return merged[[
        "category", "code", "attr", "silver_gt",
        "cascade_pred", "cascade_conf", "cascade_layer", "llm_pred",
    ]]


def build_training_dataset(
    categories: list[str],
    processed_dir: str | Path = PROCESSED_DIR,
    llm_suffix: str = "",
) -> pd.DataFrame:
    """Load all categories, drop rows w/o silver_gt, add cascade_correct target.

    Args:
        llm_suffix: Propagated to load_joined_cat; selects alternative LLM result files.
    """
    frames = [load_joined_cat(cat, processed_dir, llm_suffix=llm_suffix) for cat in categories]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["silver_gt"]).copy()
    df["cascade_correct"] = (
        df["cascade_pred"].astype(str) == df["silver_gt"].astype(str)
    ).astype(int)
    return df


def by_product_split(
    df: pd.DataFrame,
    seed: int = 42,
    val_size: float = 0.20,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by unique product code, stratified by category.

    Returns (train, val, test) DataFrames where no product appears in more than one split.
    """
    prod = df[["code", "category"]].drop_duplicates(subset=["code"]).reset_index(drop=True)

    prod_trainval, prod_test = train_test_split(
        prod, test_size=test_size, random_state=seed,
        stratify=prod["category"] if prod["category"].nunique() > 1 else None,
    )
    val_relative = val_size / (1 - test_size)
    prod_train, prod_val = train_test_split(
        prod_trainval, test_size=val_relative, random_state=seed,
        stratify=prod_trainval["category"] if prod_trainval["category"].nunique() > 1 else None,
    )

    train = df[df["code"].isin(prod_train["code"])].copy()
    val = df[df["code"].isin(prod_val["code"])].copy()
    test = df[df["code"].isin(prod_test["code"])].copy()
    return train, val, test


def main():
    from src.common import setup_logging
    setup_logging()
    df = build_training_dataset(FOOD_CATS)
    out = os.path.join(PROCESSED_DIR, "router_train.parquet")
    df.to_parquet(out, index=False)
    logger.info("Saved %s (%d rows, %d unique products)",
                 out, len(df), df["code"].nunique())
    logger.info("Class balance: %s",
                 df["cascade_correct"].value_counts(normalize=True).to_dict())


if __name__ == "__main__":
    main()
