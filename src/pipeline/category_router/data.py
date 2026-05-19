"""Build positive training set for category router from stratified datasets."""
from __future__ import annotations

import logging
import os

import pandas as pd

from src.pipeline.category_router.constants import (
    ROUTER_CLASSES,
    ROUTER_INPUT_FIELDS,
)

logger = logging.getLogger(__name__)


def _parquet_path(processed_dir: str, category: str) -> str:
    if category == "electronics":
        return os.path.join(processed_dir, "electronics_silver_standard.parquet")
    return os.path.join(processed_dir, f"{category}_stratified_raw.parquet")


def load_positive(processed_dir: str, n_per_class: int, seed: int) -> pd.DataFrame:
    """Load and balance positive examples for all known categories.

    Each class is downsampled to `min(n_per_class, smallest_class_size_across_all_classes)`,
    so the returned DataFrame has equal counts per category.
    """
    # Pass 1: load each class, keep full frame, record available size.
    raw: dict[str, pd.DataFrame] = {}
    available: dict[str, int] = {}
    for category in ROUTER_CLASSES:
        path = _parquet_path(processed_dir, category)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing positive parquet: {path}")
        df = pd.read_parquet(path)
        for col in ROUTER_INPUT_FIELDS:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)
        df = df[list(ROUTER_INPUT_FIELDS)].copy()
        df["category_label"] = category
        raw[category] = df
        available[category] = min(n_per_class, len(df))

    take = min(available.values())
    logger.info("balancing all classes to %d rows (available per class: %s)",
                take, available)

    # Pass 2: sample every class to the global min `take`.
    frames: list[pd.DataFrame] = []
    for category in ROUTER_CLASSES:
        df = raw[category].sample(n=take, random_state=seed)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["brand"] = (
        out["brands"].str.split(",").str[0].str.strip().str.lower()
    )
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out[
        list(ROUTER_INPUT_FIELDS) + ["category_label", "brand"]
    ]
