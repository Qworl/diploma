"""Sample out-of-domain products from raw OFF dump."""
from __future__ import annotations

import logging
import os

import pandas as pd

from src.data.filter import CATEGORY_CLEANUP
from src.pipeline.category_router.constants import ROUTER_INPUT_FIELDS

logger = logging.getLogger(__name__)


def _known_food_tags() -> set[str]:
    """Union of include_tags / include_substrings across food categories
    that source from OFF (pasta, chocolate, beverages, cheeses, cereals)."""
    food_in_off = ("pasta", "chocolate", "beverages", "cheeses", "cereals")
    tags: set[str] = set()
    for cat in food_in_off:
        cfg = CATEGORY_CLEANUP.get(cat, {})
        for t in cfg.get("include_tags", []):
            tags.add(t.lower())
        for sub in cfg.get("include_substrings", []):
            tags.add(f"en:{sub.lower()}")
    return tags


def sample_ood(off_parquet: str, n: int, seed: int) -> pd.DataFrame:
    """Random sample of OOD products from raw OFF, excluding known food tags.

    Returns DataFrame with columns:
        product_name, brands, ingredients_text, quantity,
        category_label="unknown", brand, categories_tags_raw

    NOTE: the resulting pool is systematically biased toward "food-non-known"
    items (snacks, dairy, baby food, etc.). Real partner-OOD from
    fundamentally different domains (tools, cosmetics-not-in-OBF, apparel,
    furniture) is NOT represented. OOD-AUROC measured against this sample
    is therefore optimistic for non-food partner products.
    """
    if not os.path.exists(off_parquet):
        raise FileNotFoundError(f"OFF parquet not found: {off_parquet}")
    cols_to_read = list(ROUTER_INPUT_FIELDS) + ["categories_tags"]
    df = pd.read_parquet(off_parquet, columns=cols_to_read)
    df = df[df["product_name"].fillna("").str.len() > 0].copy()
    for col in ROUTER_INPUT_FIELDS:
        df[col] = df[col].fillna("").astype(str)
    df["categories_tags"] = df["categories_tags"].fillna("").astype(str)

    known = _known_food_tags()

    def _has_known(tag_csv: str) -> bool:
        return any(t.strip().lower() in known for t in tag_csv.split(","))

    mask_ood = ~df["categories_tags"].apply(_has_known)
    pool = df[mask_ood].copy()
    logger.info("OOD pool size: %d (out of %d total OFF rows)",
                len(pool), len(df))

    take = min(n, len(pool))
    sample = pool.sample(n=take, random_state=seed).reset_index(drop=True)
    sample["category_label"] = "unknown"
    sample["brand"] = (
        sample["brands"].str.split(",").str[0].str.strip().str.lower()
    )
    sample = sample.rename(columns={"categories_tags": "categories_tags_raw"})
    return sample[
        list(ROUTER_INPUT_FIELDS)
        + ["category_label", "brand", "categories_tags_raw"]
    ]
