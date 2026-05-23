"""Per-taxonomy headline breakdown for MAIN_CATEGORIES.

Группирует accuracy и coverage по signal_type (nutri_derived / tag_derived / text_derived).
Это разбивает монолитный headline 91.5% на три уровня confidence:
- nutri_derived: наименьший circularity risk (silver из raw нутриентов).
- tag_derived: moderate (silver из labels_tags/categories_tags; blind audit на этих не валидирует — см. §2.10).
- text_derived: высокий circularity risk (silver = regex по ingredients_text/product_name; ML — embeddings тех же полей).

Output: datasets/processed/headline_by_taxonomy.parquet
Columns: category, signal_type, n_cells, accuracy, ci_lo, ci_hi, n_brands
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import MAIN_CATEGORIES, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

PROCESSED = Path(PROCESSED_DIR)
N_BOOTSTRAP = 1000
RNG_SEED = 42


def load_per_product_with_taxonomy(cat: str) -> pd.DataFrame:
    """Per-product cascade results with signal_type joined."""
    exp = pd.read_parquet(PROCESSED / f"experiment_per_product_{cat}_stratified.parquet")
    tax = pd.read_parquet(PROCESSED / "attribute_signal_taxonomy.parquet")
    tax_cat = tax[tax.category == cat][["attr", "signal_type"]]
    brands = pd.read_parquet(
        PROCESSED / f"{cat}_stratified_silver_standard.parquet",
        columns=["code", "brands"],
    )
    brands["code"] = brands["code"].astype(str)
    exp["code"] = exp["code"].astype(str)
    merged = exp.merge(tax_cat, on="attr", how="inner").merge(brands, on="code", how="left")
    merged["correct"] = (merged["pred"] == merged["gt"]).astype(int)
    merged["non_null_gt"] = merged["gt"].notna().astype(int)
    return merged


def bootstrap_ci(df: pd.DataFrame, n_iter: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> tuple[float, float, float]:
    """Brand-clustered bootstrap: resampling брендов, не cells."""
    df = df[df.non_null_gt == 1].copy()
    if len(df) == 0:
        return float("nan"), float("nan"), float("nan")
    brands = df.brands.unique()
    by_brand = {b: df[df.brands == b] for b in brands}
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_iter):
        sampled = rng.choice(brands, size=len(brands), replace=True)
        parts = [by_brand[b] for b in sampled]
        boot = pd.concat(parts, ignore_index=True)
        if len(boot) > 0:
            accs.append(boot.correct.mean())
    central = df.correct.mean()
    return float(central), float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main():
    rows = []
    for cat in MAIN_CATEGORIES:
        logger.info("Processing %s", cat)
        merged = load_per_product_with_taxonomy(cat)
        for st in ["nutri_derived", "tag_derived", "text_derived"]:
            sub = merged[merged.signal_type == st]
            if len(sub) == 0:
                continue
            n_brands = sub.brands.nunique()
            acc, lo, hi = bootstrap_ci(sub)
            n_cells = (sub.non_null_gt == 1).sum()
            rows.append({
                "category": cat,
                "signal_type": st,
                "n_cells": int(n_cells),
                "n_brands": int(n_brands),
                "accuracy": acc,
                "ci_lo": lo,
                "ci_hi": hi,
            })
            logger.info("  %s/%s: n=%d acc=%.4f CI=[%.4f, %.4f]",
                        cat, st, n_cells, acc, lo, hi)
    out = pd.DataFrame(rows)
    out_path = PROCESSED / "headline_by_taxonomy.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(out))


if __name__ == "__main__":
    main()
