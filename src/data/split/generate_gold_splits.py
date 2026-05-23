"""Сгенерировать brand-disjoint 60/20/20 splits для всех 6 категорий.

Для каждой категории c:
  1. Загрузить {c}_stratified_silver_standard.parquet (брендовая колонка = brands).
  2. Применить brand_disjoint_split с seed=42.
  3. Записать колонку split в parquet.
"""
from __future__ import annotations
import logging
import pandas as pd
from src.common import PROCESSED_DIR, setup_logging
from src.data.split.brand_disjoint import brand_disjoint_split

from src.common import MAIN_CATEGORIES
FOOD_CATS = MAIN_CATEGORIES  # 3 main cats для текущей итерации; для следующей — ALL_CATEGORIES

def main():
    setup_logging()
    log = logging.getLogger(__name__)
    for cat in FOOD_CATS:
        silver = pd.read_parquet(f"{PROCESSED_DIR}/{cat}_stratified_silver_standard.parquet")
        silver["code"] = silver["code"].astype(str)
        # brand колонка в OFF — "brands" (может быть list или comma-sep string)
        # Canonical multi-brand norm: «Carrefour, Carrefour BIO» и «Carrefour BIO, Carrefour»
        # дают одинаковый brand_norm «carrefour|carrefour bio», исключая subbrand leak.
        silver["brand_norm"] = (
            silver["brands"].fillna("UNKNOWN").astype(str).str.lower()
            .str.split(",")
            .apply(lambda parts: "|".join(sorted(p.strip() for p in parts if p.strip())))
        )
        splits = brand_disjoint_split(silver, brand_col="brand_norm",
                                       ratios=(0.6, 0.2, 0.2), seed=42)
        rows = []
        for name, df in splits.items():
            for code in df["code"]:
                rows.append({"code": code, "split": name})
        split_df = pd.DataFrame(rows)
        out = f"{PROCESSED_DIR}/{cat}_gold_split.parquet"
        split_df.to_parquet(out, index=False)
        sizes = split_df["split"].value_counts()
        log.info("[%s] train=%d val=%d test=%d → %s",
                 cat, sizes.get("train", 0), sizes.get("val", 0),
                 sizes.get("test", 0), out)

if __name__ == "__main__":
    main()
