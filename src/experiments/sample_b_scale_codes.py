"""Sample 650 NEW codes per category (pasta/chocolate/cheeses) for B-scale gold annotation.

Excludes any code already present in consensus_gold_v2_off_grounded.parquet.
Saves CSV files to datasets/manual_label/{cat}_b_scale_650.csv with header `code`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)

CATS = ["pasta", "chocolate", "cheeses"]
N_SAMPLE = 650
SEED = 42
OUT_DIR = Path("datasets/manual_label")
GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_off_grounded.parquet"


def main() -> None:
    setup_logging()

    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)

    for cat in CATS:
        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        )
        silver["code"] = silver["code"].astype(str)

        # Codes already annotated in v2 gold for this category
        existing_codes = set(gold[gold["category"] == cat]["code"].unique())
        logger.info("[%s] existing v2 gold codes: %d", cat, len(existing_codes))

        # Filter silver codes not yet in gold
        eligible = silver[~silver["code"].isin(existing_codes)]["code"].unique().tolist()
        logger.info("[%s] eligible new codes: %d", cat, len(eligible))

        if len(eligible) < N_SAMPLE:
            logger.warning(
                "[%s] only %d eligible codes < %d requested; using all",
                cat, len(eligible), N_SAMPLE,
            )
            sampled = eligible
        else:
            rng = pd.Series(eligible).sample(n=N_SAMPLE, random_state=SEED)
            sampled = rng.tolist()

        out_path = OUT_DIR / f"{cat}_b_scale_650.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"code": sampled}).to_csv(out_path, index=False)
        logger.info("[%s] saved %d codes to %s", cat, len(sampled), out_path)

    logger.info("Done. CSVs written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
