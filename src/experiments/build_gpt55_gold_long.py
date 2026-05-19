"""Convert gpt55_gold wide-format parquets to long-format matching consensus_gold_v2.

Reads: datasets/processed/gpt55_gold/{cat}_gpt55_gold.parquet
Writes: datasets/processed/gpt55_gold_long.parquet
Then concatenates with v2 gold -> datasets/processed/consensus_gold_v2_expanded.parquet
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)

CATS = ["pasta", "chocolate", "cheeses"]
GPT55_GOLD_DIR = Path(PROCESSED_DIR) / "gpt55_gold"
SIGNAL_TAX_PATH = Path(PROCESSED_DIR) / "attribute_signal_taxonomy.parquet"
V2_GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_off_grounded.parquet"
GPT55_LONG_PATH = Path(PROCESSED_DIR) / "gpt55_gold_long.parquet"
EXPANDED_GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet"


def main() -> None:
    setup_logging()

    sig = pd.read_parquet(SIGNAL_TAX_PATH)
    sig_map: dict[tuple[str, str], str] = {
        (row["category"], row["attr"]): row["signal_type"]
        for _, row in sig.iterrows()
    }

    all_rows: list[dict] = []

    for cat in CATS:
        p = GPT55_GOLD_DIR / f"{cat}_gpt55_gold.parquet"
        if not p.exists():
            logger.warning("[%s] gpt55 gold not found at %s, skipping", cat, p)
            continue
        df = pd.read_parquet(p)
        logger.info("[%s] converting %d codes to long format", cat, len(df))
        for _, row in df.iterrows():
            code = str(row["code"])
            try:
                parsed = json.loads(row["parsed_json"]) if row["parsed_json"] else {}
            except (json.JSONDecodeError, TypeError):
                parsed = {}
            if not parsed:
                continue
            for attr, val in parsed.items():
                is_null = val is None
                gold_value = str(val) if val is not None else None
                signal_type = sig_map.get((cat, attr), "text_derived")
                all_rows.append({
                    "category": cat,
                    "code": code,
                    "attr": attr,
                    "gold_value": gold_value,
                    "gold_is_null": is_null,
                    "opus_reasoning": None,
                    "signal_type": signal_type,
                })

    if not all_rows:
        logger.error("No rows converted — check that gpt55_gold parquets exist")
        return

    gpt55_long = pd.DataFrame(all_rows)
    gpt55_long.to_parquet(GPT55_LONG_PATH, index=False)
    logger.info("Saved gpt55_gold_long: %d rows to %s", len(gpt55_long), GPT55_LONG_PATH)

    v2 = pd.read_parquet(V2_GOLD_PATH)
    logger.info("v2 gold: %d rows", len(v2))
    expanded = pd.concat([v2, gpt55_long], ignore_index=True)
    expanded.to_parquet(EXPANDED_GOLD_PATH, index=False)
    logger.info("Saved expanded gold: %d rows to %s", len(expanded), EXPANDED_GOLD_PATH)
    logger.info("Summary:")
    for cat in CATS:
        n_v2 = len(v2[(v2["category"] == cat)]["code"].unique()) if len(v2) else 0
        n_new = len(gpt55_long[gpt55_long["category"] == cat]["code"].unique()) if len(gpt55_long) else 0
        n_total = len(expanded[expanded["category"] == cat]["code"].unique())
        logger.info("  [%s] v2=%d + new=%d = total=%d unique codes", cat, n_v2, n_new, n_total)


if __name__ == "__main__":
    main()
