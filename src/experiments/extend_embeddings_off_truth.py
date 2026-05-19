"""Extend silver_standard.parquet + embeddings.npy with OFF-truth codes
that aren't already there. Unlocks the full OFF-derived truth holdout.

Source codes: anything in off_derived_truth.parquet not in current silver.
Text fields loaded from off_cache, off_cache_b3_full, off_cache_b3_r2.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

PROCESSED = Path("datasets/processed")
CACHE_DIRS = [
    Path("datasets/manual_label/off_cache"),
    Path("datasets/manual_label/off_cache_b3_full"),
    Path("datasets/manual_label/off_cache_b3_r2"),
]

logger.info("Loading SentenceTransformer...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def load_off(code: str) -> dict | None:
    for d in CACHE_DIRS:
        p = d / f"{code}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            prod = data.get("product", {})
            return {
                "code": code,
                "product_name": prod.get("product_name", "") or "",
                "brands": prod.get("brands", "") or "",
                "ingredients_text": prod.get("ingredients_text", "") or "",
                "quantity": prod.get("quantity", "") or "",
            }
    return None


def build_text(row: dict) -> str:
    parts = [row.get(k, "") for k in ("product_name", "brands", "ingredients_text", "quantity")]
    return " ".join(p for p in parts if p).strip() or " "


truth = pd.read_parquet(PROCESSED / "off_derived_truth.parquet")
truth["code"] = truth["code"].astype(str)
truth_codes_by_cat = truth.groupby("category")["code"].agg(lambda x: set(x.unique())).to_dict()

for cat in ["pasta", "chocolate", "cheeses"]:
    silver_path = PROCESSED / f"{cat}_stratified_silver_standard.parquet"
    emb_path = PROCESSED / f"{cat}_stratified_embeddings.npy"
    silver = pd.read_parquet(silver_path)
    silver["code"] = silver["code"].astype(str)
    existing = set(silver["code"])
    new_codes = sorted(truth_codes_by_cat.get(cat, set()) - existing)
    logger.info("[%s] truth codes: %d, in silver: %d, NEW to add: %d",
                cat, len(truth_codes_by_cat.get(cat, set())),
                len(truth_codes_by_cat.get(cat, set()) & existing), len(new_codes))
    if not new_codes:
        continue

    new_rows = []
    for code in new_codes:
        row = load_off(code)
        if row is not None:
            new_rows.append(row)
    logger.info("[%s] loaded text for %d/%d", cat, len(new_rows), len(new_codes))
    if not new_rows:
        continue

    texts = [build_text(r) for r in new_rows]
    logger.info("[%s] encoding %d texts...", cat, len(texts))
    new_emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    # Extend silver
    new_silver = pd.DataFrame(new_rows)
    for col in silver.columns:
        if col not in new_silver.columns:
            new_silver[col] = None
    new_silver = new_silver[silver.columns]
    extended_silver = pd.concat([silver, new_silver], ignore_index=True)
    extended_silver.to_parquet(silver_path, index=False)
    logger.info("[%s] silver: %d → %d rows", cat, len(silver), len(extended_silver))

    # Extend embeddings
    old_emb = np.load(emb_path)
    extended_emb = np.vstack([old_emb, new_emb])
    np.save(emb_path, extended_emb)
    logger.info("[%s] embeddings: %s → %s", cat, old_emb.shape, extended_emb.shape)

logger.info("DONE")
