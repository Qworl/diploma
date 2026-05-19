"""Extend {cat}_stratified_silver_standard.parquet + _embeddings.npy with B3 codes.
Then trainer can include them.
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

BASE = Path("/Users/miafrolov/Desktop/stuff/ai_attributes/.worktrees/phase2-recomputes-on-v2-gold")
PROCESSED = BASE / "datasets" / "processed"
B3_CACHE_DIRS = [
    BASE / "datasets" / "manual_label" / "off_cache_b3_full",
    BASE / "datasets" / "manual_label" / "off_cache_b3_r2",
]

CATS = ["pasta", "chocolate", "cheeses"]

# Load sentence transformer
from sentence_transformers import SentenceTransformer
logger.info("Loading SentenceTransformer...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def load_b3_text(code: str) -> dict | None:
    """Try cache_dirs in order, return partner-input dict or None."""
    for cache_dir in B3_CACHE_DIRS:
        p = cache_dir / f"{code}.json"
        if p.exists():
            data = json.load(open(p, encoding="utf-8"))
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


# Gather all B3 codes per cat from gemini parquets
import glob
for cat in CATS:
    files = sorted(glob.glob(str(PROCESSED / f"b3_full_gemini_{cat}*.parquet"))
                   + glob.glob(str(PROCESSED / f"b3_r2_gemini_{cat}*.parquet")))
    all_codes = set()
    for f in files:
        df = pd.read_parquet(f, columns=["code"])
        all_codes.update(df["code"].astype(str).tolist())
    logger.info("[%s] B3 unique codes: %d", cat, len(all_codes))

    silver_path = PROCESSED / f"{cat}_stratified_silver_standard.parquet"
    silver = pd.read_parquet(silver_path)
    silver["code"] = silver["code"].astype(str)
    silver_codes = set(silver["code"])
    new_codes = sorted(all_codes - silver_codes)
    logger.info("[%s] NEW (not in silver): %d", cat, len(new_codes))

    # Load partner text per new code
    new_rows = []
    for code in new_codes:
        row = load_b3_text(code)
        if row is not None:
            new_rows.append(row)
    logger.info("[%s] Loaded partner-text for %d/%d codes", cat, len(new_rows), len(new_codes))

    # Compute embeddings for new codes
    texts = [build_text(r) for r in new_rows]
    logger.info("[%s] Encoding %d texts...", cat, len(texts))
    new_emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    logger.info("[%s] Embeddings shape: %s", cat, new_emb.shape)

    # Append to silver parquet
    new_silver = pd.DataFrame(new_rows)
    for col in silver.columns:
        if col not in new_silver.columns:
            new_silver[col] = None
    new_silver = new_silver[silver.columns]
    extended_silver = pd.concat([silver, new_silver], ignore_index=True)
    extended_silver.to_parquet(silver_path, index=False)
    logger.info("[%s] silver: %d → %d rows", cat, len(silver), len(extended_silver))

    # Append to embeddings
    emb_path = PROCESSED / f"{cat}_stratified_embeddings.npy"
    old_emb = np.load(emb_path)
    extended_emb = np.vstack([old_emb, new_emb])
    np.save(emb_path, extended_emb)
    logger.info("[%s] embeddings: %s → %s", cat, old_emb.shape, extended_emb.shape)

logger.info("DONE")
