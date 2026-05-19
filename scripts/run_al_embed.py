"""P2-EXP11 Step 2b: Compute embeddings for new AL + control codes (torch only, no XGBoost).

Saves:
  datasets/processed/al_new_codes_embeddings.npy
  datasets/processed/al_new_codes_list.json

Usage:
    OMP_NUM_THREADS=1 python scripts/run_al_embed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE_ROOT))

PROCESSED = WORKTREE_ROOT / "datasets" / "processed"
MANUAL_LABEL = WORKTREE_ROOT / "datasets" / "manual_label"
OFF_PARQUET = WORKTREE_ROOT / "datasets" / "raw" / "en.openfoodfacts.org.products.parquet"

CATEGORIES = ["pasta", "chocolate", "cheeses"]

OUT_EMB = PROCESSED / "al_new_codes_embeddings.npy"
OUT_LIST = PROCESSED / "al_new_codes_list.json"


def _get_text(row: pd.Series) -> str:
    parts = []
    for col in ["product_name", "brands", "ingredients_text", "quantity"]:
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            parts.append(str(v).strip())
    return " ".join(parts)


def main():
    all_codes: list[str] = []
    for cat in CATEGORIES:
        al = pd.read_csv(MANUAL_LABEL / f"al_codes_{cat}.csv")["code"].astype(str).tolist()
        ctrl = pd.read_csv(MANUAL_LABEL / f"al_control_codes_{cat}.csv")["code"].astype(str).tolist()
        all_codes.extend(list(set(al) | set(ctrl)))

    all_codes = list(dict.fromkeys(all_codes))  # dedupe preserving order
    print(f"Total new codes across all cats: {len(all_codes)}")

    # Load product text from OFF
    cols = ["code", "product_name", "brands", "ingredients_text", "quantity"]
    off = pd.read_parquet(OFF_PARQUET, columns=cols)
    off["code"] = off["code"].astype(str)
    pool = off[off["code"].isin(all_codes)].copy().reset_index(drop=True)
    print(f"Found {len(pool)} products in OFF")

    # Reorder to match all_codes order
    pool_idx = pool.set_index("code")
    texts = []
    valid_codes = []
    for c in all_codes:
        if c in pool_idx.index:
            texts.append(_get_text(pool_idx.loc[c]))
            valid_codes.append(c)
        else:
            print(f"  WARN: code {c} not in OFF parquet, skipping")

    print(f"Computing embeddings for {len(texts)} codes...")
    from src.common import get_embeddings
    emb = get_embeddings(texts)
    print(f"Embeddings shape: {emb.shape}")

    np.save(OUT_EMB, emb)
    OUT_LIST.write_text(json.dumps(valid_codes))
    print(f"Saved embeddings to {OUT_EMB}")
    print(f"Saved code list to {OUT_LIST}")


if __name__ == "__main__":
    main()
