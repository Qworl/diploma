"""Build pure LLM-only gold v3 (без hybrid с silver).

Source: {cat}_relabel_google_gemini-2.5-flash.parquet (60 parallel workers,
Gemini Flash, off_grounded mode, ~$40 total для 57k products).

Output: {cat}_llm_gold_v3.parquet
Columns: code, attr, value, source='llm_gemini25flash'

Никакой silver. Никакой per-attr selection на gold (которая = overfitting на eval).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.common import MAIN_CATEGORIES, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

PROCESSED = Path(PROCESSED_DIR)


def _norm(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).lower().strip()
    if s in ("", "none", "null", "nan"):
        return None
    return s


def build_one(cat: str) -> pd.DataFrame:
    src = PROCESSED / "vm_relabel" / f"{cat}_relabel_google_gemini-2.5-flash.parquet"
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_parquet(src)
    df = df[df.parse_status == True].copy()
    df["code"] = df["code"].astype(str)
    rows = []
    for _, r in df.iterrows():
        try:
            parsed = json.loads(r.parsed_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if not parsed:
            continue
        for attr, val in parsed.items():
            v = _norm(val)
            if v is None:
                continue
            rows.append({
                "code": r.code, "attr": attr, "value": v,
                "source": "llm_gemini25flash",
            })
    return pd.DataFrame(rows)


def main():
    for cat in MAIN_CATEGORIES:
        df = build_one(cat)
        out = PROCESSED / f"{cat}_llm_gold_v3.parquet"
        df.to_parquet(out, index=False)
        logger.info("%s: %d (code, attr) rows → %s", cat, len(df), out)


if __name__ == "__main__":
    main()
