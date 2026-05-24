"""Consolidate multi-source LLM labels into unified gold standard v1.

Sources combined:
- silver_standard (OFF tag-derived columns)
- b3_full_gemini (Gemini 2.5 Flash, off_grounded mode, 2k-5k products per cat)
- b3_promptfix (3 parts × ~600 each)
- b3_promptfix_retry (improved prompt, 4 batches × ~3-6k each)
- b3_r2 (4 parts × ~2k each)
- gemini_validation (239 products on gold-overlapping subset)

Resolution policy per (code, attr):
- 1 source → trust it ('single')
- ≥2/3 majority → use majority ('consensus' if no disagreement, 'majority' otherwise)
- No majority → highest priority source wins ('priority', flagged as conflict)

Priority order (higher = more recent/refined): silver > gemini_validation > b3_r2 >
b3_promptfix_retry > b3_promptfix > b3_full.

Output: datasets/processed/{cat}_consolidated_gold_v1.parquet
Columns: code, attr, value, n_votes, n_unique, n_sources, confidence, has_conflict
"""
from __future__ import annotations

import glob
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from src.common import MAIN_CATEGORIES, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

PROCESSED = Path(PROCESSED_DIR)

SOURCES_PRIORITY = {
    # Freshly recomputed TYPE_C buckets from raw nutriments через current rules.py.
    # Высший приоритет: rule-based deterministic для numeric attrs (fat_class и т.д.).
    "silver_type_c_fresh": 100,
    "silver": 10,
    "gemini_validation": 9,
    "b3_r2": 8,
    "b3_promptfix_retry": 7,
    "b3_promptfix": 6,
    "b3_full": 5,
}

# Schema-defined attributes per category, used to identify which columns in
# silver_standard.parquet are actually attribute labels vs raw fields.
RAW_COLUMNS = frozenset({
    "code", "product_name", "brands", "categories_tags", "countries_tags",
    "labels_tags", "ingredients_text", "ingredients_analysis_tags",
    "traces_tags", "quantity", "fat_100g", "sugars_100g", "proteins_100g",
    "carbohydrates_100g", "alcohol_100g", "nutriscore_grade", "nova_group",
    "generic_name", "serving_size", "completeness", "image_url",
    "saturated-fat_100g", "salt_100g", "sodium_100g", "fiber_100g",
    "ingredients_tags",
})


def _norm(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in ("", "none", "null", "nan"):
        return None
    return s


def _extract_silver_long(cat: str) -> pd.DataFrame:
    """Silver labels stored as columns in silver_standard.parquet."""
    path = PROCESSED / f"{cat}_stratified_silver_standard.parquet"
    silver = pd.read_parquet(path)
    silver["code"] = silver["code"].astype(str)
    attr_cols = [c for c in silver.columns if c not in RAW_COLUMNS]
    rows = []
    for attr in attr_cols:
        sub = silver[["code", attr]].dropna(subset=[attr])
        for _, r in sub.iterrows():
            v = _norm(r[attr])
            if v is None:
                continue
            rows.append({"code": r.code, "attr": attr, "value": v,
                          "source": "silver", "priority": SOURCES_PRIORITY["silver"]})
    return pd.DataFrame(rows)


def _extract_gemini_long(path: Path, source_label: str, priority: int) -> pd.DataFrame:
    """Gemini labels stored as JSON in parsed_json column."""
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        if r.parsed_json is None:
            continue
        try:
            parsed = (json.loads(r.parsed_json)
                      if isinstance(r.parsed_json, str) else r.parsed_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if not parsed:
            continue
        for attr, val in parsed.items():
            v = _norm(val)
            if v is None:
                continue
            rows.append({"code": str(r.code), "attr": attr, "value": v,
                          "source": source_label, "priority": priority})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["code", "attr", "value", "source", "priority"])


def _gather_sources(cat: str) -> list[tuple[str, Path]]:
    """Return list of (source_label, path) for all gemini-labelled artifacts."""
    pairs: list[tuple[str, Path]] = []
    pairs += [("b3_full", Path(p))
              for p in glob.glob(str(PROCESSED / f"b3_full_gemini_{cat}.parquet"))]
    pairs += [("b3_promptfix", Path(p))
              for p in sorted(glob.glob(str(PROCESSED / f"b3_promptfix_gemini_{cat}_part*.parquet")))
              if "_retry" not in p]
    pairs += [("b3_promptfix_retry", Path(p))
              for p in sorted(glob.glob(str(PROCESSED / f"b3_promptfix_gemini_{cat}_*retry*.parquet")))]
    pairs += [("b3_r2", Path(p))
              for p in sorted(glob.glob(str(PROCESSED / f"b3_r2_gemini_{cat}_part*.parquet")))]
    val_path = PROCESSED / f"gemini_validation_{cat}.parquet"
    if val_path.exists():
        pairs.append(("gemini_validation", val_path))
    return pairs


def _resolve_one(code: str, attr: str, grp: pd.DataFrame) -> dict:
    """Apply majority + priority resolution to one (code, attr) group."""
    vals = grp["value"].tolist()
    n = len(vals)
    counter = Counter(vals)
    most_common_val, most_common_count = counter.most_common(1)[0]
    n_sources = grp["source"].nunique()

    if n == 1:
        verdict, conflict, confidence = vals[0], False, "single"
    elif most_common_count / n >= 2 / 3:
        verdict = most_common_val
        conflict = len(counter) > 1
        confidence = "majority" if conflict else "consensus"
    else:
        top = grp.loc[grp["priority"].idxmax()]
        verdict, conflict, confidence = top["value"], True, "priority"

    return {
        "code": code, "attr": attr, "value": verdict,
        "n_votes": n, "n_unique": len(counter), "n_sources": n_sources,
        "confidence": confidence, "has_conflict": conflict,
    }


def _load_type_c_fresh(cat: str) -> pd.DataFrame:
    """Load recomputed TYPE_C values (priority=100 silver_type_c_fresh source)."""
    path = PROCESSED / f"{cat}_silver_type_c_fresh.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["code", "attr", "value", "source", "priority"])
    df = pd.read_parquet(path)
    return df


def consolidate_one(cat: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build consolidated gold for one category. Returns (resolved, long_table)."""
    parts = [_extract_silver_long(cat)]
    for src_name, path in _gather_sources(cat):
        df_long = _extract_gemini_long(path, src_name, SOURCES_PRIORITY[src_name])
        if len(df_long) > 0:
            parts.append(df_long)
            logger.info("  %s (%s): %d rows", src_name, path.name, len(df_long))

    # TYPE_C fresh recompute — rule-based deterministic, hard override
    fresh = _load_type_c_fresh(cat)
    if len(fresh) > 0:
        parts.append(fresh)
        logger.info("  silver_type_c_fresh: %d rows (hard override for TYPE_C attrs)",
                    len(fresh))

    full_long = pd.concat(parts, ignore_index=True)

    # Hard override: для (code, attr) пар где есть silver_type_c_fresh, удалить
    # все остальные источники — fresh value is rule-based ground truth.
    fresh_keys = set(zip(fresh["code"], fresh["attr"])) if len(fresh) > 0 else set()
    if fresh_keys:
        mask = full_long.apply(
            lambda r: (r["code"], r["attr"]) in fresh_keys and r["source"] != "silver_type_c_fresh",
            axis=1,
        )
        n_removed = mask.sum()
        full_long = full_long[~mask].reset_index(drop=True)
        logger.info("  removed %d non-fresh rows for %d (code, attr) pairs that have TYPE_C fresh",
                    n_removed, len(fresh_keys))

    logger.info("  total long-format rows: %d, unique (code, attr): %d",
                len(full_long), full_long.groupby(["code", "attr"]).ngroups)

    rows = []
    for (code, attr), grp in full_long.groupby(["code", "attr"]):
        rows.append(_resolve_one(code, attr, grp))
    resolved = pd.DataFrame(rows)
    return resolved, full_long


def main():
    for cat in MAIN_CATEGORIES:
        logger.info("=== %s ===", cat.upper())
        resolved, long_table = consolidate_one(cat)

        long_path = PROCESSED / f"{cat}_labels_all_sources_long.parquet"
        long_table.to_parquet(long_path, index=False)
        logger.info("  saved long table: %s (%d rows)", long_path, len(long_table))

        out_path = PROCESSED / f"{cat}_consolidated_gold_v1.parquet"
        resolved.to_parquet(out_path, index=False)
        logger.info("  saved resolved gold: %s (%d (code, attr) pairs)", out_path, len(resolved))

        logger.info("  confidence: %s", resolved.confidence.value_counts().to_dict())
        logger.info("  has_conflict: %d (%.1f%%)",
                    resolved.has_conflict.sum(), 100 * resolved.has_conflict.mean())


if __name__ == "__main__":
    main()
