"""Sample NEW codes per category from 300k OFF parquet pool for B3 expansion.

Output:
  datasets/manual_label/b3_codes_{cat}.csv — minimal CSV (code column)
  datasets/manual_label/off_cache_b3/{code}.json — OFF facts per code,
    in same format as off_fetcher.py output (so direct_llm_v2 can use it)
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout, force=True,
)
logger = logging.getLogger(__name__)

WORKTREE_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = WORKTREE_ROOT / "datasets" / "processed"
LABEL_DIR = WORKTREE_ROOT / "datasets" / "manual_label"
CHUNK_DIR = WORKTREE_ROOT / "datasets" / "raw" / "off_jsonl_chunks"

CATEGORY_TAGS = {
    "pasta":     ["en:pastas", "en:noodles", "en:lasagna", "en:fresh-pastas", "en:dried-pastas"],
    "chocolate": ["en:chocolates", "en:dark-chocolates", "en:milk-chocolates", "en:white-chocolates"],
    "cheeses":   ["en:cheeses"],
}


def load_off_pool() -> pd.DataFrame:
    files = sorted(glob.glob(str(CHUNK_DIR / "*.parquet")))
    if not files:
        raise SystemExit(f"No OFF chunks in {CHUNK_DIR}")
    dfs = [pd.read_parquet(p) for p in files]
    df = pd.concat(dfs, ignore_index=True)
    df["code"] = df["code"].astype(str)
    logger.info("Loaded %d products from %d OFF chunks", len(df), len(files))
    return df


def existing_codes() -> set[str]:
    """All codes already in train/test/gold so we don't duplicate."""
    out: set[str] = set()
    for f in PROCESSED_DIR.glob("*stratified*.parquet"):
        try:
            df = pd.read_parquet(f, columns=["code"])
            out.update(df["code"].astype(str))
        except Exception:
            pass
    # Also v2 gold
    p = PROCESSED_DIR / "consensus_gold_v2_expanded.parquet"
    if p.exists():
        out.update(pd.read_parquet(p, columns=["code"])["code"].astype(str))
    return out


def row_to_off_json(row: pd.Series) -> dict:
    """Convert parquet row to OFF API response format that direct_llm_v2 expects."""
    def split_tags(val):
        if not val or pd.isna(val):
            return []
        return [t for t in str(val).split("|") if t]

    def parse_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    nutriments = {}
    for nk in ("fat_100g", "saturated-fat_100g", "carbohydrates_100g",
               "sugars_100g", "proteins_100g", "salt_100g", "fiber_100g",
               "energy-kcal_100g"):
        v = parse_float(row.get(nk))
        if v is not None:
            nutriments[nk] = v

    return {
        "code": str(row.get("code", "")),
        "status": 1,
        "product": {
            "code": str(row.get("code", "")),
            "product_name": str(row.get("product_name", "") or ""),
            "brands": str(row.get("brands", "") or ""),
            "ingredients_text": str(row.get("ingredients_text", "") or ""),
            "quantity": str(row.get("quantity", "") or ""),
            "categories": str(row.get("categories", "") or ""),
            "categories_tags": split_tags(row.get("categories_tags")),
            "labels_tags": split_tags(row.get("labels_tags")),
            "ingredients_analysis_tags": split_tags(row.get("ingredients_analysis_tags")),
            "traces_tags": split_tags(row.get("traces_tags")),
            "countries_tags": split_tags(row.get("countries_tags")),
            "allergens_tags": split_tags(row.get("allergens_tags")),
            "additives_tags": split_tags(row.get("additives_tags")),
            "lang": str(row.get("lang", "") or ""),
            "manufacturing_places": str(row.get("manufacturing_places", "") or ""),
            "nutriscore_grade": str(row.get("nutriscore_grade", "") or ""),
            "nova_group": str(row.get("nova_group", "") or ""),
            "ecoscore_grade": str(row.get("ecoscore_grade", "") or ""),
            "nutriments": nutriments,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=500,
                    help="How many new codes to sample per category")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = LABEL_DIR / "off_cache_b3"
    cache_dir.mkdir(parents=True, exist_ok=True)

    off = load_off_pool()
    existing = existing_codes()
    logger.info("Existing codes (train/test/gold): %d", len(existing))

    tag_col = off["categories_tags"].fillna("")

    for cat, tags in CATEGORY_TAGS.items():
        pattern = "|".join(tags)
        mask = tag_col.str.contains(pattern, regex=True, na=False)
        cat_df = off[mask].copy()
        cat_df["code"] = cat_df["code"].astype(str)
        new_df = cat_df[~cat_df["code"].isin(existing)]
        logger.info("%s: %d total, %d NEW", cat, len(cat_df), len(new_df))

        # Filter rows with at least product_name + brand non-empty
        valid = new_df[
            (new_df["product_name"].astype(str).str.strip() != "")
            & (new_df["brands"].astype(str).str.strip() != "")
        ]
        logger.info("%s: %d NEW with name+brand", cat, len(valid))

        # Sample
        n_sample = min(args.per_cat, len(valid))
        sample = valid.sample(n=n_sample, random_state=args.seed)

        # Write minimal CSV
        csv_path = LABEL_DIR / f"b3_codes_{cat}.csv"
        sample[["code"]].to_csv(csv_path, index=False)
        logger.info("%s: wrote %d codes to %s", cat, n_sample, csv_path)

        # Write per-code OFF JSON
        n_written = 0
        for _, row in sample.iterrows():
            code = str(row["code"])
            json_path = cache_dir / f"{code}.json"
            if json_path.exists():
                continue
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(row_to_off_json(row), f, ensure_ascii=False)
            n_written += 1
        logger.info("%s: wrote %d new JSON files to %s", cat, n_written, cache_dir)


if __name__ == "__main__":
    main()
