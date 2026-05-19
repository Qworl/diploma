"""Populate OFF cache from local OFF parquet dump (no API hits).

For each code: build a flat product dict matching what off_fetcher writes,
save to {cache_dir}/{code}.json. Skip if already cached.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

NUTRIMENT_COLS = [
    "fat_100g", "proteins_100g", "carbohydrates_100g", "energy-kcal_100g",
    "sugars_100g", "fiber_100g", "salt_100g", "sodium_100g", "alcohol_100g",
]
TAG_COLS = [
    "categories_tags", "labels_tags", "ingredients_tags", "countries_tags",
    "traces_tags",
]
TEXT_COLS = [
    "product_name", "generic_name", "brands", "quantity",
    "ingredients_text", "serving_size", "image_url",
]


def _to_list(s):
    if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
        return []
    return [t.strip() for t in str(s).split(",") if t.strip()]


def _to_text(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s)


def _nutriments(row) -> dict:
    out = {}
    for col in NUTRIMENT_COLS:
        if col not in row.index:
            continue
        v = row[col]
        if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
            continue
        try:
            out[col] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codes-file", required=True, type=Path)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--parquet",
                   default="datasets/raw/en.openfoodfacts.org.products.parquet",
                   type=Path)
    args = p.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    codes = []
    with args.codes_file.open() as f:
        for line in f:
            c = line.strip()
            if c:
                codes.append(c)

    # Pre-filter parquet — much faster than per-code lookup
    print(f"Loading {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    df["code"] = df["code"].astype(str)
    df = df[df["code"].isin(set(codes))].copy()
    print(f"Found {len(df)} rows for {len(codes)} requested codes")

    indexed = df.set_index("code")

    n_new = 0
    n_skip = 0
    n_missing = 0
    for code in codes:
        out_path = args.cache_dir / f"{code}.json"
        if out_path.exists():
            n_skip += 1
            continue
        if code not in indexed.index:
            n_missing += 1
            continue
        row = indexed.loc[code]
        product = {"code": code}
        for col in TEXT_COLS:
            if col in row.index:
                product[col] = _to_text(row[col])
        for col in TAG_COLS:
            if col in row.index:
                product[col] = _to_list(row[col])
        nut = _nutriments(row)
        if nut:
            product["nutriments"] = nut
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(product, f, ensure_ascii=False)
        n_new += 1

    print(f"cached_new={n_new}  already_existed={n_skip}  not_in_parquet={n_missing}")


if __name__ == "__main__":
    main()
