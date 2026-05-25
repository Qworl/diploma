"""Build gold v4 wide-format parquet (drop-in for silver_standard).

Source 1: datasets/processed/v5_relabel/{cat}_relabel_v5.parquet
  → parse_status==True, parse parsed_json into wide columns.
  Provides: semantic attrs (grain_type, pasta_shape, cuisine_origin, etc.).

Source 2: ~/off_work/{cat}_off_full.parquet (on VM) or
  datasets/raw/en.openfoodfacts.org.products.parquet (local).
  Provides: product_name, brands, categories_tags, labels_tags, traces_tags,
  ingredients_text, quantity, nutriments (flatten to *_100g columns).

Source 3: src.pipeline.off_labels.rules.TYPE_C_RULES applied to *_100g columns.
  Provides: nutri_score_grade, protein_class, fat_class, cocoa_percentage, etc.

Output: datasets/processed/{cat}_gold_v4_wide.parquet
"""
from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import duckdb

# Project root detection
for root in ['/home/miafrolov/Desktop/diploma',
             '/Users/miafrolov/Desktop/stuff/ai_attributes']:
    if Path(root).exists():
        sys.path.insert(0, root)
        PROJECT_ROOT = Path(root)
        break

from src.pipeline.off_labels.rules import _type_c_numeric, TYPE_C_RULES

# Per-cat TYPE_C attrs we want to populate from rules
TYPE_C_FOR_CAT = {
    'pasta': ['nutri_score_grade', 'protein_class'],
    'chocolate': ['cocoa_percentage', 'nutri_score_grade', 'protein_class'],
    'cheeses': ['fat_class'],
}

# nutri_score_grade is special — comes from OFF column nutriscore_grade directly
# (not a TYPE_C bucket). Handle separately.


def _pick_text(struct_arr, prefer=('main', 'en', 'fr', 'de', 'es', 'it')):
    if struct_arr is None:
        return None
    try:
        items = list(struct_arr) if not isinstance(struct_arr, list) else struct_arr
    except TypeError:
        return None
    by_lang = {}
    for it in items:
        if isinstance(it, dict):
            text = str(it.get('text', '') or '').strip()
            if text:
                by_lang[it.get('lang', '')] = text
    for p in prefer:
        if p in by_lang:
            return by_lang[p]
    return next(iter(by_lang.values())) if by_lang else None


def _flatten_nutriments(nut_arr):
    if nut_arr is None:
        return {}
    try:
        items = list(nut_arr) if not isinstance(nut_arr, list) else nut_arr
    except TypeError:
        return {}
    out = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get('name')
        per100 = it.get('100g')
        if name and per100 is not None:
            try:
                out[f'{name}_100g'] = float(per100)
            except (ValueError, TypeError):
                pass
    return out


def _safe_list(v):
    try:
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else None
    except Exception:
        return None


def _to_str(v):
    try:
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s if s else None


def build_inputs_df(off_path: Path, codes: set) -> pd.DataFrame:
    """Read filtered OFF parquet, return rich inputs for the given codes."""
    code_list_sql = ','.join(f"'{c}'" for c in codes)
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT code, product_name, brands, ingredients_text, quantity,
               categories_tags, labels_tags, traces_tags,
               countries_tags, ingredients_analysis_tags,
               nutriments
        FROM '{off_path}'
        WHERE CAST(code AS VARCHAR) IN ({code_list_sql})
    """).fetchdf()
    print(f'  loaded {len(df)} input rows from OFF')

    df['code'] = df['code'].astype(str)
    # flatten STRUCT[] text fields
    df['product_name'] = df['product_name'].apply(_pick_text)
    df['ingredients_text'] = df['ingredients_text'].apply(_pick_text)
    # tag lists → comma-joined strings (silver_standard convention)
    for col in ['categories_tags', 'labels_tags', 'traces_tags',
                'countries_tags', 'ingredients_analysis_tags']:
        df[col] = df[col].apply(lambda v: ','.join(_safe_list(v) or []) or None)
    # quantity, brands as plain strings
    df['quantity'] = df['quantity'].apply(_to_str)
    df['brands'] = df['brands'].apply(_to_str)

    # Flatten nutriments → *_100g columns; pick known ones
    nut_records = df['nutriments'].apply(_flatten_nutriments)
    for k in ['fat_100g', 'sugars_100g', 'proteins_100g', 'carbohydrates_100g',
             'alcohol_100g']:
        df[k] = nut_records.apply(lambda d: d.get(k))
    df = df.drop(columns=['nutriments'])
    return df


def apply_type_c(df: pd.DataFrame, cat: str) -> pd.DataFrame:
    """Apply TYPE_C_RULES to populate numeric bucket attrs in-place."""
    attrs = TYPE_C_FOR_CAT.get(cat, [])
    for attr in attrs:
        if attr == 'nutri_score_grade':
            # Special: derived from OFF nutriscore_grade or computed.
            # If column exists in inputs (from OFF nutriscore_grade), use it.
            # Otherwise leave NaN (sparse).
            if 'nutri_score_grade' not in df.columns:
                df['nutri_score_grade'] = None
            continue
        if attr not in TYPE_C_RULES:
            continue
        # Apply row-wise
        def _apply(row):
            row_d = row.to_dict()
            return _type_c_numeric(row_d, attr)
        df[attr] = df.apply(_apply, axis=1)
    return df


def attach_nutriscore_grade(df: pd.DataFrame, off_path: Path) -> pd.DataFrame:
    """Add nutriscore_grade column from OFF (lowercase, A-E)."""
    code_list_sql = ','.join(f"'{c}'" for c in df['code'].astype(str))
    con = duckdb.connect()
    ns = con.execute(f"""
        SELECT CAST(code AS VARCHAR) AS code, nutriscore_grade
        FROM '{off_path}'
        WHERE CAST(code AS VARCHAR) IN ({code_list_sql})
    """).fetchdf()
    ns['nutri_score_grade'] = ns['nutriscore_grade'].apply(
        lambda v: v.upper() if isinstance(v, str) and v.strip() else None
    )
    ns = ns[['code', 'nutri_score_grade']].drop_duplicates('code')
    return df.merge(ns, on='code', how='left')


def build_one(cat: str, v5_dir: Path, off_dir: Path, out_dir: Path,
              source_file: str | None = None) -> Path:
    print(f'\n=== Building gold v4 wide for {cat} ===')
    if source_file is None:
        source_file = f'{cat}_relabel_v5.parquet'
    v5_path = v5_dir / source_file
    v5 = pd.read_parquet(v5_path)
    v5 = v5[v5.parse_status == True].copy()
    v5['code'] = v5['code'].astype(str)
    print(f'  successful labels: {len(v5)}')

    # Parse JSON → wide attrs
    def _parse(x):
        if x is None: return {}
        if isinstance(x, float) and pd.isna(x): return {}
        if not isinstance(x, (str, bytes, bytearray)): return {}
        if not x: return {}
        try: return json.loads(x)
        except Exception: return {}
    parsed = v5['parsed_json'].apply(_parse)
    attrs_df = pd.json_normalize(parsed)
    attrs_df['code'] = v5['code'].values
    print(f'  LLM attrs: {[c for c in attrs_df.columns if c != "code"]}')

    # Load inputs from OFF
    off_path = off_dir / f'{cat}_off_full.parquet'
    inputs = build_inputs_df(off_path, set(v5['code'].astype(str)))

    # Merge: LLM attrs override anything in inputs
    df = inputs.merge(attrs_df, on='code', how='inner', suffixes=('_off', ''))
    print(f'  joined: {len(df)} rows')

    # Apply TYPE_C rules
    apply_type_c(df, cat)
    # Attach nutri_score_grade from OFF
    df = attach_nutriscore_grade(df, off_path)

    # Add brand column (silver convention: lowercase first brand)
    df['brand'] = df['brands'].apply(
        lambda b: b.split(',')[0].strip().lower() if b else None
    )

    out_path = out_dir / f'{cat}_gold_v4_wide.parquet'
    df.to_parquet(out_path, index=False)
    print(f'  wrote {out_path} ({len(df)} rows, {len(df.columns)} cols)')
    # Coverage report
    label_cols = list(attrs_df.columns) + TYPE_C_FOR_CAT.get(cat, [])
    print('  label coverage:')
    for c in label_cols:
        if c in df.columns and c != 'code':
            non_null = df[c].notna().sum()
            print(f'    {c:25s}: {non_null:6d} / {len(df)} ({100*non_null/len(df):.1f}%)')
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cats', nargs='+', default=['pasta', 'chocolate', 'cheeses'])
    p.add_argument('--source-version', default='v6',
                   help='which relabel output to read: v5 (v5_relabel/{cat}_relabel_v5.parquet) '
                        'or v6 (processed/{cat}_relabel_v4_deepseek_deepseek-v4-flash_nitro__v6.parquet)')
    args = p.parse_args()

    off_dir = Path.home() / 'off_work'
    if not off_dir.exists():
        off_dir = PROJECT_ROOT / 'datasets/raw'  # local fallback
    out_dir = PROJECT_ROOT / 'datasets/processed'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source_version == 'v5':
        src_dir = PROJECT_ROOT / 'datasets/processed/v5_relabel'
        src_file_for = lambda cat: f'{cat}_relabel_v5.parquet'
    elif args.source_version == 'v6':
        src_dir = PROJECT_ROOT / 'datasets/processed'
        src_file_for = lambda cat: f'{cat}_relabel_v4_deepseek_deepseek-v4-flash_nitro__v6.parquet'
    else:
        raise ValueError(f'unknown source-version: {args.source_version}')

    for cat in args.cats:
        build_one(cat, src_dir, off_dir, out_dir, source_file=src_file_for(cat))


if __name__ == '__main__':
    main()
