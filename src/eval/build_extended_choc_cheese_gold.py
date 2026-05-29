"""Sample 150 in-scope NEW codes for chocolate + cheeses to extend consensus gold.

Strategy:
- Read v5_relabel for cat, parse JSON labels, take parse_status=True
- Exclude codes already in consensus_manual (avoid duplicates)
- Join with OFF dump to confirm in-scope via categories_tags
- Stratify lightly by key attribute (chocolate_type for choc, milk_source for cheese)
- Output one parquet per cat: manual_eval_extension_{cat}.parquet
  with rows (category, code, attr, manual=None, product_name_v5)
- Consensus_manual_gold.py then runs LLM relabel on these codes.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np

for root in ['/home/miafrolov/Desktop/diploma',
             '/Users/miafrolov/Desktop/stuff/ai_attributes']:
    if Path(root).exists():
        sys.path.insert(0, root)
        PROJECT_ROOT = Path(root)
        break

RNG = np.random.default_rng(42)

CAT_VALID_TAGS = {
    'chocolate': {'en:chocolates', 'en:dark-chocolates', 'en:milk-chocolates',
                  'en:white-chocolates', 'en:filled-chocolates', 'en:pralines',
                  'en:chocolate-bars', 'en:chocolate-confectionery',
                  'en:chocolate-truffles', 'en:chocolate-candies'},
    'cheeses':   {'en:cheeses', 'en:fresh-cheeses', 'en:processed-cheese-products',
                  'en:cheese-spreads', 'en:grated-cheeses', 'en:shredded-cheeses',
                  'en:sliced-cheeses', 'en:hard-cheeses', 'en:soft-cheeses',
                  'en:blue-cheeses', 'en:semi-soft-cheeses'},
}

CAT_ATTRS = {
    'chocolate': ['chocolate_type', 'contains_nuts', 'chocolate_extra',
                  'is_organic', 'flavor_profile'],
    'cheeses':   ['milk_source', 'texture', 'country_of_origin',
                  'is_pdo', 'is_organic', 'is_ultra_processed', 'aging'],
}

QUOTAS = {
    'chocolate': {
        'milk':  45, 'dark': 45, 'white': 25, 'filled': 20,
        'praline': 10, 'cocoa_powder': 5,
    },
    'cheeses': {
        'cow':   60, 'goat':  25, 'sheep': 15, 'buffalo': 10,
        'mixed': 20, 'other': 10, None: 10,
    },
}

STRATIFY_KEY = {'chocolate': 'chocolate_type', 'cheeses': 'milk_source'}


def parse_json_col(series):
    def _p(x):
        if x is None or (isinstance(x, float) and pd.isna(x)) or not isinstance(x, str):
            return {}
        try: return json.loads(x)
        except: return {}
    return series.apply(_p)


def sample_cat(cat: str, target_total: int = 150) -> pd.DataFrame:
    v5 = pd.read_parquet(PROJECT_ROOT / f'datasets/processed/v5_relabel/{cat}_relabel_v5.parquet')
    v5 = v5[v5.parse_status == True].copy()
    v5['code'] = v5['code'].astype(str)
    print(f'\n=== {cat.upper()} ===')
    print(f'  v5 successful: {len(v5)}')

    # Excluded codes: those already in consensus_manual
    existing_path = PROJECT_ROOT / f'datasets/processed/consensus_manual/{cat}_consensus_qwen_qwen3.7-max.parquet'
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        existing_codes = set(existing.code.astype(str).unique())
        print(f'  excluding {len(existing_codes)} already in consensus_manual')
    else:
        existing_codes = set()

    v5 = v5[~v5.code.isin(existing_codes)]
    print(f'  candidate pool: {len(v5)}')

    parsed = parse_json_col(v5['parsed_json'])
    attrs_df = pd.json_normalize(parsed)
    attrs_df['code'] = v5['code'].values
    attrs_df['product_name_v5'] = v5['product_name'].values

    sk = STRATIFY_KEY[cat]
    quotas = QUOTAS[cat]
    print(f'  {sk} distribution in pool:')
    print(attrs_df[sk].value_counts(dropna=False).head(15).to_string())

    selected = []
    for value, target_n in quotas.items():
        pool = attrs_df[attrs_df[sk] == value] if value is not None else attrs_df[attrs_df[sk].isna()]
        n_avail = len(pool)
        n_take = min(target_n, n_avail)
        if n_take == 0:
            print(f'  [skip] {sk}={value}: 0 available')
            continue
        chosen = pool.sample(n=n_take, random_state=42)
        selected.append(chosen)
        print(f'  {sk}={str(value):15s}: {n_take}/{target_n} (pool={n_avail})')

    sample = pd.concat(selected).drop_duplicates(subset='code').reset_index(drop=True)

    # If we got fewer than target_total, top up randomly from remaining
    if len(sample) < target_total:
        remaining = attrs_df[~attrs_df.code.isin(sample.code)]
        n_extra = min(target_total - len(sample), len(remaining))
        if n_extra > 0:
            sample = pd.concat([sample, remaining.sample(n=n_extra, random_state=43)],
                               ignore_index=True)
            print(f'  topped up with {n_extra} random extras')

    print(f'  total selected: {len(sample)}')

    rows = []
    for _, r in sample.iterrows():
        for attr in CAT_ATTRS[cat]:
            rows.append({
                'category': cat, 'code': r['code'], 'attr': attr, 'manual': None,
                'product_name_v5': r['product_name_v5'],
            })
    return pd.DataFrame(rows)


def main():
    for cat in ['chocolate', 'cheeses']:
        ext = sample_cat(cat, target_total=150)
        out = PROJECT_ROOT / f'datasets/processed/manual_eval_extension_{cat}.parquet'
        ext.to_parquet(out, index=False)
        print(f'  saved: {out}  ({len(ext)} rows, {ext.code.nunique()} codes)')


if __name__ == '__main__':
    main()
