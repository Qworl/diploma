"""Sample 200 in-scope pasta codes for consensus relabel.

Strategy:
- Stratify by grain_type (from v6 labels): 30% wheat, 20% rice, 10% legume,
  10% potato (gnocchi), 10% buckwheat, 10% corn, 10% other/mixed
- Stratify by brand: deduplicate brands so no single brand dominates
- Exclude codes already in manual_eval_per_product (avoid double-counting)
- Output: list of codes for consensus_manual_gold.py to process
"""
from __future__ import annotations

import sys
import json
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


def main():
    # Read v6 relabel for pasta
    v6 = pd.read_parquet(
        PROJECT_ROOT / 'datasets/processed/pasta_relabel_v4_deepseek_deepseek-v4-flash_nitro__v6.parquet'
    )
    v6 = v6[v6.parse_status == True].copy()
    v6['code'] = v6['code'].astype(str)
    print(f'v6 pasta: {len(v6)} successful labels')

    # Parse JSON → wide
    def _parse(x):
        if x is None or (isinstance(x, float) and pd.isna(x)) or not isinstance(x, str):
            return {}
        try: return json.loads(x)
        except: return {}
    parsed = v6['parsed_json'].apply(_parse)
    attrs_df = pd.json_normalize(parsed)
    attrs_df['code'] = v6['code'].values
    attrs_df['product_name'] = v6['product_name'].values

    # Exclude codes already in manual_eval
    manual = pd.read_parquet(
        PROJECT_ROOT / 'datasets/processed/manual_eval_per_product.parquet'
    )
    manual_pasta_codes = set(
        manual[manual.category == 'pasta']['code'].astype(str).unique()
    )
    print(f'  excluding {len(manual_pasta_codes)} already-in-manual-eval codes')
    attrs_df = attrs_df[~attrs_df.code.isin(manual_pasta_codes)]
    print(f'  candidate pool: {len(attrs_df)}')

    # We need to also join with categories_tags to confirm in-scope
    # → use the OFF dump column. But it's on VM. Approximate: trust v6 (already filtered en:pastas)
    # If running on VM, join. For local, just use the v6 sample.

    # Stratified sampling by grain_type
    print('\n  grain_type distribution in pool:')
    print(attrs_df.grain_type.value_counts(dropna=False).head(10).to_string())

    # Target: 200 codes — over-sample minority classes
    quotas = {
        'wheat':     60,   # most common (~80% of pool)
        'rice':      30,
        'legume':    25,   # red lentil etc.
        'potato':    25,   # gnocchi
        'buckwheat': 15,
        'corn':      15,
        'spelt':     10,
        'oat':        5,
        'mixed':      8,
        'other':      7,
    }

    selected = []
    for grain, target_n in quotas.items():
        pool = attrs_df[attrs_df.grain_type == grain]
        n_avail = len(pool)
        n_take = min(target_n, n_avail)
        if n_take == 0:
            print(f'  [skip] grain={grain}: 0 available')
            continue
        chosen = pool.sample(n=n_take, random_state=42)
        selected.append(chosen)
        print(f'  grain={grain:10s}: {n_take}/{target_n} (pool={n_avail})')

    sample = pd.concat(selected).drop_duplicates(subset='code').reset_index(drop=True)
    print(f'\n  total selected: {len(sample)}')

    # Save as fake manual_eval_per_product-style table for consensus_manual_gold.py
    # The script reads category, code, attr — we provide all attrs from current
    # schema; the consensus relabel will overwrite.
    rows = []
    for _, r in sample.iterrows():
        for attr in ['grain_type', 'pasta_shape', 'is_filled', 'is_organic',
                     'is_gluten_free', 'is_vegan', 'cuisine_origin']:
            rows.append({
                'category': 'pasta', 'code': r['code'], 'attr': attr,
                'manual': None,  # to be filled by consensus
                'product_name_v6': r['product_name'],
            })
    extra_gold_input = pd.DataFrame(rows)

    # Merge with existing manual gold to form combined
    combined = pd.concat(
        [manual, extra_gold_input[['category', 'code', 'attr', 'manual']]],
        ignore_index=True,
    )
    out = PROJECT_ROOT / 'datasets/processed/manual_eval_extended.parquet'
    combined.to_parquet(out, index=False)
    print(f'\nSaved extended manual_eval to {out}')
    print(f'  total rows: {len(combined)} (original {len(manual)} + extra {len(extra_gold_input)})')
    print(f'  unique pasta codes: {combined[combined.category == "pasta"]["code"].nunique()}')


if __name__ == '__main__':
    main()
