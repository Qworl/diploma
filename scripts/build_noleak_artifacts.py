"""Build noleak train artifacts: filtered gold_v4_wide + embeddings.

For each cat in {pasta, chocolate, cheeses}:
- Drops ALL test codes (manual_eval_per_product + manual_eval_extension_{cat}) from train.
- Saves as `{cat}_gold_v4_wide_noleak.parquet` + `{cat}_v4_embeddings_noleak.npy`.
- Embedding indices stay aligned with filtered gold rows (same positional order).
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


def gather_test_codes(cat: str) -> set:
    """Codes that MUST be excluded from ML train: all manual + extension + consensus codes."""
    codes = set()
    # Original manual eval (50 per cat)
    m = pd.read_parquet(PROJECT_ROOT / 'datasets/processed/manual_eval_per_product.parquet')
    codes |= set(m[m.category == cat].code.astype(str).unique())
    # Extension files (chocolate/cheeses had explicit ones)
    for fname in [f'manual_eval_extension_{cat}.parquet', 'manual_eval_extended.parquet']:
        path = PROJECT_ROOT / f'datasets/processed/{fname}'
        if path.exists():
            ext = pd.read_parquet(path)
            ext_codes = ext[ext.category == cat].code.astype(str).unique() if 'category' in ext.columns else []
            codes |= set(ext_codes)
    # Whatever ended up in consensus gold
    g = pd.read_parquet(PROJECT_ROOT / 'datasets/processed/manual_gold_consensus.parquet')
    codes |= set(g[g.category == cat].code.astype(str).unique())
    return codes


def main():
    for cat in ['pasta', 'chocolate', 'cheeses']:
        gold_path = PROJECT_ROOT / f'datasets/processed/{cat}_gold_v4_wide.parquet'
        emb_path = PROJECT_ROOT / f'datasets/processed/{cat}_v4_embeddings.npy'
        gold_noleak = PROJECT_ROOT / f'datasets/processed/{cat}_gold_v4_wide_noleak.parquet'
        emb_noleak = PROJECT_ROOT / f'datasets/processed/{cat}_v4_embeddings_noleak.npy'

        df = pd.read_parquet(gold_path)
        df['code'] = df['code'].astype(str)
        emb = np.load(emb_path)
        if len(df) != len(emb):
            print(f'!!! {cat}: row mismatch gold={len(df)} emb={len(emb)}')
            continue

        test_codes = gather_test_codes(cat)
        print(f'\n{cat}:')
        print(f'  full gold rows: {len(df)}')
        print(f'  test codes to drop: {len(test_codes)}')
        keep_mask = ~df['code'].isin(test_codes)
        n_drop = (~keep_mask).sum()
        print(f'  dropping rows: {n_drop} ({100*n_drop/len(df):.1f}%)')

        df_clean = df[keep_mask].reset_index(drop=True)
        emb_clean = emb[keep_mask.values]
        print(f'  noleak gold rows: {len(df_clean)}')
        print(f'  noleak emb shape: {emb_clean.shape}')

        df_clean.to_parquet(gold_noleak, index=False)
        np.save(emb_noleak, emb_clean)
        print(f'  saved: {gold_noleak.name}')
        print(f'  saved: {emb_noleak.name}')


if __name__ == '__main__':
    main()
