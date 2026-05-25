"""Dump pasta.grain_type and pasta.cuisine_origin disagreements vs consensus gold."""
import sys
from pathlib import Path

import pandas as pd

for root in ['/home/miafrolov/Desktop/diploma',
             '/Users/miafrolov/Desktop/stuff/ai_attributes']:
    if Path(root).exists():
        sys.path.insert(0, root)
        PROJECT_ROOT = Path(root)
        break

from scripts.eval_v4_manual import predict_ml, predict_rules, norm_value
from scripts.build_gold_v4_wide import build_inputs_df
from scripts.eval_v4_consensus import _process_struct


def main():
    gold = pd.read_parquet(PROJECT_ROOT / 'datasets/processed/manual_gold_consensus.parquet')
    g = gold[(gold.category == 'pasta') & ~gold.disputed].copy()
    g['code'] = g['code'].astype(str)
    codes = set(g.code.unique())

    off_dir = Path.home() / 'off_work'
    if not off_dir.exists():
        off_dir = PROJECT_ROOT / 'datasets/raw'

    inputs = build_inputs_df(off_dir / 'pasta_off_full.parquet', codes)
    missing = codes - set(inputs.code.astype(str))
    if missing:
        import duckdb
        miss_sql = ','.join(f"'{c}'" for c in missing)
        con = duckdb.connect()
        extra = con.execute(f"""
            SELECT code, product_name, brands, ingredients_text, quantity,
                   categories_tags, labels_tags, traces_tags,
                   countries_tags, ingredients_analysis_tags,
                   nutriments
            FROM '{off_dir / 'food.parquet'}'
            WHERE CAST(code AS VARCHAR) IN ({miss_sql})
        """).fetchdf()
        if len(extra):
            extra = _process_struct(extra)
            inputs = pd.concat([inputs, extra], ignore_index=True)
    inputs['code'] = inputs['code'].astype(str)

    ml_preds = predict_ml(inputs, 'pasta_v4', prefix='pasta_v4')
    ml_preds['m_key'] = ml_preds.code.astype(str) + '|' + ml_preds.attr

    for attr in ['grain_type', 'cuisine_origin']:
        print(f'\n\n### pasta.{attr} ###')
        sub = g[g.attr == attr].copy()
        sub['m_key'] = sub.code + '|' + sub.attr
        sub = sub.merge(ml_preds[['m_key', 'ml_pred', 'ml_conf', 'ml_fired']],
                        on='m_key', how='left')
        sub = sub.merge(inputs[['code', 'product_name', 'ingredients_text',
                                'categories_tags']], on='code', how='left')
        sub['gold_n'] = sub['gold_value'].apply(norm_value)
        sub['ml_n'] = sub['ml_pred'].apply(norm_value)
        # Only cases where cascade fired (ml_fired=True), this is the conditional accuracy
        fired = sub[sub.ml_fired == True]
        wrong = fired[fired.gold_n != fired.ml_n]
        print(f'  fired: {len(fired)} | wrong: {len(wrong)}')
        from collections import Counter
        c = Counter()
        for _, r in wrong.iterrows():
            c[f'{r.gold_n} -> {r.ml_n}'] += 1
        for pair, n in c.most_common(10):
            print(f'    [{n:2d}] {pair}')
        print('  examples:')
        for _, r in wrong.head(30).iterrows():
            pn = (r['product_name'] or '?')[:50]
            ing = (r.get('ingredients_text') or '')[:80]
            cats = (r.get('categories_tags') or '')[:80]
            agree = r.get('agreement_ratio', 1.0)
            print(f'    [{r.code:13s}] {pn:50s} gold={r.gold_n:18s} ml={r.ml_n:14s} conf={r.ml_conf:.2f} agree={agree:.2f}')
            if cats: print(f'        cats: {cats}')
            if ing:  print(f'        ing:  {ing}')


if __name__ == '__main__':
    main()
