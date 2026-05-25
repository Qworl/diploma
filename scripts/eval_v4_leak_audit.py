"""Honest eval: split headline into LEAK vs CLEAN test codes.

LEAK codes = present in {cat}_gold_v4_wide.parquet (ML train) AND in gold.
CLEAN codes = only in gold (never seen by ML).

Reports headline for both subsets separately. The CLEAN number is the honest
out-of-sample accuracy; the LEAK number measures memorization.
"""
from __future__ import annotations
import sys, json
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
from scripts.eval_v4_consensus_clean import is_in_scope, CAT_VALID_TAGS


def eval_cat_split(cat, gold_df, off_dir, leak_codes):
    print(f'\n=== {cat.upper()} ===')
    g = gold_df[(gold_df.category == cat) & (~gold_df.disputed)].copy()
    g['code'] = g['code'].astype(str)
    if cat == 'cheeses':
        g['gold_value'] = g['gold_value'].replace({'semi_soft': 'soft'})
    codes = set(g.code.unique())

    inputs = build_inputs_df(off_dir / f'{cat}_off_full.parquet', codes)
    missing = codes - set(inputs.code.astype(str))
    if missing:
        import duckdb
        miss_sql = ','.join(f"'{c}'" for c in missing)
        con = duckdb.connect()
        extra = con.execute(f"""
            SELECT code, product_name, brands, ingredients_text, quantity,
                   categories_tags, labels_tags, traces_tags,
                   countries_tags, ingredients_analysis_tags, nutriments
            FROM '{off_dir / 'food.parquet'}'
            WHERE CAST(code AS VARCHAR) IN ({miss_sql})
        """).fetchdf()
        if len(extra):
            extra = _process_struct(extra)
            inputs = pd.concat([inputs, extra], ignore_index=True)
    inputs['code'] = inputs['code'].astype(str)
    inputs['in_scope'] = inputs['categories_tags'].apply(
        lambda s: is_in_scope(s, CAT_VALID_TAGS[cat])
    )

    import os as _os
    cat_v4 = f'{cat}_v4'
    candidates = [f'{cat}_v4_mpnet_tfidf_noleak', f'{cat}_v4_mpnet_tfidf',
                  f'{cat}_v4_mpnet', cat_v4]
    override = _os.environ.get('EVAL_MODEL_SUFFIX')
    if override:
        candidates = [f'{cat}_v4{override}'] + candidates
    use_prefix = next((p for p in candidates if _os.path.exists(
        f'{PROJECT_ROOT}/models/{p}_thresholds.pkl')), cat_v4)
    print(f'  using prefix: {use_prefix}')
    ml_preds = predict_ml(inputs, cat_v4, prefix=use_prefix)
    rule_preds = predict_rules(inputs)
    ml_preds['m_key'] = ml_preds.code.astype(str) + '|' + ml_preds.attr
    rule_preds['m_key'] = rule_preds.code.astype(str) + '|' + rule_preds.attr
    g['m_key'] = g.code + '|' + g.attr
    g = g.merge(inputs[['code', 'in_scope']], on='code', how='left')
    merged = g.merge(ml_preds[['m_key', 'ml_pred', 'ml_conf', 'ml_fired']], on='m_key', how='left')
    merged = merged.merge(rule_preds[['m_key', 'rule_pred', 'rule_tier']], on='m_key', how='left')

    def _cascade(row):
        has_rule = (row['rule_pred'] is not None
                    and not (isinstance(row['rule_pred'], float) and pd.isna(row['rule_pred'])))
        tier = row.get('rule_tier')
        if has_rule and tier == 'high':
            return ('rule_h', row['rule_pred'])
        if row['ml_fired'] is True:
            return ('ml', row['ml_pred'])
        if has_rule and tier == 'low':
            return ('rule_l', row['rule_pred'])
        return ('fallback', None)
    res = merged.apply(_cascade, axis=1)
    merged['cascade_source'] = [c[0] for c in res]
    merged['cascade_pred'] = [c[1] for c in res]
    merged['is_leak'] = merged['code'].isin(leak_codes)

    def _stats(df):
        gn = df['gold_value'].apply(norm_value)
        pn = df['cascade_pred'].apply(norm_value)
        valid = gn.notna() & pn.notna()
        return int(valid.sum()), int((gn == pn)[valid].sum()), int(len(df))

    rows = []
    print(f'  {"attr":22s}  {"clean_n":>7s} {"clean_acc":>9s}  {"leak_n":>7s} {"leak_acc":>9s}  {"delta":>5s}')
    print('  ' + '-' * 80)
    for attr in sorted(merged.attr.unique()):
        sub_in = merged[(merged.attr == attr) & (merged.in_scope == True)]
        sub_clean = sub_in[~sub_in.is_leak]
        sub_leak = sub_in[sub_in.is_leak]
        c_cov, c_cor, c_tot = _stats(sub_clean)
        l_cov, l_cor, l_tot = _stats(sub_leak)
        c_acc = 100*c_cor/c_cov if c_cov > 0 else float('nan')
        l_acc = 100*l_cor/l_cov if l_cov > 0 else float('nan')
        delta = l_acc - c_acc if (c_cov > 0 and l_cov > 0) else float('nan')
        print(f'  {attr:22s}  {c_cov:>7d}  {c_acc:>8.1f}%  {l_cov:>7d}  {l_acc:>8.1f}%  {delta:>+5.1f}')
        rows.append({'cat': cat, 'attr': attr,
                     'clean_n': c_cov, 'clean_correct': c_cor, 'clean_acc': c_acc/100,
                     'leak_n': l_cov, 'leak_correct': l_cor, 'leak_acc': l_acc/100})

    # Cat totals
    sub_clean = merged[(merged.in_scope == True) & (~merged.is_leak)]
    sub_leak = merged[(merged.in_scope == True) & (merged.is_leak)]
    c_cov, c_cor, _ = _stats(sub_clean)
    l_cov, l_cor, _ = _stats(sub_leak)
    c_acc = 100*c_cor/c_cov if c_cov > 0 else float('nan')
    l_acc = 100*l_cor/l_cov if l_cov > 0 else float('nan')
    print(f'  {"OVERALL":22s}  {c_cov:>7d}  {c_acc:>8.1f}%  {l_cov:>7d}  {l_acc:>8.1f}%  {l_acc-c_acc:>+5.1f}')
    return rows


def main():
    # Find leak codes
    leak_per_cat = {}
    for cat in ['pasta', 'chocolate', 'cheeses']:
        tr = pd.read_parquet(PROJECT_ROOT / f'datasets/processed/{cat}_gold_v4_wide.parquet')
        leak_per_cat[cat] = set(tr.code.astype(str).unique())

    gold = pd.read_parquet(PROJECT_ROOT / 'datasets/processed/manual_gold_consensus.parquet')
    off_dir = Path.home() / 'off_work'
    if not off_dir.exists():
        off_dir = PROJECT_ROOT / 'datasets/raw'

    all_rows = []
    for cat in ['pasta', 'chocolate', 'cheeses']:
        all_rows.extend(eval_cat_split(cat, gold, off_dir, leak_per_cat[cat]))

    df = pd.DataFrame(all_rows)
    out = PROJECT_ROOT / 'datasets/processed/v4_eval_leak_audit.parquet'
    df.to_parquet(out, index=False)
    c_cov = df.clean_n.sum(); c_cor = df.clean_correct.sum()
    l_cov = df.leak_n.sum(); l_cor = df.leak_correct.sum()
    print(f'\n=== GRAND ===')
    print(f'  CLEAN (out-of-sample, honest):  {c_cor}/{c_cov} ({100*c_cor/max(c_cov,1):.1f}%)')
    print(f'  LEAK (memorization):            {l_cor}/{l_cov} ({100*l_cor/max(l_cov,1):.1f}%)')
    print(f'  Δ (leak − clean):               {100*l_cor/max(l_cov,1) - 100*c_cor/max(c_cov,1):+.1f}pp')
    print(f'  saved: {out}')


if __name__ == '__main__':
    main()
