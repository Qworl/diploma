"""Eval v4 cascade vs consensus gold, filtering out mislabeled non-cat products.

For each category, only keeps codes where categories_tags actually contains
the cat-specific tags. Reports both raw and clean accuracies.
"""
from __future__ import annotations
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


# Category tags that signal a real "in-scope" product for each cat.
# Codes whose categories_tags lack ANY of these are considered mislabeled.
CAT_VALID_TAGS = {
    'pasta':     {'en:pastas', 'en:noodles', 'en:asian-noodles', 'en:stuffed-pastas',
                  'en:gnocchi', 'en:ravioli', 'en:tortellini', 'en:lasagna',
                  'en:cannelloni', 'en:filled-pastas', 'en:dry-pastas',
                  'en:fresh-pastas', 'en:pasta-dishes', 'en:macaronis',
                  'en:spaghettis', 'en:fusilli', 'en:tagliatelles'},
    'chocolate': {'en:chocolates', 'en:dark-chocolates', 'en:milk-chocolates',
                  'en:white-chocolates', 'en:filled-chocolates', 'en:pralines',
                  'en:chocolate-bars', 'en:chocolate-confectionery',
                  'en:chocolate-truffles', 'en:chocolate-candies'},
    'cheeses':   {'en:cheeses', 'en:fresh-cheeses', 'en:processed-cheese-products',
                  'en:cheese-spreads', 'en:grated-cheeses', 'en:shredded-cheeses',
                  'en:sliced-cheeses', 'en:hard-cheeses', 'en:soft-cheeses',
                  'en:blue-cheeses', 'en:semi-soft-cheeses'},
}


def is_in_scope(cats_str, valid_tags):
    if not cats_str or not isinstance(cats_str, str):
        return False
    tags = set(t.strip() for t in cats_str.split(',') if t.strip())
    return bool(tags & valid_tags)


def eval_cat(cat, gold_df, off_dir):
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
    n_in_scope = inputs.in_scope.sum()
    print(f'  total inputs: {len(inputs)} | in-scope (real {cat}): {n_in_scope} '
          f'({100*n_in_scope/len(inputs):.0f}%)')

    cat_v4 = f'{cat}_v4'
    # Prefer noleak (out-of-sample correct) → MPNet+TFIDF → MPNet → legacy.
    import os as _os
    candidates = [
        f'{cat}_v4_mpnet_tfidf_noleak',
        f'{cat}_v4_mpnet_tfidf',
        f'{cat}_v4_mpnet',
        cat_v4,
    ]
    # Allow override via env var EVAL_MODEL_SUFFIX (for benchmarking different versions)
    override = _os.environ.get('EVAL_MODEL_SUFFIX')
    if override:
        candidates = [f'{cat}_v4{override}'] + candidates
    use_prefix = next((p for p in candidates if _os.path.exists(
        f'{PROJECT_ROOT}/models/{p}_thresholds.pkl')), cat_v4)
    print(f'  using model prefix: {use_prefix}')
    ml_preds = predict_ml(inputs, cat_v4, prefix=use_prefix)
    rule_preds = predict_rules(inputs)
    if rule_preds is None or len(rule_preds) == 0:
        rule_preds = pd.DataFrame(columns=['code', 'attr', 'rule_pred', 'rule_tier', 'm_key'])
    ml_preds['m_key'] = ml_preds.code.astype(str) + '|' + ml_preds.attr
    rule_preds['m_key'] = rule_preds.code.astype(str) + '|' + rule_preds.attr
    g['m_key'] = g.code + '|' + g.attr
    g = g.merge(inputs[['code', 'in_scope']], on='code', how='left')

    merged = g.merge(ml_preds[['m_key', 'ml_pred', 'ml_conf', 'ml_fired']],
                     on='m_key', how='left')
    merged = merged.merge(rule_preds[['m_key', 'rule_pred', 'rule_tier']],
                          on='m_key', how='left')

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

    def _stats(df):
        gn = df['gold_value'].apply(norm_value)
        pn = df['cascade_pred'].apply(norm_value)
        valid = gn.notna() & pn.notna()
        return int(valid.sum()), int((gn == pn)[valid].sum())

    rows = []
    print(f'  {"attr":22s}  {"src":>10s}  {"raw_n":>5s}  {"raw_acc":>8s}  '
          f'{"clean_n":>7s}  {"clean_acc":>9s}')
    print('  ' + '-' * 70)
    for attr in sorted(merged.attr.unique()):
        sub_all = merged[merged.attr == attr]
        sub_clean = sub_all[sub_all.in_scope == True]
        cov_a, cor_a = _stats(sub_all)
        cov_c, cor_c = _stats(sub_clean)
        src = ','.join(sorted(set(sub_all[sub_all.cascade_pred.notna()]['cascade_source'].unique())))
        a_acc = 100*cor_a/cov_a if cov_a > 0 else float('nan')
        c_acc = 100*cor_c/cov_c if cov_c > 0 else float('nan')
        print(f'  {attr:22s}  {src:>10s}  {cov_a:>5d}  {a_acc:>7.1f}%  '
              f'{cov_c:>7d}  {c_acc:>8.1f}%')
        rows.append({'cat': cat, 'attr': attr, 'src': src,
                     'raw_covered': cov_a, 'raw_correct': cor_a, 'raw_acc': a_acc/100,
                     'clean_covered': cov_c, 'clean_correct': cor_c, 'clean_acc': c_acc/100})
    cov_a, cor_a = _stats(merged)
    cov_c, cor_c = _stats(merged[merged.in_scope == True])
    print(f'  {"OVERALL":22s}             {cov_a:>5d}  '
          f'{100*cor_a/max(cov_a,1):>7.1f}%  {cov_c:>7d}  '
          f'{100*cor_c/max(cov_c,1):>8.1f}%')
    return rows


def main():
    gold = pd.read_parquet(PROJECT_ROOT / 'datasets/processed/manual_gold_consensus.parquet')
    off_dir = Path.home() / 'off_work'
    if not off_dir.exists():
        off_dir = PROJECT_ROOT / 'datasets/raw'

    all_rows = []
    for cat in ['pasta', 'chocolate', 'cheeses']:
        all_rows.extend(eval_cat(cat, gold, off_dir))

    df = pd.DataFrame(all_rows)
    out = PROJECT_ROOT / 'datasets/processed/v4_eval_consensus_clean.parquet'
    df.to_parquet(out, index=False)
    raw_cov = df.raw_covered.sum(); raw_cor = df.raw_correct.sum()
    cln_cov = df.clean_covered.sum(); cln_cor = df.clean_correct.sum()
    print(f'\n=== GRAND ===')
    print(f'  RAW (incl. mislabeled non-cat products): {raw_cor}/{raw_cov} '
          f'({100*raw_cor/raw_cov:.1f}%)')
    print(f'  CLEAN (only in-scope products):          {cln_cor}/{cln_cov} '
          f'({100*cln_cor/cln_cov:.1f}%)')
    print(f'  saved: {out}')


if __name__ == '__main__':
    main()
