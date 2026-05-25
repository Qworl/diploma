"""Evaluate v4 cascade (rules + ML) against LLM consensus gold.

Like eval_v4_manual.py, but uses datasets/processed/manual_gold_consensus.parquet
(majority-vote of qwen3.7-max + ds-r1 + mistral-large) instead of single-rater
manual_eval_per_product.parquet.

Reports per-attr + overall accuracy, splits by 'unanimous' vs 'split-2-1' для
честности (на unanimous gold accuracy показывает реальную силу модели,
на split-2-1 ошибки модели могут отражать legitimate ambiguity).
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


def eval_cat(cat: str, gold_df: pd.DataFrame, off_dir: Path):
    print(f'\n=== {cat.upper()} ===')
    g = gold_df[gold_df.category == cat].copy()
    # filter to consensus only (drop disputed)
    g_full = g.copy()
    g = g[~g.disputed].copy()
    g['code'] = g['code'].astype(str)
    if cat == 'cheeses':
        g['gold_value'] = g['gold_value'].replace({'semi_soft': 'soft'})
    print(f'  consensus cells: {len(g)} of {len(g_full)} total ({len(g_full) - len(g)} disputed dropped)')
    codes = set(g.code.unique())

    inputs = build_inputs_df(off_dir / f'{cat}_off_full.parquet', codes)
    # Fallback to full food.parquet for codes outside cat filter
    if len(inputs) < len(codes):
        import duckdb
        food_path = off_dir / 'food.parquet'
        if food_path.exists():
            missing = codes - set(inputs.code.astype(str))
            if missing:
                miss_sql = ','.join(f"'{c}'" for c in missing)
                con = duckdb.connect()
                extra = con.execute(f"""
                    SELECT code, product_name, brands, ingredients_text, quantity,
                           categories_tags, labels_tags, traces_tags,
                           countries_tags, ingredients_analysis_tags,
                           nutriments
                    FROM '{food_path}'
                    WHERE CAST(code AS VARCHAR) IN ({miss_sql})
                """).fetchdf()
                if len(extra):
                    extra = _process_struct(extra)
                    inputs = pd.concat([inputs, extra], ignore_index=True)
                    print(f'  +{len(extra)} from full food.parquet')
    print(f'  inputs loaded: {len(inputs)}/{len(codes)}')

    cat_v4 = f'{cat}_v4'
    ml_preds = predict_ml(inputs, cat_v4, prefix=cat_v4)
    rule_preds = predict_rules(inputs)
    if rule_preds is None or len(rule_preds) == 0:
        rule_preds = pd.DataFrame(columns=['code', 'attr', 'rule_pred', 'rule_tier', 'm_key'])
    ml_preds['m_key'] = ml_preds.code.astype(str) + '|' + ml_preds.attr
    rule_preds['m_key'] = rule_preds.code.astype(str) + '|' + rule_preds.attr
    g['m_key'] = g.code + '|' + g.attr

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

    rows = []
    print(f'  {"attr":22s}  {"src":>10s}  {"n_pred":>6s}  {"correct":>8s}  {"acc":>7s}  {"unanim":>7s}')
    print('  ' + '-' * 70)
    for attr in sorted(merged.attr.unique()):
        sub = merged[merged.attr == attr]
        gold_n = sub['gold_value'].apply(norm_value)
        pred_n = sub['cascade_pred'].apply(norm_value)
        valid = gold_n.notna() & pred_n.notna()
        covered = valid.sum()
        correct = (gold_n == pred_n)[valid].sum()
        acc = correct / covered if covered > 0 else float('nan')
        unanim_sub = sub[(sub.agreement_ratio >= 0.99)]
        u_gold = unanim_sub['gold_value'].apply(norm_value)
        u_pred = unanim_sub['cascade_pred'].apply(norm_value)
        u_valid = u_gold.notna() & u_pred.notna()
        u_cov = u_valid.sum()
        u_cor = (u_gold == u_pred)[u_valid].sum()
        u_acc = u_cor / u_cov if u_cov > 0 else float('nan')
        src = ','.join(sorted(set(sub[sub.cascade_pred.notna()]['cascade_source'].unique())))
        u_str = f'{u_cor}/{u_cov}={u_acc*100:.0f}%' if u_cov > 0 else 'n/a'
        print(f'  {attr:22s}  {src:>10s}  {covered:>6d}  {correct:>8d}  {acc*100:>6.1f}%  {u_str:>7s}')
        rows.append({'cat': cat, 'attr': attr, 'src': src,
                     'covered': int(covered), 'correct': int(correct), 'acc': float(acc),
                     'unanim_cov': int(u_cov), 'unanim_correct': int(u_cor),
                     'unanim_acc': float(u_acc) if u_cov > 0 else None})
    total_cov = sum(r['covered'] for r in rows)
    total_cor = sum(r['correct'] for r in rows)
    print(f'  {"OVERALL":22s}             {total_cov:>6d}  {total_cor:>8d}  '
          f'{100*total_cor/max(total_cov,1):>6.1f}%')
    return rows


def _process_struct(df: pd.DataFrame) -> pd.DataFrame:
    """Same flattening as build_inputs_df for fallback rows."""
    from scripts.build_gold_v4_wide import _pick_text, _flatten_nutriments, _safe_list, _to_str
    df['code'] = df['code'].astype(str)
    df['product_name'] = df['product_name'].apply(_pick_text)
    df['ingredients_text'] = df['ingredients_text'].apply(_pick_text)
    for col in ['categories_tags', 'labels_tags', 'traces_tags',
                'countries_tags', 'ingredients_analysis_tags']:
        df[col] = df[col].apply(lambda v: ','.join(_safe_list(v) or []) or None)
    df['quantity'] = df['quantity'].apply(_to_str)
    df['brands'] = df['brands'].apply(_to_str)
    nut_records = df['nutriments'].apply(_flatten_nutriments)
    for k in ['fat_100g', 'sugars_100g', 'proteins_100g', 'carbohydrates_100g',
              'alcohol_100g']:
        df[k] = nut_records.apply(lambda d: d.get(k))
    return df.drop(columns=['nutriments'])


def main():
    gold = pd.read_parquet(PROJECT_ROOT / 'datasets/processed/manual_gold_consensus.parquet')
    off_dir = Path.home() / 'off_work'
    if not off_dir.exists():
        off_dir = PROJECT_ROOT / 'datasets/raw'

    all_rows = []
    for cat in ['pasta', 'chocolate', 'cheeses']:
        all_rows.extend(eval_cat(cat, gold, off_dir))

    df = pd.DataFrame(all_rows)
    out = PROJECT_ROOT / 'datasets/processed/v4_eval_consensus_gold.parquet'
    df.to_parquet(out, index=False)

    grand_cov = df.covered.sum()
    grand_cor = df.correct.sum()
    u_cov = df.unanim_cov.sum()
    u_cor = df.unanim_correct.sum()
    print(f'\n=== GRAND TOTAL ===')
    print(f'  consensus gold:  {grand_cor}/{grand_cov} ({100*grand_cor/grand_cov:.1f}%)')
    if u_cov > 0:
        print(f'  unanimous gold:  {u_cor}/{u_cov} ({100*u_cor/u_cov:.1f}%)')
    print(f'  saved per-attr breakdown: {out}')


if __name__ == '__main__':
    main()
