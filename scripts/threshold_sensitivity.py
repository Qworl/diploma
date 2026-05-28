"""Threshold sensitivity analysis (post-hoc, no retraining).

Для каждого offset ∈ {-0.10, -0.05, 0, +0.05, +0.10}:
  * thresholds_eff[attr] = clip(thresholds[attr] + offset, 0.0, 1.0)
  * Если ml_conf < thresholds_eff[attr] → ml_fired=False → ячейка эскалируется
    в Layer 4 (cascade_pred=None), увеличивая LLM-share.
  * Если ml_conf ≥ thresholds_eff[attr] → ml_fired=True → cascade принимает ML.

Метрики:
  * cascade_acc — точность каскада на ячейках, где cascade_pred != None
    (cascade-valid знаменатель — соответствует headline ВКР).
  * llm_share — доля ячеек, где cascade_pred == None (escalation).
  * e2e_acc_proxy — гибридная точность: cascade_acc на cascade-valid + LLM_acc
    на fallback (LLM_acc оценивается из cascade_plus_llm4_summary.parquet,
    Sonnet 4.5; cells-weighted average ~96%).

Источники:
  * cascade_raw_with_conf.parquet — генерируется этим же скриптом (или
    pulled из VM). Содержит per-cell: code, attr, gold_value, ml_pred,
    ml_conf, rule_pred, rule_tier, in_scope.
  * models/{cat}_v4_mpnet_tfidf_noleak_thresholds.pkl — base thresholds.

Если cascade_raw_with_conf.parquet не существует локально (нужны OFF
parquets + heavy ML inference), запускать на VM через:
  python -m scripts.threshold_sensitivity --regen-raw

Затем pull:
  rsync -az -e "ssh -S ~/.ssh/control/yandex-vm" \\
    miafrolov@158.160.88.176:~/Desktop/diploma/datasets/processed/cascade_raw_with_conf_*.parquet \\
    datasets/processed/

И запускать post-hoc:
  python -m scripts.threshold_sensitivity

Output:
  datasets/processed/threshold_sensitivity.parquet — offset × category ×
    (cascade_acc, llm_share, e2e_acc, n_cascade_valid, n_total)
  datasets/processed/threshold_sensitivity_summary.json — global summary.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root detection (works both locally and on VM).
for root in ['/home/miafrolov/Desktop/diploma',
             '/Users/miafrolov/Desktop/stuff/ai_attributes']:
    if Path(root).exists():
        sys.path.insert(0, root)
        PROJECT_ROOT = Path(root)
        break

PROCESSED = PROJECT_ROOT / 'datasets/processed'
MODELS = PROJECT_ROOT / 'models'

CATEGORIES = ['pasta', 'chocolate', 'cheeses']
OFFSETS = [-0.10, -0.05, 0.0, 0.05, 0.10]
PREFIX = '{cat}_v4_mpnet_tfidf_noleak'

# LLM-fallback accuracy proxy per category (Sonnet 4.5, cells-weighted,
# источник: datasets/processed/cascade_plus_llm4_hybrid.parquet,
# acc_on_covered усреднённое по cells per category для sonnet45).
# Если LLM-share = 0 → e2e_acc = cascade_acc.
LLM_FALLBACK_ACC_DEFAULT = 0.86  # консервативная оценка


def compute_llm_acc_by_cat() -> dict:
    """Усреднённая acc Layer 4 LLM (Sonnet 4.5) per category, cells-weighted."""
    path = PROCESSED / 'cascade_plus_llm4_hybrid.parquet'
    if not path.exists():
        return {c: LLM_FALLBACK_ACC_DEFAULT for c in CATEGORIES}
    df = pd.read_parquet(path)
    df = df[df.llm_model == 'sonnet45']
    out = {}
    for cat in CATEGORIES:
        sub = df[df.category == cat]
        if len(sub) == 0 or sub.n_covered.sum() == 0:
            out[cat] = LLM_FALLBACK_ACC_DEFAULT
            continue
        # cells-weighted average
        weighted = (sub.acc_on_covered * sub.n_covered).sum() / sub.n_covered.sum()
        out[cat] = float(weighted)
    return out


def regenerate_raw_with_conf():
    """Запускается на VM. Перегенерирует cascade_raw_with_conf_{cat}.parquet
    с per-cell ml_conf, rule_pred, rule_tier (всё без применения thresholds)."""
    # Lazy import — heavy deps (sentence-transformers, xgboost).
    from scripts.eval_v4_manual import predict_ml, predict_rules
    from scripts.build_gold_v4_wide import build_inputs_df
    from scripts.eval_v4_consensus import _process_struct
    from scripts.eval_v4_consensus_clean import is_in_scope, CAT_VALID_TAGS
    import os

    off_dir = Path(os.environ.get('OFF_DATA_ROOT', '/home/miafrolov/off_work'))
    gold = pd.read_parquet(PROCESSED / 'manual_gold_consensus.parquet')
    gold['code'] = gold['code'].astype(str)
    gold_field = 'gold_value' if 'gold_value' in gold.columns else 'manual'

    for cat in CATEGORIES:
        print(f'\n=== Regenerating raw cascade for {cat} ===')
        g = gold[gold.category == cat].copy()
        if cat == 'cheeses':
            g[gold_field] = g[gold_field].replace({'semi_soft': 'soft'})
        if 'disputed' in g.columns:
            g = g[~g.disputed]
        codes = set(g.code.unique())

        inputs = build_inputs_df(off_dir / f'{cat}_off_full.parquet', codes)
        missing = codes - set(inputs.code.astype(str))
        if missing:
            import duckdb
            miss_sql = ','.join(f"'{c}'" for c in missing)
            con = duckdb.connect()
            extra = con.execute(f"""
                SELECT code, product_name, brands, ingredients_text, quantity,
                       categories_tags, labels_tags, traces_tags, countries_tags,
                       ingredients_analysis_tags, nutriments
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

        prefix = PREFIX.format(cat=cat)
        ml_preds = predict_ml(inputs, f'{cat}_v4', prefix=prefix)
        rule_preds = predict_rules(inputs)

        ml_preds['m_key'] = ml_preds.code.astype(str) + '|' + ml_preds.attr
        rule_preds['m_key'] = rule_preds.code.astype(str) + '|' + rule_preds.attr
        g['m_key'] = g.code + '|' + g.attr

        g = g.merge(inputs[['code', 'in_scope']], on='code', how='left')
        merged = g.merge(
            ml_preds[['m_key', 'ml_pred', 'ml_conf', 'ml_threshold']],
            on='m_key', how='left'
        )
        merged = merged.merge(
            rule_preds[['m_key', 'rule_pred', 'rule_tier']],
            on='m_key', how='left'
        )

        # Cast to strings (для parquet, mixed bool/str).
        for col in ('ml_pred', 'rule_pred'):
            merged[col] = merged[col].apply(
                lambda v: None if v is None or (isinstance(v, float) and pd.isna(v))
                else str(v).lower() if isinstance(v, bool) else str(v)
            )
        merged[gold_field] = merged[gold_field].apply(
            lambda v: None if v is None or (isinstance(v, float) and pd.isna(v))
            else str(v).lower() if isinstance(v, bool) else str(v)
        )

        cols = ['code', 'attr', gold_field, 'ml_pred', 'ml_conf',
                'ml_threshold', 'rule_pred', 'rule_tier', 'in_scope']
        out = merged[cols].rename(columns={gold_field: 'gold_value'})
        out_path = PROCESSED / f'cascade_raw_with_conf_{cat}.parquet'
        out.to_parquet(out_path, index=False)
        print(f'  saved {out_path} (n={len(out)})')


def derive_cascade(raw: pd.DataFrame, thresholds_eff: dict) -> pd.DataFrame:
    """Apply thresholds offset → derive cascade_source / cascade_pred."""
    df = raw.copy()

    def _row(r):
        has_rule = r['rule_pred'] is not None and not (
            isinstance(r['rule_pred'], float) and pd.isna(r['rule_pred']))
        tier = r.get('rule_tier')
        # Rule HIGH always wins.
        if has_rule and tier == 'high':
            return ('rule_h', r['rule_pred'])
        # ML fired if conf ≥ effective threshold (and ml_pred exists).
        thr = thresholds_eff.get(r['attr'], 0.5)
        ml_pred = r['ml_pred']
        ml_conf = r['ml_conf']
        ml_valid = (ml_pred is not None
                    and not (isinstance(ml_pred, float) and pd.isna(ml_pred))
                    and ml_conf is not None and not pd.isna(ml_conf))
        if ml_valid and ml_conf >= thr:
            return ('ml', ml_pred)
        # Rule LOW as fallback.
        if has_rule and tier == 'low':
            return ('rule_l', r['rule_pred'])
        return ('fallback', None)

    res = df.apply(_row, axis=1)
    df['cascade_source'] = [r[0] for r in res]
    df['cascade_pred'] = [r[1] for r in res]
    return df


def norm_value(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in ('', 'none', 'null', 'nan'):
        return None
    return s


def compute_metrics(df: pd.DataFrame, llm_acc: float) -> dict:
    """Compute cascade_acc / llm_share / e2e_acc on in_scope cells."""
    sub = df[df.in_scope == True].copy()
    sub['gn'] = sub.gold_value.apply(norm_value)
    sub['cn'] = sub.cascade_pred.apply(norm_value)
    sub = sub[sub.gn.notna()]  # gold-valid

    n_total = len(sub)
    if n_total == 0:
        return dict(n_total=0, n_cascade_valid=0, n_fallback=0,
                    cascade_acc=float('nan'), llm_share=float('nan'),
                    e2e_acc=float('nan'))

    cascade_valid_mask = sub.cn.notna()
    n_cascade_valid = int(cascade_valid_mask.sum())
    n_fallback = int(n_total - n_cascade_valid)

    cascade_correct = int(((sub.gn == sub.cn) & cascade_valid_mask).sum())
    cascade_acc = cascade_correct / max(n_cascade_valid, 1)
    llm_share = n_fallback / n_total
    # E2E proxy: cascade_acc на cascade-valid + llm_acc на fallback.
    e2e_acc = (cascade_correct + llm_acc * n_fallback) / n_total

    return dict(
        n_total=n_total,
        n_cascade_valid=n_cascade_valid,
        n_fallback=n_fallback,
        cascade_acc=float(cascade_acc),
        llm_share=float(llm_share),
        e2e_acc=float(e2e_acc),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regen-raw', action='store_true',
                    help='Regenerate cascade_raw_with_conf_*.parquet (heavy, VM-only)')
    args = ap.parse_args()

    if args.regen_raw:
        regenerate_raw_with_conf()
        return

    # Check raw files exist.
    missing = [c for c in CATEGORIES
               if not (PROCESSED / f'cascade_raw_with_conf_{c}.parquet').exists()]
    if missing:
        print(f'BLOCKER: missing cascade_raw_with_conf_*.parquet for {missing}')
        print('Run on VM: python -m scripts.threshold_sensitivity --regen-raw')
        print('Then rsync pull from VM to local datasets/processed/.')
        sys.exit(1)

    llm_acc_by_cat = compute_llm_acc_by_cat()
    print('LLM-fallback acc per cat (Sonnet 4.5, cells-weighted):')
    for c, a in llm_acc_by_cat.items():
        print(f'  {c}: {a:.4f}')

    rows = []
    for cat in CATEGORIES:
        raw = pd.read_parquet(PROCESSED / f'cascade_raw_with_conf_{cat}.parquet')
        thr_path = MODELS / f'{PREFIX.format(cat=cat)}_thresholds.pkl'
        with open(thr_path, 'rb') as f:
            base_thr = pickle.load(f)
        # Конвертируем np.float64 → float, на случай разных версий pkl.
        base_thr = {k: float(v) for k, v in base_thr.items()}

        for off in OFFSETS:
            thr_eff = {k: float(np.clip(v + off, 0.0, 1.0))
                       for k, v in base_thr.items()}
            df_eff = derive_cascade(raw, thr_eff)
            m = compute_metrics(df_eff, llm_acc_by_cat[cat])
            rows.append({
                'category': cat,
                'offset': float(off),
                **m,
                'base_thr_min': float(min(base_thr.values())),
                'base_thr_max': float(max(base_thr.values())),
            })
            print(f'  {cat:10s} off={off:+.2f}  '
                  f'cascade_acc={m["cascade_acc"]*100:5.2f}%  '
                  f'llm_share={m["llm_share"]*100:5.2f}%  '
                  f'e2e_acc={m["e2e_acc"]*100:5.2f}%  '
                  f'(n={m["n_total"]})')

    df_out = pd.DataFrame(rows)

    # Global per offset (cells-weighted across categories).
    glob_rows = []
    for off in OFFSETS:
        sub = df_out[df_out.offset == off]
        n_tot = sub.n_total.sum()
        n_cv = sub.n_cascade_valid.sum()
        n_fb = sub.n_fallback.sum()
        # cells-weighted accuracy
        cascade_correct = (sub.cascade_acc * sub.n_cascade_valid).sum()
        cascade_acc = cascade_correct / max(n_cv, 1)
        llm_share = n_fb / max(n_tot, 1)
        # e2e weighted: per-cat already includes llm_acc, just weight by n_total
        e2e_weighted = (sub.e2e_acc * sub.n_total).sum() / max(n_tot, 1)
        glob_rows.append({
            'category': 'GLOBAL',
            'offset': float(off),
            'n_total': int(n_tot),
            'n_cascade_valid': int(n_cv),
            'n_fallback': int(n_fb),
            'cascade_acc': float(cascade_acc),
            'llm_share': float(llm_share),
            'e2e_acc': float(e2e_weighted),
            'base_thr_min': float('nan'),
            'base_thr_max': float('nan'),
        })
    df_out = pd.concat([df_out, pd.DataFrame(glob_rows)], ignore_index=True)

    out_path = PROCESSED / 'threshold_sensitivity.parquet'
    df_out.to_parquet(out_path, index=False)
    print(f'\nSaved {out_path}')

    # JSON summary (global only).
    summary = {
        'offsets': OFFSETS,
        'llm_acc_per_cat': llm_acc_by_cat,
        'global': df_out[df_out.category == 'GLOBAL'].to_dict('records'),
    }
    sum_path = PROCESSED / 'threshold_sensitivity_summary.json'
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved {sum_path}')

    # Pretty print global table.
    print('\n=== GLOBAL (cells-weighted across pasta+chocolate+cheeses) ===')
    g = df_out[df_out.category == 'GLOBAL'].copy()
    print(g[['offset', 'n_total', 'n_cascade_valid', 'n_fallback',
             'cascade_acc', 'llm_share', 'e2e_acc']].to_string(index=False))


if __name__ == '__main__':
    main()
