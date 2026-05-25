"""Evaluate v4 models on manual_eval_per_product gold.

Inference flow:
1. Load 50 codes/cat from manual_eval_per_product.
2. Pull inputs from OFF dump.
3. Compute embeddings with cached sentence-transformers.
4. For each attr present in ML CATEGORY_CONFIG: predict + threshold from {cat}_v4_thresholds.pkl.
5. For TYPE_C attrs (protein_class, fat_class, etc.): apply rules.TYPE_C_RULES on nutriments.
6. Compare to manual_eval gold; report acc per (cat, attr) and overall.

Run on VM (models + OFF dump live there).
"""
from __future__ import annotations

import os
import sys
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

# Project root detection
for root in ['/home/miafrolov/Desktop/diploma',
             '/Users/miafrolov/Desktop/stuff/ai_attributes']:
    if Path(root).exists():
        sys.path.insert(0, root)
        PROJECT_ROOT = Path(root)
        break

from src.common import build_text
from src.pipeline.off_labels.rules import (
    TYPE_C_RULES, _type_c_numeric,
    TYPE_E_RULES, _type_e_regex,
    TYPE_F_RULES, _type_f_regex_multiclass,
)
from src.pipeline.ml.train import CATEGORY_CONFIG

# Re-use the v6 build_inputs helper for consistency
from scripts.build_gold_v4_wide import build_inputs_df


def predict_ml(df_inputs: pd.DataFrame, cat: str, prefix: str = None):
    """Compute ML predictions + confidences for every classifier in CATEGORY_CONFIG[cat]."""
    if prefix is None:
        prefix = cat  # e.g. pasta_v4
    cfg = CATEGORY_CONFIG[cat]['classifiers']

    # Compute embeddings via sentence-transformers (no cache for tiny eval set).
    # Use EMBEDDING_MODEL from common.py to match what training used.
    texts = build_text(df_inputs)
    from sentence_transformers import SentenceTransformer
    from src.common import EMBEDDING_MODEL
    model = SentenceTransformer(EMBEDDING_MODEL)
    X = model.encode(texts, show_progress_bar=False, batch_size=64).astype(np.float32)

    models_dir = PROJECT_ROOT / 'models'
    # load thresholds dict if exists
    thr_path = models_dir / f'{prefix}_thresholds.pkl'
    thresholds = {}
    if thr_path.exists():
        with open(thr_path, 'rb') as f:
            thresholds = pickle.load(f)

    # Hybrid TF-IDF + SVD support: если есть {prefix}_tfidf.pkl + _tfidf_svd.pkl, считаем
    # tfidf-128 и конкатенируем к SBERT для plain dense (896-dim) фич.
    X_hybrid = X
    tfidf_path = models_dir / f'{prefix}_tfidf.pkl'
    svd_path = models_dir / f'{prefix}_tfidf_svd.pkl'
    if tfidf_path.exists() and svd_path.exists():
        with open(tfidf_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(svd_path, 'rb') as f:
            svd = pickle.load(f)
        X_tfidf = svd.transform(vectorizer.transform(texts)).astype(np.float32)
        X_hybrid = np.hstack([X, X_tfidf])
        print(f'  using hybrid TF-IDF+SVD features: shape {X_hybrid.shape}')

    preds_rows = []
    for attr, spec in cfg.items():
        clf_path = models_dir / f'{prefix}_{attr}_xgb.pkl'
        if not clf_path.exists():
            print(f'  [skip] no model for {attr}')
            continue
        with open(clf_path, 'rb') as f:
            clf = pickle.load(f)
        thr = thresholds.get(attr, 0.5)
        proba = clf.predict_proba(X_hybrid)
        if spec['type'] == 'binary':
            confs = np.maximum(proba[:, 1], 1 - proba[:, 1])
            preds = (proba[:, 1] >= 0.5).astype(int)
            label_map = {0: False, 1: True}
            pred_lbls = [label_map[p] for p in preds]
        else:
            # multiclass: need label encoder
            le_path = models_dir / f'{prefix}_{attr}_le.pkl'
            with open(le_path, 'rb') as f:
                le = pickle.load(f)
            confs = proba.max(axis=1)
            preds = proba.argmax(axis=1)
            pred_lbls = [str(le.classes_[p]) for p in preds]
        for i, code in enumerate(df_inputs['code'].astype(str)):
            preds_rows.append({
                'code': code, 'attr': attr,
                'ml_pred': pred_lbls[i],
                'ml_conf': float(confs[i]),
                'ml_threshold': float(thr),
                'ml_fired': confs[i] >= thr,
            })
    return pd.DataFrame(preds_rows)


# TYPE_F rules with explicit canonical-name regex (high precision when match).
# Per-attr override: these rules win over ML when they fire.
TYPE_F_HIGH_PRECISION = {'texture'}


def predict_rules(df_inputs: pd.DataFrame):
    """Apply TYPE_C, TYPE_E (high-precision) + TYPE_F (per-attr tier).

    TYPE_F rules in TYPE_F_HIGH_PRECISION → 'high' tier (override ML).
    Остальные TYPE_F → 'low' tier (ML wins when confident).
    """
    rows = []
    for _, row in df_inputs.iterrows():
        code = str(row['code'])
        row_d = row.to_dict()
        for attr in TYPE_C_RULES:
            val = _type_c_numeric(row_d, attr)
            if val is not None:
                rows.append({'code': code, 'attr': attr,
                             'rule_pred': str(val), 'rule_tier': 'high'})
        for attr in TYPE_E_RULES:
            val = _type_e_regex(row_d, attr)
            if val is not None:
                rows.append({'code': code, 'attr': attr,
                             'rule_pred': str(val).lower(), 'rule_tier': 'high'})
        for attr in TYPE_F_RULES:
            val = _type_f_regex_multiclass(row_d, attr)
            if val is not None:
                tier = 'high' if attr in TYPE_F_HIGH_PRECISION else 'low'
                rows.append({'code': code, 'attr': attr,
                             'rule_pred': str(val).lower(), 'rule_tier': tier})
    return pd.DataFrame(rows)


def norm_value(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in ('', 'none', 'null', 'nan', 'true', 'false'):
        if s == 'true': return 'true'
        if s == 'false': return 'false'
        return None
    return s


def eval_cat(cat: str, manual_df: pd.DataFrame, off_dir: Path):
    print(f'\n=== {cat.upper()} ===')
    gold = manual_df[manual_df.category == cat].copy()
    gold['code'] = gold['code'].astype(str)
    codes = set(gold.code.unique())
    print(f'  manual gold: {len(gold)} (code,attr) cells, {len(codes)} unique codes')

    # Apply pasta v4 cuisine merge to gold if needed (NOT in manual)
    # Apply cheeses semi_soft → soft merge
    if cat == 'cheeses':
        gold['manual'] = gold['manual'].replace({'semi_soft': 'soft'})

    off_path = off_dir / f'{cat}_off_full.parquet'
    inputs = build_inputs_df(off_path, codes)
    print(f'  loaded {len(inputs)} OFF rows for inference')

    cat_v4 = f'{cat}_v4'
    ml_preds = predict_ml(inputs, cat_v4, prefix=cat_v4)
    rule_preds = predict_rules(inputs)

    # Merge gold with predictions
    gold['m_key'] = gold.code + '|' + gold.attr
    ml_preds['m_key'] = ml_preds.code + '|' + ml_preds.attr
    rule_preds['m_key'] = rule_preds.code + '|' + rule_preds.attr
    g = gold.merge(ml_preds[['m_key', 'ml_pred', 'ml_conf', 'ml_fired']],
                   on='m_key', how='left')
    g = g.merge(rule_preds[['m_key', 'rule_pred', 'rule_tier']],
                on='m_key', how='left')

    # Cascade decision:
    #   1. TYPE_C/E rule (high tier): always use
    #   2. ML confident: use ML
    #   3. TYPE_F rule (low tier): use if ML didn't fire
    #   4. Fallback
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
    cascade_results = g.apply(_cascade, axis=1)
    g['cascade_source'] = [c[0] for c in cascade_results]
    g['cascade_pred'] = [c[1] for c in cascade_results]

    # Per-attr accuracy
    print(f'  {"attr":22s}  {"src":>10s}  {"n_pred":>6s}  {"correct":>8s}  {"acc":>7s}')
    print('  ' + '-' * 60)
    overall = {'total': 0, 'correct': 0, 'covered': 0}
    rows = []
    for attr in sorted(gold.attr.unique()):
        sub = g[g.attr == attr]
        total = len(sub)
        m_norm = sub['manual'].apply(norm_value)
        p_norm = sub['cascade_pred'].apply(norm_value)
        valid = m_norm.notna() & p_norm.notna()
        covered = valid.sum()
        correct = (m_norm == p_norm)[valid].sum()
        acc = correct / covered if covered > 0 else float('nan')
        src = ','.join(sorted(set(sub[sub.cascade_pred.notna()]['cascade_source'].unique())))
        print(f'  {attr:22s}  {src:>10s}  {covered:>6d}  {correct:>8d}  {acc*100:>6.1f}%')
        overall['total'] += total
        overall['correct'] += int(correct)
        overall['covered'] += int(covered)
        rows.append({'cat': cat, 'attr': attr, 'src': src,
                     'covered': int(covered), 'correct': int(correct),
                     'acc': float(acc), 'total_gold': int(total)})

    acc_overall = overall['correct'] / overall['covered'] if overall['covered'] > 0 else float('nan')
    print(f'  {"OVERALL":22s}             {overall["covered"]:>6d}  {overall["correct"]:>8d}  {acc_overall*100:>6.1f}%')
    print(f'  coverage: {overall["covered"]}/{overall["total"]} cells (cascade answered)')
    return rows


def main():
    manual_path = PROJECT_ROOT / 'datasets/processed/manual_eval_per_product.parquet'
    manual_df = pd.read_parquet(manual_path)

    off_dir = Path.home() / 'off_work'
    if not off_dir.exists():
        off_dir = PROJECT_ROOT / 'datasets/raw'

    all_rows = []
    for cat in ['pasta', 'chocolate', 'cheeses']:
        all_rows.extend(eval_cat(cat, manual_df, off_dir))

    # Save
    out = PROJECT_ROOT / 'datasets/processed/v4_eval_manual_gold.parquet'
    pd.DataFrame(all_rows).to_parquet(out, index=False)
    print(f'\nSaved per-attr breakdown: {out}')

    # Grand total
    df = pd.DataFrame(all_rows)
    grand_cov = df.covered.sum()
    grand_corr = df.correct.sum()
    grand_total = df.total_gold.sum()
    if grand_cov > 0:
        print(f'\n=== GRAND TOTAL ===')
        print(f'  covered: {grand_cov}/{grand_total} ({100*grand_cov/grand_total:.1f}%)')
        print(f'  accuracy: {grand_corr}/{grand_cov} ({100*grand_corr/grand_cov:.1f}%)')


if __name__ == '__main__':
    main()
