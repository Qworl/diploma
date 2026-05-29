"""Eval ML classifiers on brand-disjoint vs code-disjoint test sets (тикет 2026-05-28).

Loads:
  - `{cat}_stratified_brand_disjoint_*` models (trained on brand-disjoint train)
  - `{cat}_stratified_code_disjoint_*` models (trained on code-disjoint train baseline)
  - Test code parquets `{cat}_(brand|code)_disjoint_test_codes.parquet`
  - Source labels `{cat}_stratified_silver_extended.parquet`

For each (cat, attribute, mode):
  Computes ML-only metrics on its own test set:
    - accuracy (top-1)
    - macro-F1 (sklearn.metrics)
    - n_test (after restricting to known classes from LabelEncoder)

Headline output `datasets/processed/v4_brand_disjoint_eval.json`:
  Per-cat micro/macro accuracy and F1 for both modes + degradation (brand - code).

Usage (на VM или local):
    python -m src.eval.eval_brand_disjoint
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

for root in ['/home/miafrolov/Desktop/diploma',
             '/Users/miafrolov/Desktop/stuff/ai_attributes']:
    if Path(root).exists():
        PROJECT_ROOT = Path(root)
        sys.path.insert(0, str(root))
        break

from src.common import MODELS_DIR, PROCESSED_DIR, build_text, get_embeddings, setup_logging
from src.pipeline.ml.train import CATEGORY_CONFIG

logger = logging.getLogger(__name__)

CATEGORIES = ['pasta', 'chocolate', 'cheeses']
MODES = ['code_disjoint', 'brand_disjoint']


def load_thresholds_for_mode(cat: str, mode: str) -> dict:
    prefix = f'{cat}_stratified_{mode}'
    p = Path(MODELS_DIR) / f'{prefix}_thresholds.pkl'
    if p.exists():
        with open(p, 'rb') as f:
            return pickle.load(f)
    return {}


def evaluate_one(cat: str, mode: str, embeddings_cache: dict) -> dict:
    """Evaluate all classifiers for {cat}_stratified_{mode}_* on the {mode} test split."""
    category_key = f'{cat}_stratified'
    config = CATEGORY_CONFIG[category_key]
    prefix = f'{cat}_stratified_{mode}'

    # Load labels & test codes
    ss_filename = config['silver_standard'].replace('_silver_standard', '_silver_extended')
    df = pd.read_parquet(Path(PROCESSED_DIR) / ss_filename)
    df['code'] = df['code'].astype(str)

    test_codes_path = Path(PROCESSED_DIR) / f'{cat}_{mode}_test_codes.parquet'
    test_codes = set(pd.read_parquet(test_codes_path)['code'].astype(str).tolist())
    test_df = df[df['code'].isin(test_codes)].reset_index(drop=True).copy()
    logger.info('[%s/%s] test rows=%d', cat, mode, len(test_df))

    # Compute test embeddings (cache by (cat, mode))
    key = (cat, mode)
    if key in embeddings_cache:
        X_test = embeddings_cache[key]
    else:
        texts = build_text(test_df)
        X_test = get_embeddings(texts, cache_path=None)
        embeddings_cache[key] = X_test

    thresholds = load_thresholds_for_mode(cat, mode)

    per_attr = {}
    for attr_name, attr_config in config['classifiers'].items():
        if attr_name not in test_df.columns:
            continue
        clf_path = Path(MODELS_DIR) / f'{prefix}_{attr_name}_xgb.pkl'
        if not clf_path.exists():
            logger.warning('  skip %s: no model (%s)', attr_name, clf_path.name)
            continue
        with open(clf_path, 'rb') as f:
            clf = pickle.load(f)
        le = None
        le_path = Path(MODELS_DIR) / f'{prefix}_{attr_name}_le.pkl'
        if le_path.exists():
            with open(le_path, 'rb') as f:
                le = pickle.load(f)

        y_true_full = test_df[attr_name]
        mask = y_true_full.notna()
        if attr_config['type'] == 'multiclass' and le is not None:
            mask = mask & y_true_full.isin(set(le.classes_))
        if mask.sum() < 5:
            logger.warning('  skip %s: test too small (%d)', attr_name, int(mask.sum()))
            continue

        Xs = X_test[mask.values]
        y_true = y_true_full[mask].values

        proba = clf.predict_proba(Xs)
        if attr_config['type'] == 'binary':
            # binary classifier — proba[:, 1] is P(positive)
            # threshold applied at >= 0.5 for prediction; we follow original train logic
            y_pred = (proba[:, 1] >= 0.5).astype(int)
            y_true_int = pd.Series(y_true).astype(bool).astype(int).values
            acc = accuracy_score(y_true_int, y_pred)
            f1 = f1_score(y_true_int, y_pred, average='macro', zero_division=0)
        else:
            # multiclass
            y_pred_idx = proba.argmax(axis=1)
            y_pred = le.inverse_transform(y_pred_idx)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        per_attr[attr_name] = {
            'n': int(mask.sum()),
            'accuracy': float(acc),
            'macro_f1': float(f1),
            'threshold_used': float(thresholds.get(attr_name, 0.5)),
        }
        logger.info('  %s: n=%d, acc=%.4f, macro_f1=%.4f',
                    attr_name, int(mask.sum()), acc, f1)

    # Aggregate (cells-weighted micro-avg and attr-unweighted macro-avg)
    if per_attr:
        n_total = sum(v['n'] for v in per_attr.values())
        cells_micro_acc = sum(v['accuracy'] * v['n'] for v in per_attr.values()) / n_total
        cells_micro_f1 = sum(v['macro_f1'] * v['n'] for v in per_attr.values()) / n_total
        attr_macro_acc = sum(v['accuracy'] for v in per_attr.values()) / len(per_attr)
        attr_macro_f1 = sum(v['macro_f1'] for v in per_attr.values()) / len(per_attr)
    else:
        n_total = cells_micro_acc = cells_micro_f1 = attr_macro_acc = attr_macro_f1 = 0.0

    return {
        'category': cat,
        'mode': mode,
        'n_attrs': len(per_attr),
        'n_total_cells': int(n_total),
        'cells_weighted_accuracy': float(cells_micro_acc),
        'cells_weighted_macro_f1': float(cells_micro_f1),
        'attr_unweighted_accuracy': float(attr_macro_acc),
        'attr_unweighted_macro_f1': float(attr_macro_f1),
        'per_attr': per_attr,
    }


def compute_global_deltas(per_run: list[dict]) -> dict:
    """For each cat, compute delta = brand_disjoint - code_disjoint on shared attrs."""
    by_cat_mode = {(r['category'], r['mode']): r for r in per_run}
    deltas = {}
    for cat in CATEGORIES:
        code = by_cat_mode.get((cat, 'code_disjoint'))
        brand = by_cat_mode.get((cat, 'brand_disjoint'))
        if not code or not brand:
            continue
        # Shared attrs only
        shared = sorted(set(code['per_attr'].keys()) & set(brand['per_attr'].keys()))
        if not shared:
            continue
        # Recompute aggregates on shared attrs
        def agg(run, attrs):
            n_total = sum(run['per_attr'][a]['n'] for a in attrs)
            if n_total == 0:
                return 0.0, 0.0
            acc = sum(run['per_attr'][a]['accuracy'] * run['per_attr'][a]['n'] for a in attrs) / n_total
            f1 = sum(run['per_attr'][a]['macro_f1'] * run['per_attr'][a]['n'] for a in attrs) / n_total
            return acc, f1
        code_acc, code_f1 = agg(code, shared)
        brand_acc, brand_f1 = agg(brand, shared)
        deltas[cat] = {
            'shared_attrs': shared,
            'code_disjoint_accuracy': float(code_acc),
            'brand_disjoint_accuracy': float(brand_acc),
            'delta_accuracy_pp': float((brand_acc - code_acc) * 100),
            'code_disjoint_macro_f1': float(code_f1),
            'brand_disjoint_macro_f1': float(brand_f1),
            'delta_macro_f1': float(brand_f1 - code_f1),
        }
    # Global avg on shared attrs across all cats (cells-weighted)
    total_code_n = total_brand_n = 0
    total_code_acc = total_brand_acc = 0.0
    total_code_f1 = total_brand_f1 = 0.0
    for cat, d in deltas.items():
        # weight by sum of shared attr n in each run
        code_run = by_cat_mode[(cat, 'code_disjoint')]
        brand_run = by_cat_mode[(cat, 'brand_disjoint')]
        for a in d['shared_attrs']:
            n_c = code_run['per_attr'][a]['n']
            n_b = brand_run['per_attr'][a]['n']
            total_code_n += n_c
            total_brand_n += n_b
            total_code_acc += code_run['per_attr'][a]['accuracy'] * n_c
            total_brand_acc += brand_run['per_attr'][a]['accuracy'] * n_b
            total_code_f1 += code_run['per_attr'][a]['macro_f1'] * n_c
            total_brand_f1 += brand_run['per_attr'][a]['macro_f1'] * n_b
    if total_code_n > 0 and total_brand_n > 0:
        global_summary = {
            'n_total_code_cells': int(total_code_n),
            'n_total_brand_cells': int(total_brand_n),
            'global_code_disjoint_accuracy': float(total_code_acc / total_code_n),
            'global_brand_disjoint_accuracy': float(total_brand_acc / total_brand_n),
            'global_delta_accuracy_pp': float(((total_brand_acc / total_brand_n) - (total_code_acc / total_code_n)) * 100),
            'global_code_disjoint_macro_f1': float(total_code_f1 / total_code_n),
            'global_brand_disjoint_macro_f1': float(total_brand_f1 / total_brand_n),
            'global_delta_macro_f1': float((total_brand_f1 / total_brand_n) - (total_code_f1 / total_code_n)),
        }
    else:
        global_summary = {}
    return {'per_category_delta': deltas, 'global_summary': global_summary}


def main():
    setup_logging()
    per_run = []
    embeddings_cache: dict = {}
    for mode in MODES:
        for cat in CATEGORIES:
            logger.info('=' * 60)
            logger.info('EVAL %s / %s', cat, mode)
            logger.info('=' * 60)
            try:
                r = evaluate_one(cat, mode, embeddings_cache)
                per_run.append(r)
                logger.info('[%s/%s] aggregate: cells_acc=%.4f cells_f1=%.4f (n_attrs=%d, n_cells=%d)',
                            cat, mode, r['cells_weighted_accuracy'], r['cells_weighted_macro_f1'],
                            r['n_attrs'], r['n_total_cells'])
            except Exception as e:
                logger.exception('FAIL %s / %s: %s', cat, mode, e)
                per_run.append({'category': cat, 'mode': mode, 'status': 'fail', 'error': str(e)})

    deltas = compute_global_deltas(per_run)

    out = {
        'created': pd.Timestamp.now().isoformat(),
        'description': 'brand-disjoint side-study eval (тикет 2026-05-28). '
                       'Compares ML-only classifier metrics on brand-disjoint vs code-disjoint test sets. '
                       'Source data: silver_extended (~1250 labeled codes/cat). '
                       'No regex/Bayes/LLM — pure ML layer comparison.',
        'categories': CATEGORIES,
        'per_run': per_run,
        **deltas,
    }
    out_path = Path(PROCESSED_DIR) / 'v4_brand_disjoint_eval.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info('Saved: %s', out_path)
    # Print headline
    if deltas.get('global_summary'):
        g = deltas['global_summary']
        logger.info('GLOBAL HEADLINE: code-disjoint acc=%.4f, brand-disjoint acc=%.4f, delta=%+.2f pp',
                    g['global_code_disjoint_accuracy'], g['global_brand_disjoint_accuracy'],
                    g['global_delta_accuracy_pp'])
        logger.info('GLOBAL F1: code-disjoint f1=%.4f, brand-disjoint f1=%.4f, delta=%+.4f',
                    g['global_code_disjoint_macro_f1'], g['global_brand_disjoint_macro_f1'],
                    g['global_delta_macro_f1'])


if __name__ == '__main__':
    main()
