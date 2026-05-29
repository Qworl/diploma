"""Train ML classifiers on brand-disjoint split (тикет 2026-05-28, v2).

Architecture (REVISED 2026-05-28 после первого прогона):

  Источник labelled data — `{cat}_stratified_silver_extended.parquet` (~1250 codes/cat).
  Это единственный pool с плотной разметкой; silver_standard имеет labels только
  на ~1250 manual_gold_consensus codes из 13-21k полных rows (всё остальное NaN).

  Embeddings: пересчитываются на-лету через get_embeddings() для silver_extended subset.
  Это не использует существующий *_stratified_embeddings.npy кэш (он на 15k rows
  для silver_standard, не выровнен с 1250-row silver_extended).

  Брэнд-disjoint split codes pre-built script'ом build_brand_disjoint_split.py:
    - `{cat}_brand_disjoint_train_codes.parquet`
    - `{cat}_brand_disjoint_test_codes.parquet`
    - `{cat}_code_disjoint_train_codes.parquet` (baseline)
    - `{cat}_code_disjoint_test_codes.parquet`

  Output models суффиксом `_brand_disjoint` / `_code_disjoint`:
    cheeses_stratified_brand_disjoint_milk_source_xgb.pkl, etc.

Usage (на VM):
    cd ~/Desktop/diploma
    source .venv/bin/activate
    nohup nice -n 10 python -m src.eval.train_brand_disjoint > brand_disjoint_train.log 2>&1 &

Запускать ТОЛЬКО на VM (правило проекта: train_on_vm_only).
"""
from __future__ import annotations

import os
import sys
import pickle
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Project root detection
for root in ['/home/miafrolov/Desktop/diploma',
             '/Users/miafrolov/Desktop/stuff/ai_attributes']:
    if Path(root).exists():
        PROJECT_ROOT = Path(root)
        sys.path.insert(0, str(root))
        break

from src.common import (
    MODELS_DIR, PROCESSED_DIR,
    build_text, get_embeddings, setup_logging,
)
from src.pipeline.ml.train import (
    CATEGORY_CONFIG, train_multiclass, train_binary,
)

logger = logging.getLogger(__name__)

CATEGORIES = ['pasta', 'chocolate', 'cheeses']


def load_split_codes(cat: str, mode: str) -> tuple[set[str], set[str]]:
    """Load pre-built {cat}_{mode}_(train|test)_codes.parquet files."""
    train_path = Path(PROCESSED_DIR) / f'{cat}_{mode}_train_codes.parquet'
    test_path = Path(PROCESSED_DIR) / f'{cat}_{mode}_test_codes.parquet'
    train_codes = set(pd.read_parquet(train_path)['code'].astype(str).tolist())
    test_codes = set(pd.read_parquet(test_path)['code'].astype(str).tolist())
    logger.info('[%s/%s] codes loaded: train=%d, test=%d',
                cat, mode, len(train_codes), len(test_codes))
    return train_codes, test_codes


def train_one_category(cat: str, mode: str = 'brand_disjoint',
                       embeddings_cache: dict[str, np.ndarray] | None = None):
    """Train all classifiers for one category in given split mode.

    mode='brand_disjoint': use brand-disjoint train/test codes.
    mode='code_disjoint':  baseline random split (same TEST_FRAC, same seed, no brand filter).

    Uses silver_extended as labelled data source.
    Embeddings — fresh compute (cache key = cat) shared across modes for speed.
    """
    assert mode in ('brand_disjoint', 'code_disjoint')
    category_key = f'{cat}_stratified'
    config = CATEGORY_CONFIG[category_key]

    suffix = f'_{mode}'
    model_prefix = f'{cat}_stratified{suffix}'

    # Use silver_extended (labelled subset) — NOT silver_standard (sparsely labelled).
    ss_filename = config['silver_standard'].replace('_silver_standard', '_silver_extended')
    ss_path = Path(PROCESSED_DIR) / ss_filename

    logger.info('=' * 60)
    logger.info('[%s/%s] training prefix=%s, source=%s', cat, mode, model_prefix, ss_filename)
    logger.info('=' * 60)

    df = pd.read_parquet(ss_path)
    df['code'] = df['code'].astype(str)
    logger.info('  silver_extended loaded: %d rows', len(df))

    # Apply train/test code masks
    train_codes, test_codes = load_split_codes(cat, mode)
    train_df = df[df['code'].isin(train_codes)].reset_index(drop=True).copy()
    test_df = df[df['code'].isin(test_codes)].reset_index(drop=True).copy()
    logger.info('  split: train=%d rows, test=%d rows (sum=%d, total=%d)',
                len(train_df), len(test_df), len(train_df) + len(test_df), len(df))

    # Embeddings: compute or reuse from cache by category
    if embeddings_cache is not None and cat in embeddings_cache:
        X_train, X_test = embeddings_cache[cat][mode]
        logger.info('  using cached embeddings: train=%s, test=%s',
                    X_train.shape, X_test.shape)
    else:
        train_texts = build_text(train_df)
        test_texts = build_text(test_df)
        logger.info('  computing embeddings for train (n=%d) and test (n=%d)',
                    len(train_texts), len(test_texts))
        # Pass cache_path=None to skip cache file (silver_extended is small ~1k rows)
        X_train = get_embeddings(train_texts, cache_path=None)
        X_test = get_embeddings(test_texts, cache_path=None)
        if embeddings_cache is not None:
            embeddings_cache.setdefault(cat, {})[mode] = (X_train, X_test)
    logger.info('  embedding shapes: train=%s, test=%s', X_train.shape, X_test.shape)

    thresholds = {}
    for attr_name, attr_config in config['classifiers'].items():
        if attr_name not in df.columns:
            logger.warning('  skip %s: not in silver_extended', attr_name)
            continue

        if attr_config['type'] == 'multiclass':
            y_train_full = train_df[attr_name].copy()
            y_test_full = test_df[attr_name].copy()
            train_mask_attr = y_train_full.notna()
            counts = y_train_full[train_mask_attr].value_counts()
            valid = counts[counts >= attr_config['min_samples']].index
            if 'exclude_classes' in attr_config:
                valid = valid.difference(attr_config['exclude_classes'])
            train_mask_attr = train_mask_attr & y_train_full.isin(valid)
            test_mask_attr = y_test_full.notna() & y_test_full.isin(valid)

            if y_train_full[train_mask_attr].nunique() < 2:
                logger.warning('  skip %s: <2 classes in train', attr_name)
                continue
            if test_mask_attr.sum() < 5:
                logger.warning('  skip %s: test too small (%d)', attr_name, int(test_mask_attr.sum()))
                continue

            try:
                _, _, t = train_multiclass(
                    X_train[train_mask_attr.values], X_test[test_mask_attr.values],
                    y_train_full[train_mask_attr], y_test_full[test_mask_attr],
                    attr_name, model_prefix,
                    do_calibrate=True, calibration_method='sigmoid',
                )
                thresholds[attr_name] = float(t) if hasattr(t, 'item') else t
            except Exception as e:
                logger.error('  FAIL %s: %s', attr_name, e)

        elif attr_config['type'] == 'binary':
            y_train_full = train_df[attr_name]
            y_test_full = test_df[attr_name]
            train_mask_attr = y_train_full.notna()
            test_mask_attr = y_test_full.notna()
            if train_mask_attr.sum() < 30:
                logger.warning('  skip %s: train too small (%d)', attr_name, int(train_mask_attr.sum()))
                continue

            y_train_int = y_train_full[train_mask_attr].astype(bool).astype(int)
            y_test_int = y_test_full[test_mask_attr].astype(bool).astype(int)
            if y_train_int.sum() < attr_config['min_positive']:
                logger.warning('  skip %s: only %d positive', attr_name, int(y_train_int.sum()))
                continue
            if test_mask_attr.sum() < 5:
                logger.warning('  skip %s: test too small', attr_name)
                continue

            try:
                _, t = train_binary(
                    X_train[train_mask_attr.values], X_test[test_mask_attr.values],
                    y_train_int, y_test_int,
                    attr_name, model_prefix,
                    do_calibrate=True, calibration_method='sigmoid',
                )
                thresholds[attr_name] = float(t) if hasattr(t, 'item') else t
            except Exception as e:
                logger.error('  FAIL %s: %s', attr_name, e)

    thresh_path = Path(MODELS_DIR) / f'{model_prefix}_thresholds.pkl'
    thresh_path.parent.mkdir(parents=True, exist_ok=True)
    with open(thresh_path, 'wb') as f:
        pickle.dump(thresholds, f)
    logger.info('[%s/%s] DONE. trained_attrs=%s. models saved with prefix=%s',
                cat, mode, list(thresholds.keys()), model_prefix)
    return thresholds


def main():
    setup_logging()
    summary = {'created': pd.Timestamp.now().isoformat(), 'per_run': []}
    # No shared cache between brand_disjoint and code_disjoint because splits differ.
    # Embeddings are recomputed each time (1250 rows × MiniLM-L12 → ~10s/cat).
    for mode in ['code_disjoint', 'brand_disjoint']:
        for cat in CATEGORIES:
            logger.info('### START %s / %s ###', cat, mode)
            try:
                t = train_one_category(cat, mode=mode)
                summary['per_run'].append({
                    'category': cat, 'mode': mode, 'status': 'ok',
                    'n_attrs_trained': len(t),
                    'attrs': list(t.keys()),
                    'thresholds': t,
                })
            except Exception as e:
                logger.exception('### FAIL %s / %s: %s', cat, mode, e)
                summary['per_run'].append({
                    'category': cat, 'mode': mode, 'status': 'fail',
                    'error': str(e),
                })

    out_path = Path(PROCESSED_DIR) / 'brand_disjoint_train_summary.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info('Summary written: %s', out_path)


if __name__ == '__main__':
    main()
