"""Build brand-disjoint train/test split for side-study (тикет 2026-05-28).

Покрытие: pasta, chocolate, cheeses (3 main food categories с densely labeled silver_extended).

Логика (REVISED 2026-05-28 — после обнаружения, что labelled data живёт в silver_extended ~1250 codes/cat):

  - Источник labelled data — `{cat}_stratified_silver_extended.parquet` (~1250 codes/cat).
  - Это единственный pool, где есть labels на ВСЕХ codes (silver_standard
    имеет labels только на ~1250 manual_gold_consensus codes из 13-21k полных rows).
  - Brand-disjoint split: greedy выбираем 20% codes → test, удаляем из train pool
    все codes которые делят brand с test.
  - Code-disjoint baseline: 20% random codes → test, остальные → train (brand overlap допускается).

  Артефакт (для каждой категории):
    - `{cat}_brand_disjoint_train_codes.parquet`
    - `{cat}_brand_disjoint_test_codes.parquet`
    - `{cat}_code_disjoint_train_codes.parquet` (baseline для сравнения)
    - `{cat}_code_disjoint_test_codes.parquet`
    Плюс summary JSON.

Метрики split: n_train, n_test, n_unique_brands_train, n_unique_brands_test, overlap.

Usage:
  python src/eval/build_brand_disjoint_split.py
"""
from __future__ import annotations

import json
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np

for root in ['/home/miafrolov/Desktop/diploma',
             '/Users/miafrolov/Desktop/stuff/ai_attributes']:
    if Path(root).exists():
        PROJECT_ROOT = Path(root)
        sys.path.insert(0, str(root))
        break

CATEGORIES = ['pasta', 'chocolate', 'cheeses']
TEST_FRAC = 0.20
RANDOM_SEED = 42


def tokenize_brands(s: str | None) -> set[str]:
    """Brand string → set of normalized tokens.

    OFF brands формат: "Brand A, Brand B" или "Brand A,Brand B" — comma-separated.
    Normalize: lowercase, strip, drop empty.
    """
    if not s or pd.isna(s):
        return set()
    parts = re.split(r'[,;/]', str(s))
    tokens = set()
    for p in parts:
        p = p.strip().lower()
        if p and len(p) > 1:
            tokens.add(p)
    return tokens


def greedy_brand_disjoint_split(df: pd.DataFrame, test_frac: float, seed: int) -> tuple[set[str], set[str]]:
    """Greedy brand-disjoint split with iterative brand-closure.

    Algorithm:
      1. Shuffle brands; seed test bucket by accumulating brands until ~test_frac codes covered.
      2. Iterate closure: any code that touches a test brand goes to test;
         any new brand on such a code is added to test_brands; repeat until fixed point.
         This produces TRUE brand-disjoint (0% overlap) — train ∩ test = ∅ at brand level.
      3. Train = remaining codes (whose brands never overlap test_brands).
      4. No-brand codes split randomly by test_frac.

    Returns (train_codes, test_codes, dropped_codes).
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df['code'] = df['code'].astype(str)
    df['brand_tokens'] = df['brands'].apply(tokenize_brands)
    df['has_brand'] = df['brand_tokens'].apply(bool)

    no_brand_df = df[~df['has_brand']].copy()
    has_brand_df = df[df['has_brand']].copy()

    # Build brand → codes and code → brands indices
    code_to_brands: dict[str, set[str]] = {}
    brand_to_codes: dict[str, set[str]] = {}
    for _, row in has_brand_df.iterrows():
        code = row['code']
        brands = row['brand_tokens']
        code_to_brands[code] = brands
        for b in brands:
            brand_to_codes.setdefault(b, set()).add(code)

    brand_list = list(brand_to_codes.keys())
    rng.shuffle(brand_list)

    target_n_test = int(round(test_frac * len(df)))
    test_brands: set[str] = set()
    test_codes: set[str] = set()

    # Phase 1: seed test bucket by adding rare-first brands AND simulating closure
    # incrementally. Stop when n_test_codes ≥ target. Skip a brand if adding it
    # (with closure) would overshoot by >25%.
    brand_sizes = [(b, len(brand_to_codes[b])) for b in brand_list]
    brand_sizes.sort(key=lambda x: (x[1], x[0]))  # asc by size, then alpha for determinism

    def simulate_closure(seed_brands: set[str], seed_codes: set[str]) -> tuple[set[str], set[str]]:
        """Return (final_codes, final_brands) after running closure from given seed."""
        tb = set(seed_brands)
        tc = set(seed_codes)
        # Absorb brands of all initial codes
        for c in list(tc):
            tb |= code_to_brands.get(c, set())
        changed = True
        while changed:
            changed = False
            for cd, brs in code_to_brands.items():
                if cd in tc:
                    continue
                if brs & tb:
                    tc.add(cd)
                    nb = brs - tb
                    if nb:
                        tb |= nb
                        changed = True
        return tc, tb

    for b, _sz in brand_sizes:
        if len(test_codes) >= target_n_test:
            break
        # Try adding b + closure; if overshoot >25%, skip.
        candidate_brands = test_brands | {b}
        candidate_codes = test_codes | brand_to_codes[b]
        closed_codes, closed_brands = simulate_closure(candidate_brands, candidate_codes)
        if test_codes and len(closed_codes) > 1.25 * target_n_test:
            continue
        # Commit
        test_brands = closed_brands
        test_codes = closed_codes

    # After seed: also absorb ALL brands of codes currently in test_codes
    # (a multi-brand code added via one of its brands brings the others too).
    for code in list(test_codes):
        test_brands |= code_to_brands.get(code, set())

    # Phase 2: closure — any code touching test_brands must be in test;
    # repeat until fixed point.
    changed = True
    while changed:
        changed = False
        for code, brands in code_to_brands.items():
            if code in test_codes:
                continue
            if brands & test_brands:
                test_codes.add(code)
                new_brands = brands - test_brands
                if new_brands:
                    test_brands |= new_brands
                    changed = True

    # Phase 3: train = has-brand codes never touching test_brands
    train_codes: set[str] = set()
    dropped_codes: set[str] = set()
    for code, brands in code_to_brands.items():
        if code in test_codes:
            continue
        # invariant: after closure, this should hold for all non-test codes
        assert not (brands & test_brands), f'closure bug: {code} {brands} ∩ {brands & test_brands}'
        train_codes.add(code)

    # No-brand codes split randomly by test_frac
    no_brand_codes = no_brand_df['code'].tolist()
    rng.shuffle(no_brand_codes)
    n_nb_test = int(round(test_frac * len(no_brand_codes)))
    test_codes |= set(no_brand_codes[:n_nb_test])
    train_codes |= set(no_brand_codes[n_nb_test:])

    return train_codes, test_codes, dropped_codes


def code_disjoint_split(df: pd.DataFrame, test_frac: float, seed: int) -> tuple[set[str], set[str]]:
    """Baseline random code-disjoint split (no brand filter)."""
    rng = np.random.RandomState(seed)
    codes = df['code'].astype(str).tolist()
    rng.shuffle(codes)
    n_test = int(round(test_frac * len(codes)))
    test_codes = set(codes[:n_test])
    train_codes = set(codes[n_test:])
    return train_codes, test_codes


def measure_brand_overlap(df: pd.DataFrame, train_codes: set[str], test_codes: set[str]) -> dict:
    """Compute brand overlap stats between train and test."""
    df = df.copy()
    df['code'] = df['code'].astype(str)
    df['brand_tokens'] = df['brands'].apply(tokenize_brands)

    train_brands: set[str] = set()
    for toks in df[df.code.isin(train_codes)]['brand_tokens']:
        train_brands |= toks
    test_brands: set[str] = set()
    for toks in df[df.code.isin(test_codes)]['brand_tokens']:
        test_brands |= toks
    overlap = train_brands & test_brands
    return {
        'n_unique_brands_train': len(train_brands),
        'n_unique_brands_test': len(test_brands),
        'n_brand_overlap': len(overlap),
        'pct_test_brands_overlap': (
            100.0 * len(overlap) / len(test_brands) if test_brands else 0.0
        ),
    }


def build_split_for_cat(cat: str) -> dict:
    """Build BOTH brand-disjoint AND code-disjoint splits for one category.

    Source data — silver_extended (densely labeled ~1250 codes).
    """
    silver_path = PROJECT_ROOT / f'datasets/processed/{cat}_stratified_silver_extended.parquet'
    silver = pd.read_parquet(silver_path)
    silver['code'] = silver['code'].astype(str)
    print(f"\n=== {cat} ===")
    print(f"  silver_extended (labeled pool): {len(silver)} rows")

    # 1. Brand-disjoint split
    bd_train, bd_test, bd_dropped = greedy_brand_disjoint_split(silver, TEST_FRAC, RANDOM_SEED)
    bd_metrics = measure_brand_overlap(silver, bd_train, bd_test)
    print(f"  [brand_disjoint] train={len(bd_train)}, test={len(bd_test)}, "
          f"dropped={len(bd_dropped)}, brand_overlap={bd_metrics['n_brand_overlap']}")

    # 2. Code-disjoint baseline split (same seed, no brand filter)
    cd_train, cd_test = code_disjoint_split(silver, TEST_FRAC, RANDOM_SEED)
    cd_metrics = measure_brand_overlap(silver, cd_train, cd_test)
    print(f"  [code_disjoint] train={len(cd_train)}, test={len(cd_test)}, "
          f"brand_overlap={cd_metrics['n_brand_overlap']} "
          f"({cd_metrics['pct_test_brands_overlap']:.1f}% of test brands)")

    out_dir = PROJECT_ROOT / 'datasets/processed'

    pd.DataFrame({'code': sorted(bd_train)}).to_parquet(
        out_dir / f'{cat}_brand_disjoint_train_codes.parquet', index=False)
    pd.DataFrame({'code': sorted(bd_test)}).to_parquet(
        out_dir / f'{cat}_brand_disjoint_test_codes.parquet', index=False)
    pd.DataFrame({'code': sorted(cd_train)}).to_parquet(
        out_dir / f'{cat}_code_disjoint_train_codes.parquet', index=False)
    pd.DataFrame({'code': sorted(cd_test)}).to_parquet(
        out_dir / f'{cat}_code_disjoint_test_codes.parquet', index=False)

    return {
        'category': cat,
        'source_data': f'{cat}_stratified_silver_extended.parquet',
        'n_silver_rows': int(len(silver)),
        'test_frac_target': TEST_FRAC,
        'random_seed': RANDOM_SEED,
        'brand_disjoint': {
            'n_train_codes': int(len(bd_train)),
            'n_test_codes': int(len(bd_test)),
            'n_dropped_brand_overlap': int(len(bd_dropped)),
            **{k: int(v) if isinstance(v, (int, np.integer)) else float(v)
               for k, v in bd_metrics.items()},
        },
        'code_disjoint': {
            'n_train_codes': int(len(cd_train)),
            'n_test_codes': int(len(cd_test)),
            **{k: int(v) if isinstance(v, (int, np.integer)) else float(v)
               for k, v in cd_metrics.items()},
        },
    }


def main():
    results = [build_split_for_cat(cat) for cat in CATEGORIES]

    summary = {
        'created': pd.Timestamp.now().isoformat(),
        'note': 'brand-disjoint side-study split v2 (тикет 2026-05-28). '
                'Source: silver_extended (densely labeled subset ~1250 codes/cat). '
                'Two splits per cat: brand_disjoint (greedy) and code_disjoint (random baseline).',
        'per_category': results,
    }
    out_json = PROJECT_ROOT / 'datasets/processed/brand_disjoint_split_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {out_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
