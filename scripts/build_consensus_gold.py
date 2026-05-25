"""Build manual_gold_consensus from 3 LLM relabellings.

Reads {cat}_consensus_{model}.parquet for each model, performs majority vote:
- ≥2 models agree on (code, attr) value → accept it
- 3-way split → mark as 'disputed', exclude from main gold

Output: datasets/processed/manual_gold_consensus.parquet
Schema: code, category, attr, gold_value, n_voters, agreement_ratio, disputed
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

for root in ['/home/miafrolov/Desktop/diploma',
             '/Users/miafrolov/Desktop/stuff/ai_attributes']:
    if Path(root).exists():
        sys.path.insert(0, root)
        PROJECT_ROOT = Path(root)
        break


def norm(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in ('', 'none', 'null', 'nan'):
        return None
    return s


def parse_json(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return {}
    try:
        d = json.loads(s) if isinstance(s, str) else (s or {})
    except Exception:
        return {}
    return {k: norm(v) for k, v in d.items()}


MODELS = [
    'qwen_qwen3.7-max',
    'deepseek_deepseek-r1',
    'mistralai_mistral-large-2411',
]


def build_cat(cat: str, consensus_dir: Path) -> pd.DataFrame:
    print(f'\n=== {cat.upper()} ===')
    per_model = {}
    for slug in MODELS:
        path = consensus_dir / f'{cat}_consensus_{slug}.parquet'
        if not path.exists():
            print(f'  [skip] no {path.name}')
            continue
        df = pd.read_parquet(path)
        df = df[df.parse_status == True].copy()
        df['code'] = df['code'].astype(str)
        per_model[slug] = df
        print(f'  {slug:35s}: {len(df)} successful labels')
    if len(per_model) < 2:
        raise SystemExit(f'Need ≥2 models for consensus, got {len(per_model)}')

    # Build vote table: (code, attr) → list of (model, value)
    votes = defaultdict(list)
    for model_slug, df in per_model.items():
        for _, r in df.iterrows():
            parsed = parse_json(r['parsed_json'])
            for attr, val in parsed.items():
                votes[(r['code'], attr)].append((model_slug, val))

    # Majority vote per cell
    rows = []
    n_consensus = 0
    n_disputed = 0
    n_unanimous = 0
    for (code, attr), entries in votes.items():
        n_voters = len(entries)
        # Drop None votes for purpose of consensus — they're "abstentions"
        non_null = [(m, v) for m, v in entries if v is not None]
        if not non_null:
            continue  # no opinion
        cnt = Counter(v for _, v in non_null)
        top_value, top_count = cnt.most_common(1)[0]
        agreement = top_count / n_voters
        if top_count >= 2:
            disputed = False
            n_consensus += 1
            if top_count == n_voters and len(cnt) == 1:
                n_unanimous += 1
        else:
            disputed = True
            n_disputed += 1
        rows.append({
            'category': cat, 'code': code, 'attr': attr,
            'gold_value': top_value,
            'n_voters': n_voters,
            'n_non_null': len(non_null),
            'top_count': top_count,
            'agreement_ratio': agreement,
            'disputed': disputed,
            'votes_repr': str(cnt.most_common()),
        })

    out = pd.DataFrame(rows)
    print(f'  cells with opinions: {len(out)}')
    print(f'    consensus (≥2 agree): {n_consensus}')
    print(f'      unanimous (3/3):    {n_unanimous}')
    print(f'    disputed (no majority): {n_disputed}')
    if len(out) > 0:
        print(f'    mean agreement_ratio: {out.agreement_ratio.mean():.3f}')
    return out


def main():
    consensus_dir = PROJECT_ROOT / 'datasets/processed/consensus_manual'
    out_path = PROJECT_ROOT / 'datasets/processed/manual_gold_consensus.parquet'

    all_dfs = []
    for cat in ['pasta', 'chocolate', 'cheeses']:
        try:
            df = build_cat(cat, consensus_dir)
            all_dfs.append(df)
        except SystemExit as e:
            print(f'  ERROR: {e}')
            continue

    if not all_dfs:
        print('No data')
        return
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_parquet(out_path, index=False)
    print(f'\nSaved: {out_path}')
    print(f'  total rows: {len(combined)}')
    print(f'  by category:')
    for cat in combined.category.unique():
        sub = combined[combined.category == cat]
        non_disp = sub[~sub.disputed]
        print(f'    {cat}: {len(sub)} cells, consensus {len(non_disp)} '
              f'({100*len(non_disp)/len(sub):.0f}%)')


if __name__ == '__main__':
    main()
