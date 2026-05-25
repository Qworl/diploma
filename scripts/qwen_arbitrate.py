"""Qwen 3 Max arbitration pilot.

Takes N codes per cat from v5_relabel/{cat}_relabel_v5.parquet (successful parses),
re-labels via qwen/qwen3-max with the same prompt template, saves to
datasets/processed/qwen3max_arb/{cat}_qwen3max_arb.parquet.

Then compares per-attr agreement v5 (DeepSeek-V4-flash) vs Qwen 3 Max.

Usage:
    python scripts/qwen_arbitrate.py --n 100 --workers 20
"""
import os
import sys
import json
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Env on VM
env_path = Path.home() / 'Desktop/diploma/.env'
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if '=' in line and not line.strip().startswith('#'):
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip()

# Project root on VM vs local
for root_candidate in ['/home/miafrolov/Desktop/diploma',
                       '/Users/miafrolov/Desktop/stuff/ai_attributes']:
    if Path(root_candidate).exists():
        sys.path.insert(0, root_candidate)
        PROJECT_ROOT = Path(root_candidate)
        break

import pandas as pd
import duckdb

from src.llm.client import call_openrouter
from src.llm.parsing import _parse_with_status
from src.pipeline.llm_fallback.prompts import build_prompt
from src.pipeline.schemas import PASTA_SCHEMA, CHOCOLATE_SCHEMA, CHEESES_SCHEMA

SCHEMAS = {'pasta': PASTA_SCHEMA, 'chocolate': CHOCOLATE_SCHEMA, 'cheeses': CHEESES_SCHEMA}
SAVE_LOCK = threading.Lock()


def _pick_text(struct_arr, prefer=('main', 'en', 'fr', 'de', 'es', 'it')):
    if struct_arr is None:
        return None
    try:
        items = list(struct_arr) if not isinstance(struct_arr, list) else struct_arr
    except TypeError:
        return None
    by_lang = {}
    for it in items:
        if isinstance(it, dict):
            text = str(it.get('text', '') or '').strip()
            if text:
                by_lang[it.get('lang', '')] = text
    for p in prefer:
        if p in by_lang:
            return by_lang[p]
    return next(iter(by_lang.values())) if by_lang else None


def _flatten_nutriments(nut_arr):
    if nut_arr is None:
        return {}
    try:
        items = list(nut_arr) if not isinstance(nut_arr, list) else nut_arr
    except TypeError:
        return {}
    out = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get('name')
        per100 = it.get('100g')
        if name and per100 is not None:
            try:
                out[f'{name}_100g'] = float(per100)
            except (ValueError, TypeError):
                pass
    return out


def _safe_list(v):
    try:
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else None
    except Exception:
        return None


def build_product_dict(row):
    return {
        'code': str(row['code']),
        'product_name': _pick_text(row.get('product_name')),
        'brands': row.get('brands'),
        'ingredients_text': _pick_text(row.get('ingredients_text')),
        'quantity': row.get('quantity'),
        'nutriments': _flatten_nutriments(row.get('nutriments')),
        'categories_tags': _safe_list(row.get('categories_tags')),
        'labels_tags': _safe_list(row.get('labels_tags')),
    }


def process_one(prod, schema, model, api_key, max_tokens=512):
    if not prod['ingredients_text'] or not prod['product_name']:
        return None
    try:
        prompt = build_prompt(prod, schema, include_examples=True)
        t1 = time.time()
        result = call_openrouter(
            messages=[{'role': 'user', 'content': prompt}],
            model=model, api_key=api_key,
            max_tokens=max_tokens, enforce_json=True,
        )
        latency = (time.time() - t1) * 1000
        raw = result.get('raw') or ''
        parsed, status = _parse_with_status(raw, schema)
        usage = result.get('usage', {})
        in_tok = usage.get('prompt_tokens', 0)
        out_tok = usage.get('completion_tokens', 0)
        # Qwen 3 Max pricing on OpenRouter: $1.20/M in, $6.00/M out (approx)
        cost = in_tok * 1.20e-6 + out_tok * 6.00e-6
        return {
            'code': prod['code'], 'model': model,
            'product_name': prod['product_name'][:200],
            'in_tokens': in_tok, 'out_tokens': out_tok, 'cost_usd': cost,
            'latency_ms': latency, 'parse_status': status,
            'raw': raw[:2000],
            'parsed_json': json.dumps(parsed) if parsed else None,
        }
    except Exception as e:
        return {'code': prod['code'], 'error': str(e)[:200]}


def append_batch(buf, out_path):
    if not buf:
        return
    with SAVE_LOCK:
        new_df = pd.DataFrame(buf)
        if Path(out_path).exists():
            prev = pd.read_parquet(out_path)
            new_df = pd.concat([prev, new_df], ignore_index=True)
        new_df.to_parquet(out_path, index=False)


def run_cat(cat, n, workers, model, api_key, v5_dir, off_dir, out_dir, max_tokens=512):
    schema = SCHEMAS[cat]

    v5_path = v5_dir / f'{cat}_relabel_v5.parquet'
    v5 = pd.read_parquet(v5_path)
    ok = v5[v5.parse_status == True].copy()
    # deterministic sample by code-sort
    ok['code'] = ok['code'].astype(str)
    sampled_codes = ok.sort_values('code').head(n)['code'].tolist()
    print(f'[{cat}] sampled {len(sampled_codes)} codes from v5 ({len(ok)} total ok)')

    off_path = off_dir / f'{cat}_off_full.parquet'
    if not off_path.exists():
        # try local raw fallback
        off_path = PROJECT_ROOT / 'datasets/raw/en.openfoodfacts.org.products.parquet'
    code_list_sql = ','.join(f"'{c}'" for c in sampled_codes)
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT code, product_name, brands, ingredients_text, quantity,
               categories_tags, labels_tags, nutriments
        FROM '{off_path}'
        WHERE CAST(code AS VARCHAR) IN ({code_list_sql})
    """).fetchdf()
    print(f'[{cat}] loaded {len(df)} input rows from OFF')

    model_slug = model.replace('/', '_').replace(':', '_')
    out_path = out_dir / f'{cat}_{model_slug}_arb.parquet'
    done = set()
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        done = set(prev.code.astype(str))
        print(f'[{cat}] resume: {len(done)} already done')

    df = df[~df.code.astype(str).isin(done)]
    if len(df) == 0:
        print(f'[{cat}] nothing to do')
        return

    buf, n_ok, n_fail = [], 0, 0
    t0 = time.time()
    total_cost = 0.0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for _, row in df.iterrows():
            prod = build_product_dict(row)
            futs.append(ex.submit(process_one, prod, schema, model, api_key, max_tokens))
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r is None:
                n_fail += 1
            elif 'error' in r:
                n_fail += 1
                if n_fail <= 3:
                    print(f'  err: {r.get("error")[:120]}')
            else:
                buf.append(r)
                n_ok += 1
                total_cost += r['cost_usd']
            if len(buf) >= 20:
                append_batch(buf, str(out_path))
                buf = []
            if i % 25 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f'[{cat}] {i}/{len(df)} ok={n_ok} fail={n_fail} '
                      f'{rate:.1f}/s cost=${total_cost:.3f}')
    append_batch(buf, str(out_path))
    print(f'[{cat}] DONE ok={n_ok} fail={n_fail} '
          f'time={(time.time()-t0)/60:.1f}min cost=${total_cost:.3f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=100, help='sample size per cat')
    p.add_argument('--workers', type=int, default=20)
    p.add_argument('--model', default='qwen/qwen3-max')
    p.add_argument('--cats', nargs='+', default=['pasta', 'chocolate', 'cheeses'])
    p.add_argument('--max-tokens', type=int, default=512,
                   help='max output tokens (raise for thinking models)')
    args = p.parse_args()

    v5_dir = PROJECT_ROOT / 'datasets/processed/v5_relabel'
    off_dir = Path.home() / 'off_work'
    out_dir = PROJECT_ROOT / 'datasets/processed/qwen3max_arb'
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ['OPENROUTER_API_KEY']
    print(f'Model: {args.model}  workers={args.workers}  n={args.n}/cat')

    for cat in args.cats:
        run_cat(cat, args.n, args.workers, args.model, api_key,
                v5_dir, off_dir, out_dir, args.max_tokens)


if __name__ == '__main__':
    main()
