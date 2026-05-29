"""LLM-consensus re-annotation of manual gold (~150 codes × 3 cats).

3 truly independent strong models (different families, NOT in cascade):
  - qwen/qwen3.7-max         (Alibaba, generalist top)
  - deepseek/deepseek-r1     (DeepSeek reasoning — different from v4-flash trainer)
  - mistralai/mistral-large-2411  (Mistral, fully independent European family)

GPT-OSS-120B intentionally excluded — it's the Layer 4 fallback model in the
cascade; using it as gold-creator would create circularity in eval.
DeepSeek-V4-flash is used for training labels (v6_relabel); its R1 sibling is
fundamentally different (RL-tuned reasoning vs SFT flash) but lineage is shared,
so треть голос от Mistral дополнительно защищает от correlated errors.

Each model relabels the manual_eval_per_product codes with current schemas.
Output: {cat}_consensus_manual.parquet with one row per (code, model).

Then build_consensus_gold.py merges to majority vote → manual_gold_consensus.parquet.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

env_path = Path('/home/miafrolov/Desktop/diploma/.env')
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if '=' in line and not line.strip().startswith('#'):
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip()

for root in ['/home/miafrolov/Desktop/diploma',
             '/Users/miafrolov/Desktop/stuff/ai_attributes']:
    if Path(root).exists():
        sys.path.insert(0, root)
        PROJECT_ROOT = Path(root)
        break

import pandas as pd
import duckdb

from src.llm.client import call_openrouter
from src.llm.parsing import _parse_with_status
from src.pipeline.llm_fallback.prompts import build_prompt
from src.pipeline.schemas import PASTA_SCHEMA, CHOCOLATE_SCHEMA, CHEESES_SCHEMA

SCHEMAS = {'pasta': PASTA_SCHEMA, 'chocolate': CHOCOLATE_SCHEMA, 'cheeses': CHEESES_SCHEMA}
SAVE_LOCK = threading.Lock()

# Re-use helpers from qwen_arbitrate.py for STRUCT[] handling
from src.eval.qwen_arbitrate import build_product_dict


def process_one(prod, schema, model, api_key, max_tokens=1024,
                 max_retries=3, base_backoff=4.0):
    if not prod['ingredients_text'] or not prod['product_name']:
        return None
    last_err = None
    for attempt in range(max_retries):
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
            return {
                'code': prod['code'], 'model': model,
                'product_name': prod['product_name'][:200],
                'in_tokens': in_tok, 'out_tokens': out_tok,
                'latency_ms': latency, 'parse_status': status,
                'raw': raw[:2000],
                'parsed_json': json.dumps(parsed) if parsed else None,
            }
        except Exception as e:
            last_err = str(e)[:200]
            # Retry on transient errors: 429 (rate-limit), 403 (key/quota race),
            # 5xx, and connection timeouts.
            retriable = any(s in last_err for s in
                            ('429', '403', '500', '502', '503', '504',
                             'timeout', 'Timeout', 'Connection'))
            if retriable and attempt < max_retries - 1:
                sleep = base_backoff * (2 ** attempt)
                time.sleep(sleep)
                continue
            return {'code': prod['code'], 'model': model, 'error': last_err}


def append_batch(buf, out_path):
    if not buf:
        return
    with SAVE_LOCK:
        new_df = pd.DataFrame(buf)
        if Path(out_path).exists():
            prev = pd.read_parquet(out_path)
            new_df = pd.concat([prev, new_df], ignore_index=True)
        new_df.to_parquet(out_path, index=False)


def run_cat_model(cat, model, workers, max_tokens, api_key, out_dir, off_dir,
                  manual_path=None):
    schema = SCHEMAS[cat]
    if manual_path is None:
        manual_path = PROJECT_ROOT / 'datasets/processed/manual_eval_per_product.parquet'
    manual = pd.read_parquet(manual_path)
    manual['code'] = manual['code'].astype(str)
    codes = set(manual[manual.category == cat]['code'].unique())
    print(f'[{cat}/{model}] target codes: {len(codes)}', flush=True)

    code_sql = ','.join(f"'{c}'" for c in codes)
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT code, product_name, brands, ingredients_text, quantity,
               categories_tags, labels_tags, nutriments
        FROM '{off_dir / f"{cat}_off_full.parquet"}'
        WHERE CAST(code AS VARCHAR) IN ({code_sql})
    """).fetchdf()

    # Also try full food.parquet for codes not in filtered cat parquet (some pasta
    # codes aren't tagged en:pastas — manual gold may have rice products etc.)
    found_codes = set(df.code.astype(str))
    missing = codes - found_codes
    if missing:
        miss_sql = ','.join(f"'{c}'" for c in missing)
        food_path = off_dir / 'food.parquet'
        if food_path.exists():
            extra = con.execute(f"""
                SELECT code, product_name, brands, ingredients_text, quantity,
                       categories_tags, labels_tags, nutriments
                FROM '{food_path}'
                WHERE CAST(code AS VARCHAR) IN ({miss_sql})
            """).fetchdf()
            if len(extra):
                df = pd.concat([df, extra], ignore_index=True)
                print(f'[{cat}/{model}] +{len(extra)} from full food.parquet', flush=True)

    print(f'[{cat}/{model}] inputs loaded: {len(df)}/{len(codes)}', flush=True)

    model_slug = model.replace('/', '_').replace(':', '_')
    out_path = out_dir / f'{cat}_consensus_{model_slug}.parquet'

    done = set()
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        done = set(prev.code.astype(str))
    df = df[~df.code.astype(str).isin(done)]
    if len(df) == 0:
        print(f'[{cat}/{model}] nothing to do (already done)', flush=True)
        return

    buf, n_ok, n_fail = [], 0, 0
    t0 = time.time()
    flush_threshold = 5
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
                if n_fail <= 5:
                    print(f'  err: {r.get("error", "")[:160]}', flush=True)
            else:
                buf.append(r); n_ok += 1
            if len(buf) >= flush_threshold:
                append_batch(buf, str(out_path))
                buf = []
            if i % 5 == 0:
                elapsed = time.time() - t0
                print(f'[{cat}/{model}] {i}/{len(df)} ok={n_ok} fail={n_fail} '
                      f'rate={i/elapsed:.2f}/s', flush=True)
    append_batch(buf, str(out_path))
    print(f'[{cat}/{model}] DONE ok={n_ok} fail={n_fail} '
          f'time={(time.time()-t0)/60:.1f}min', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cats', nargs='+', default=['pasta', 'chocolate', 'cheeses'])
    p.add_argument('--models', nargs='+', default=[
        'qwen/qwen3.7-max',
        'deepseek/deepseek-r1',
        'mistralai/mistral-large-2411',
    ])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--manual-path', default=None,
                   help='alt path to source parquet (default: manual_eval_per_product.parquet)')
    p.add_argument('--out-dir', default=None,
                   help='alt output directory (default: datasets/processed/consensus_manual)')
    args = p.parse_args()

    api_key = os.environ['OPENROUTER_API_KEY']
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / 'datasets/processed/consensus_manual'
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    off_dir = Path.home() / 'off_work'
    if not off_dir.exists():
        off_dir = PROJECT_ROOT / 'datasets/raw'

    manual_path = Path(args.manual_path) if args.manual_path else None

    for cat in args.cats:
        for model in args.models:
            run_cat_model(cat, model, args.workers, args.max_tokens,
                          api_key, out_dir, off_dir, manual_path=manual_path)


if __name__ == '__main__':
    main()
