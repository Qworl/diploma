"""Direct LLM baselines on v2 OFF-grounded products with token + latency capture.

For each (model, product) pair: build the same Layer-4 LLM prompt the cascade
would use, send the request, capture token counts (from the provider's `usage`
field), wall-clock latency, and refusal/answer payload.

Two prompt modes (--context-mode):
  partner_input — only product_name + brands + ingredients_text + quantity
                  (what the cascade ML layer sees in production).
  off_grounded  — full curated OFF fields from cache (what blind Opus saw).
                  Requires --off-cache-dir pointing to Phase 1's OFF cache.

Resume-safe: skips codes already present in the output parquet.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.llm.client import call_openrouter
from src.llm.parsing import parse_llm_response
from src.manual_label.schemas_loader import load_domain_attrs
from src.pipeline.llm_fallback.prompts import build_prompt
from src.pipeline.schemas import (
    BEVERAGE_SCHEMA, CEREALS_SCHEMA, CHEESES_SCHEMA, CHOCOLATE_SCHEMA,
    COSMETICS_SCHEMA, PASTA_SCHEMA,
)

logger = logging.getLogger(__name__)

_SCHEMA = {
    "pasta": PASTA_SCHEMA, "chocolate": CHOCOLATE_SCHEMA, "cheeses": CHEESES_SCHEMA,
    "beverages": BEVERAGE_SCHEMA, "cereals": CEREALS_SCHEMA, "cosmetics": COSMETICS_SCHEMA,
}

# OpenRouter prices in USD per 1M tokens, snapshot 2026-05-16.
PRICING = {
    "openai/gpt-oss-120b": {"in": 0.04, "out": 0.18},
    "openai/gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "openai/gpt-5.5": {"in": 5.00, "out": 30.00},
    "google/gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    "google/gemini-3.1-pro-preview": {"in": 2.00, "out": 12.00},
    "anthropic/claude-sonnet-4.5": {"in": 3.00, "out": 15.00},
    "anthropic/claude-opus-4.5": {"in": 5.00, "out": 25.00},
}

_PARTNER_FIELDS = ["product_name", "brands", "ingredients_text", "quantity"]


def _load_off_grounded_fields(code: str, cache_dir: Path) -> dict:
    from src.manual_label.off_field_filter import curate_prompt_fields
    cache_path = Path(cache_dir) / f"{code}.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"OFF cache miss for {code}; run cache population first")
    with cache_path.open(encoding="utf-8") as f:
        off_response = json.load(f)
    # Support both wrapped {"product": {...}} (OFF API v2) and flat (Phase 1 cache) formats.
    product = off_response.get("product") or {}
    if not product:
        # Flat format: the top-level dict IS the product object.
        product = off_response
    return curate_prompt_fields(product)


def run_llm_on_products(
    products: pd.DataFrame,
    *,
    domain: str,
    model: str,
    api_key: str,
    out_path: Path,
    context_mode: str = "partner_input",
    off_cache_dir: Path | None = None,
    max_cost_usd: float = 30.0,
    sleep_between: float = 0.0,
    call_fn: Callable = call_openrouter,
) -> pd.DataFrame:
    """Run `model` on every product row. Resume-safe.

    `call_fn` must return a dict with keys: raw (str), usage
    ({prompt_tokens, completion_tokens}), latency_ms (float).
    """
    if context_mode not in {"partner_input", "off_grounded"}:
        raise ValueError(f"unknown context_mode: {context_mode}")
    if context_mode == "off_grounded" and off_cache_dir is None:
        raise ValueError("off_grounded mode requires off_cache_dir")

    schema = _SCHEMA[domain]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_parquet(out_path) if out_path.exists() else None
    done_codes = set(existing["code"].astype(str)) if existing is not None else set()

    pricing = PRICING.get(model)
    if pricing is None:
        raise KeyError(f"No pricing for {model} — add to PRICING dict")

    rows = list(existing.to_dict("records")) if existing is not None else []
    total_cost = sum(r["cost_usd"] for r in rows) if rows else 0.0

    for _, prod in products.iterrows():
        code = str(prod["code"])
        if code in done_codes:
            continue
        if context_mode == "partner_input":
            product_dict = {k: prod.get(k, "") for k in _PARTNER_FIELDS}
        else:
            try:
                product_dict = _load_off_grounded_fields(code, off_cache_dir)
            except FileNotFoundError as exc:
                logger.warning("Skipping %s: %s", code, exc)
                continue
        prompt = build_prompt(product_dict, schema)
        messages = [{"role": "user", "content": prompt}]
        t0 = time.perf_counter()
        try:
            result = call_fn(messages=messages, model=model, api_key=api_key,
                             enforce_json=True, max_tokens=4096)
        except Exception as exc:
            logger.warning("code=%s model=%s failed: %s", code, model, exc)
            continue
        latency_ms = result.get("latency_ms")
        if latency_ms is None:
            latency_ms = (time.perf_counter() - t0) * 1000.0
        usage = result.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens") or max(1, len(prompt) // 4))
        raw = result.get("raw") or ""
        out_tok = int(usage.get("completion_tokens") or max(1, len(raw) // 4))
        cost = (in_tok / 1_000_000) * pricing["in"] + (out_tok / 1_000_000) * pricing["out"]
        total_cost += cost

        try:
            parsed = parse_llm_response(raw, schema) if raw else {}
        except Exception:
            parsed = {}

        rows.append({
            "code": code, "model": model, "domain": domain,
            "context_mode": context_mode,
            "latency_ms": latency_ms, "in_tokens": in_tok, "out_tokens": out_tok,
            "cost_usd": cost, "raw": raw, "parsed_json": json.dumps(parsed),
        })

        if len(rows) % 20 == 0:
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            logger.info("[%s/%s/%s] %d rows, est_cost=$%.3f",
                        domain, model, context_mode, len(rows), total_cost)

        if total_cost >= max_cost_usd:
            logger.warning("Cost cap hit ($%.3f) — stopping", total_cost)
            break
        if sleep_between > 0:
            time.sleep(sleep_between)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    return df


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--gold-codes", required=True, type=Path,
                   help="CSV with 'code' column (e.g. pasta_gold_250.csv)")
    p.add_argument("--domain", required=True, choices=list(_SCHEMA))
    p.add_argument("--model", required=True)
    p.add_argument("--context-mode", choices=["partner_input", "off_grounded"],
                   default="partner_input")
    p.add_argument("--off-cache-dir", type=Path, default=None,
                   help="Required if context-mode=off_grounded")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-cost-usd", type=float, default=12.0)
    p.add_argument("--sleep", type=float, default=0.0)
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    # Read codes from gold CSV
    with args.gold_codes.open(newline="", encoding="utf-8") as f:
        codes = [str(r["code"]).strip() for r in csv.DictReader(f) if r.get("code")]
    codes = list(dict.fromkeys(codes))  # dedupe preserving order

    # Load product fields from silver standard for partner_input mode
    silver = pd.read_parquet(Path(PROCESSED_DIR) / f"{args.domain}_stratified_silver_standard.parquet")
    silver["code"] = silver["code"].astype(str)
    products = silver[silver["code"].isin(codes)].copy()

    # For off_grounded mode: codes not in silver can be backed by OFF cache JSON
    if args.context_mode == "off_grounded" and args.off_cache_dir:
        missing = set(codes) - set(products["code"])
        stub_rows = []
        for code in missing:
            cache_path = Path(args.off_cache_dir) / f"{code}.json"
            if not cache_path.exists():
                continue
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            prod = data.get("product", {})
            stub_rows.append({
                "code": code,
                "product_name": prod.get("product_name", "") or "",
                "brands": prod.get("brands", "") or "",
                "ingredients_text": prod.get("ingredients_text", "") or "",
                "quantity": prod.get("quantity", "") or "",
            })
        if stub_rows:
            stub_df = pd.DataFrame(stub_rows)
            # Add any missing columns from silver schema
            for col in products.columns:
                if col not in stub_df.columns:
                    stub_df[col] = ""
            products = pd.concat([products, stub_df[products.columns]], ignore_index=True)

    if len(products) == 0:
        raise SystemExit(f"No matching codes in silver or OFF cache for domain {args.domain}")

    df = run_llm_on_products(
        products, domain=args.domain, model=args.model, api_key=api_key,
        out_path=args.out, context_mode=args.context_mode,
        off_cache_dir=args.off_cache_dir,
        max_cost_usd=args.max_cost_usd, sleep_between=args.sleep,
    )
    summary = {
        "n_rows": len(df),
        "total_cost_usd": float(df["cost_usd"].sum()) if len(df) else 0.0,
        "p50_latency_ms": float(df["latency_ms"].quantile(0.5)) if len(df) else None,
        "p95_latency_ms": float(df["latency_ms"].quantile(0.95)) if len(df) else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
