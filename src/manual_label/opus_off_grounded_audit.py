"""Blind Opus audit using OFF-grounded context (no silver, no prefill).

Difference from src.manual_label.opus_audit_caller (deprecated):
  - Pulls fresh product data from OFF public API (cached) instead of relying
    on partner-supplied CSV row.
  - Excludes OFF-derived target classifications (see off_field_filter.py).
  - Does NOT receive any silver_values or current_state from our pipeline.

Resume-safe: re-running with the same `--out` JSON skips codes already present.

Run::

    OPENROUTER_API_KEY=sk-or-... \\
    .venv/bin/python -m src.manual_label.opus_off_grounded_audit \\
        --csv datasets/manual_label/pasta_gold_239.csv \\
        --domain pasta \\
        --out datasets/manual_label/opus_batches/blind_v2/pasta_decisions.json \\
        --cache-dir datasets/manual_label/off_cache \\
        --max-cost-usd 50.0
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from src.llm.client import call_openrouter
from src.llm.parsing import parse_llm_response
from src.manual_label.off_fetcher import OFFFetchError, fetch_off_product
from src.manual_label.off_field_filter import curate_prompt_fields
from src.manual_label.schemas_loader import load_domain_attrs
from src.pipeline.llm_fallback.prompts import build_prompt
from src.pipeline.schemas import EXAMPLES_BY_SCHEMA  # noqa: F401
from src.pipeline.schemas.beverages import BEVERAGE_SCHEMA
from src.pipeline.schemas.cereals import CEREALS_SCHEMA
from src.pipeline.schemas.cheeses import CHEESES_SCHEMA
from src.pipeline.schemas.chocolate import CHOCOLATE_SCHEMA
from src.pipeline.schemas.cosmetics import COSMETICS_SCHEMA
from src.pipeline.schemas.pasta import PASTA_SCHEMA

logger = logging.getLogger(__name__)

_DOMAIN_SCHEMA_OBJ: dict[str, dict] = {
    "pasta": PASTA_SCHEMA,
    "chocolate": CHOCOLATE_SCHEMA,
    "cheeses": CHEESES_SCHEMA,
    "beverages": BEVERAGE_SCHEMA,
    "cereals": CEREALS_SCHEMA,
    "cosmetics": COSMETICS_SCHEMA,
}

_PRICE_INPUT_PER_MTOK = float(os.environ.get("OPUS_PRICE_INPUT_PER_MTOK", "15.0"))
_PRICE_OUTPUT_PER_MTOK = float(os.environ.get("OPUS_PRICE_OUTPUT_PER_MTOK", "75.0"))


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _coerce_for_csv(value: Any, spec: dict) -> str | None:
    if value is None:
        return None
    if spec["type"] == "bool":
        return "True" if value else "False"
    return str(value)


def audit_product(
    code: str,
    *,
    domain: str,
    cache_dir: Path,
    api_key: str,
    model: str = "anthropic/claude-opus-4",
    fetch_fn: Callable = fetch_off_product,
    call_fn: Callable = call_openrouter,
) -> tuple[dict, dict]:
    """Audit a single product using OFF-grounded context.

    Returns (decision_dict, usage_dict).
    decision_dict shape: {attr: {"value": str|None, "status_hint": None, "reasoning": None}}
    """
    if domain not in _DOMAIN_SCHEMA_OBJ:
        raise KeyError(f"Unknown domain: {domain}")
    schema = _DOMAIN_SCHEMA_OBJ[domain]
    domain_attrs = load_domain_attrs(domain)

    off_product = fetch_fn(code, cache_dir=Path(cache_dir))
    curated = curate_prompt_fields(off_product)

    prompt = build_prompt(curated, schema)
    messages = [{"role": "user", "content": prompt}]

    result = call_fn(
        messages=messages, model=model, api_key=api_key,
        enforce_json=True, max_tokens=1024,
    )
    raw = result["raw"] if isinstance(result, dict) else result
    parsed = parse_llm_response(raw, schema)

    decision: dict[str, dict] = {}
    for attr, spec in schema.items():
        if attr not in parsed or parsed[attr] is None:
            decision[attr] = {"value": None, "status_hint": None, "reasoning": None}
            continue
        coerced = _coerce_for_csv(parsed[attr], spec)
        allowed = domain_attrs[attr].get("values")
        if allowed is not None and coerced is not None and coerced not in allowed:
            decision[attr] = {"value": None, "status_hint": None,
                              "reasoning": f"invalid value {coerced!r}"}
            continue
        decision[attr] = {"value": coerced, "status_hint": None, "reasoning": None}

    usage = {"in_tokens": _approx_tokens(prompt), "out_tokens": _approx_tokens(raw)}
    return decision, usage


def _load_existing_decisions(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cannot read %s: %s — starting fresh", path, exc)
        return {}


def _write_decisions_atomic(path: Path, decisions: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def run_audit(
    csv_path: Path,
    *,
    domain: str,
    out_path: Path,
    cache_dir: Path,
    api_key: str,
    model: str = "anthropic/claude-opus-4",
    max_cost_usd: float = 50.0,
    limit: int | None = None,
    progress_every: int = 10,
    fetch_fn: Callable = fetch_off_product,
    call_fn: Callable = call_openrouter,
) -> dict:
    """Run blind Opus audit on a CSV of product codes. Resume-safe + cost-capped."""
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    cache_dir = Path(cache_dir)

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    decisions = _load_existing_decisions(out_path)

    n_called = 0
    n_skipped = 0
    n_failed = 0
    total_in_tok = 0
    total_out_tok = 0
    started = time.monotonic()

    for idx, row in enumerate(rows):
        if limit is not None and n_called >= limit:
            break
        code = (row.get("code") or "").strip()
        if not code or code in decisions:
            n_skipped += 1
            continue

        try:
            decision, usage = audit_product(
                code=code, domain=domain, cache_dir=cache_dir,
                api_key=api_key, model=model,
                fetch_fn=fetch_fn, call_fn=call_fn,
            )
        except OFFFetchError as exc:
            logger.warning("OFF fetch failed for %s: %s", code, exc)
            n_failed += 1
            continue
        except Exception as exc:
            logger.warning("Audit failed for %s: %s", code, exc)
            n_failed += 1
            continue

        decisions[code] = decision
        total_in_tok += usage["in_tokens"]
        total_out_tok += usage["out_tokens"]
        n_called += 1

        if n_called % progress_every == 0 or n_called == 1:
            cost = (total_in_tok / 1e6) * _PRICE_INPUT_PER_MTOK + \
                   (total_out_tok / 1e6) * _PRICE_OUTPUT_PER_MTOK
            elapsed = time.monotonic() - started
            rate = n_called / max(elapsed, 1e-6)
            logger.info(
                "%d/%d code=%s called=%d failed=%d skipped=%d est_cost=$%.3f rate=%.2f/s",
                idx + 1, len(rows), code, n_called, n_failed, n_skipped,
                cost, rate,
            )
            _write_decisions_atomic(out_path, decisions)

        cost_now = (total_in_tok / 1e6) * _PRICE_INPUT_PER_MTOK + \
                   (total_out_tok / 1e6) * _PRICE_OUTPUT_PER_MTOK
        if cost_now >= max_cost_usd:
            logger.warning(
                "Cost cap hit: $%.3f >= $%.2f after %d rows", cost_now,
                max_cost_usd, n_called,
            )
            break

    _write_decisions_atomic(out_path, decisions)
    elapsed = time.monotonic() - started
    final_cost = (total_in_tok / 1e6) * _PRICE_INPUT_PER_MTOK + \
                 (total_out_tok / 1e6) * _PRICE_OUTPUT_PER_MTOK
    return {
        "rows_in_csv": len(rows),
        "rows_called": n_called,
        "rows_skipped": n_skipped,
        "rows_failed": n_failed,
        "total_in_tokens": total_in_tok,
        "total_out_tokens": total_out_tok,
        "estimated_cost_usd": final_cost,
        "elapsed_sec": elapsed,
        "out_path": str(out_path),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--domain", required=True, choices=list(_DOMAIN_SCHEMA_OBJ))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--model", default="anthropic/claude-opus-4")
    p.add_argument("--max-cost-usd", type=float, default=50.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    result = run_audit(
        csv_path=args.csv, domain=args.domain, out_path=args.out,
        cache_dir=args.cache_dir, api_key=api_key, model=args.model,
        max_cost_usd=args.max_cost_usd, limit=args.limit,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
