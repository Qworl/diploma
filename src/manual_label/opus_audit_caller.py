"""Programmatic Opus audit caller for Trek E gold sets.

For each row in a domain gold CSV that hasn't been audited yet (all cells
with mode=""), this script:
  1. Builds a structured prompt using the existing pipeline.llm_fallback
     scaffolding (schema + few-shot examples).
  2. Calls OpenRouter (default: anthropic/claude-opus-4) with response_format
     enforced to JSON.
  3. Parses + validates the response against the schema.
  4. Writes a decision in the format consumed by `llm_audit.apply_llm_decisions`:
        {
          "<code>": {
            "<attr>": {"value": "...", "status_hint": null, "reasoning": null},
            ...
          },
          ...
        }
  5. Writes incremental output to a JSON file after every batch — re-running
     resumes from where it stopped.

Cost is tracked approximately by counting characters in/out (4 chars ≈ 1 token);
a hard cap aborts cleanly if the running estimate exceeds `--max-cost-usd`.

Run::

    OPENROUTER_API_KEY=sk-or-... \\
    python -m src.manual_label.opus_audit_caller \\
        --csv datasets/manual_label/chocolate_gold_239.csv \\
        --domain chocolate \\
        --model anthropic/claude-opus-4 \\
        --max-cost-usd 15.0 \\
        --out datasets/manual_label/opus_batches/chocolate_decisions.json
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
from src.manual_label.schemas_loader import load_domain_attrs
from src.pipeline.llm_fallback.prompts import build_prompt
from src.pipeline.schemas import EXAMPLES_BY_SCHEMA  # noqa: F401  (forces id-mapping import)
from src.pipeline.schemas.beverages import BEVERAGE_SCHEMA
from src.pipeline.schemas.cereals import CEREALS_SCHEMA
from src.pipeline.schemas.cheeses import CHEESES_SCHEMA
from src.pipeline.schemas.chocolate import CHOCOLATE_SCHEMA
from src.pipeline.schemas.cosmetics import COSMETICS_SCHEMA
from src.pipeline.schemas.pasta import PASTA_SCHEMA

logger = logging.getLogger(__name__)

# Maps a domain key to its raw schema dict (the one keyed in EXAMPLES_BY_SCHEMA).
_DOMAIN_SCHEMA_OBJ: dict[str, dict] = {
    "pasta": PASTA_SCHEMA,
    "chocolate": CHOCOLATE_SCHEMA,
    "cheeses": CHEESES_SCHEMA,
    "beverages": BEVERAGE_SCHEMA,
    "cereals": CEREALS_SCHEMA,
    "cosmetics": COSMETICS_SCHEMA,
}

# Approx Opus 4 pricing as of 2026-Q2 (input $15/M, output $75/M).
# Override via env if pricing changes.
_PRICE_INPUT_PER_MTOK = float(os.environ.get("OPUS_PRICE_INPUT_PER_MTOK", "15.0"))
_PRICE_OUTPUT_PER_MTOK = float(os.environ.get("OPUS_PRICE_OUTPUT_PER_MTOK", "75.0"))


def _approx_tokens(text: str) -> int:
    """Rough token estimate (1 token ≈ 4 chars). Conservative."""
    return max(1, len(text) // 4)


def _coerce_for_csv(value: Any, spec: dict) -> str | None:
    """Convert a parsed/validated LLM value into the string form used in the
    gold CSV. Bools → "True"/"False", enums kept as-is, nulls → None (skipped).
    """
    if value is None:
        return None
    if spec["type"] == "bool":
        return "True" if value else "False"
    return str(value)


def audit_product(
    product: dict,
    *,
    domain: str,
    api_key: str,
    model: str = "anthropic/claude-opus-4",
    call_fn: Callable = call_openrouter,
) -> tuple[dict, dict]:
    """Audit a single product. Returns (decision_dict, usage_dict).

    decision_dict shape: {attr: {"value": str|null, "status_hint": None, "reasoning": None}}
    usage_dict shape: {"in_tokens": int, "out_tokens": int}

    call_fn must return either a dict with key "raw" (new call_openrouter contract)
    or a plain str (legacy / test fakes). Both are handled transparently.
    """
    if domain not in _DOMAIN_SCHEMA_OBJ:
        raise KeyError(f"Unknown domain: {domain}")
    schema = _DOMAIN_SCHEMA_OBJ[domain]
    domain_attrs = load_domain_attrs(domain)

    prompt = build_prompt(product, schema)
    messages = [{"role": "user", "content": prompt}]

    result = call_fn(
        messages=messages,
        model=model,
        api_key=api_key,
        enforce_json=True,
        max_tokens=1024,
    )
    raw = result["raw"] if isinstance(result, dict) else result
    parsed = parse_llm_response(raw, schema)

    decision: dict[str, dict] = {}
    for attr, spec in schema.items():
        if attr not in parsed:
            decision[attr] = {"value": None, "status_hint": None, "reasoning": None}
            continue
        coerced = _coerce_for_csv(parsed[attr], spec)
        # Validate enum membership against normalised attrs (bool values become "True"/"False").
        allowed = domain_attrs[attr].get("values")
        if allowed is not None and coerced is not None and coerced not in allowed:
            decision[attr] = {"value": None, "status_hint": None,
                              "reasoning": f"invalid value {coerced!r}"}
            continue
        decision[attr] = {"value": coerced, "status_hint": None, "reasoning": None}

    usage = {
        "in_tokens": _approx_tokens(prompt),
        "out_tokens": _approx_tokens(raw),
    }
    return decision, usage


def _load_existing_decisions(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cannot read existing decisions file %s: %s — starting fresh", path, exc)
        return {}


def _write_decisions_atomic(path: Path, decisions: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _row_already_audited(row: dict, attrs: list[str]) -> bool:
    """True if at least one cell has mode != "" (audited or human-touched)."""
    for a in attrs:
        if (row.get(f"manual_{a}_mode") or "").strip():
            return True
    return False


def _row_to_product(row: dict) -> dict:
    """Extract input fields from a CSV row for prompt building."""
    return {
        "product_name": row.get("product_name", ""),
        "brands": row.get("brands", ""),
        "categories_tags": row.get("categories_tags", ""),
        "ingredients_text": row.get("ingredients_text", ""),
        "quantity": row.get("quantity", ""),
    }


def run_audit(
    csv_path: Path,
    *,
    domain: str,
    out_path: Path,
    api_key: str,
    model: str = "anthropic/claude-opus-4",
    max_cost_usd: float = 15.0,
    limit: int | None = None,
    progress_every: int = 10,
    sleep_between: float = 0.0,
    call_fn: Callable = call_openrouter,
) -> dict:
    """Run Opus audit on a domain gold CSV. Resume-safe + cost-capped.

    Returns a summary dict with cost + row counts.
    """
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    domain_attrs = list(load_domain_attrs(domain))
    decisions = _load_existing_decisions(out_path)

    total_in_tok = 0
    total_out_tok = 0
    n_called = 0
    n_skipped_already_audited = 0
    n_failed = 0
    started = time.monotonic()

    for idx, row in enumerate(rows):
        if limit is not None and n_called >= limit:
            break
        code = (row.get("code") or "").strip()
        if not code:
            continue
        if code in decisions:
            n_skipped_already_audited += 1
            continue
        if _row_already_audited(row, domain_attrs):
            n_skipped_already_audited += 1
            continue

        product = _row_to_product(row)
        try:
            decision, usage = audit_product(
                product, domain=domain, api_key=api_key, model=model, call_fn=call_fn,
            )
        except Exception as exc:
            logger.warning("Row %s (%s): audit failed: %s", idx, code, exc)
            n_failed += 1
            continue

        decisions[code] = decision
        total_in_tok += usage["in_tokens"]
        total_out_tok += usage["out_tokens"]
        n_called += 1

        if n_called % progress_every == 0 or n_called == 1:
            cost = (total_in_tok / 1_000_000) * _PRICE_INPUT_PER_MTOK + \
                   (total_out_tok / 1_000_000) * _PRICE_OUTPUT_PER_MTOK
            elapsed = time.monotonic() - started
            rate = n_called / max(elapsed, 1e-6)
            logger.info(
                "row %d/%d code=%s — called=%d failed=%d skipped=%d est_cost=$%.3f rate=%.2f rows/s",
                idx + 1, len(rows), code, n_called, n_failed,
                n_skipped_already_audited, cost, rate,
            )
            _write_decisions_atomic(out_path, decisions)

        cost_now = (total_in_tok / 1_000_000) * _PRICE_INPUT_PER_MTOK + \
                   (total_out_tok / 1_000_000) * _PRICE_OUTPUT_PER_MTOK
        if cost_now >= max_cost_usd:
            logger.warning(
                "Cost cap hit: estimated $%.3f >= $%.2f — stopping after %d rows",
                cost_now, max_cost_usd, n_called,
            )
            break

        if sleep_between > 0:
            time.sleep(sleep_between)

    _write_decisions_atomic(out_path, decisions)
    elapsed = time.monotonic() - started
    final_cost = (total_in_tok / 1_000_000) * _PRICE_INPUT_PER_MTOK + \
                 (total_out_tok / 1_000_000) * _PRICE_OUTPUT_PER_MTOK
    return {
        "rows_in_csv": len(rows),
        "rows_called": n_called,
        "rows_skipped_already_audited": n_skipped_already_audited,
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
    p.add_argument("--model", default="anthropic/claude-opus-4")
    p.add_argument("--max-cost-usd", type=float, default=15.0)
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N rows (for dry-run / smoke tests)")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="Seconds to sleep between requests (rate-limiting)")
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    result = run_audit(
        args.csv,
        domain=args.domain,
        out_path=args.out,
        api_key=api_key,
        model=args.model,
        max_cost_usd=args.max_cost_usd,
        limit=args.limit,
        sleep_between=args.sleep,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
