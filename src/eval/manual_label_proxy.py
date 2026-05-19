"""Run a proxy LLM (default Qwen 2.5 72B) on the pasta gold-annotation seed.

The proxy is intentionally outside the consensus_gold model family
(Sonnet 4.5, GPT-4o, Gemini 2.5 Flash) so that human-vs-proxy agreement
provides an independent cross-check.

Input  : datasets/manual_label/pasta_gold_250.csv (partner fields + manual_* columns)
Output : <out>.csv with `code` + `proxy_<attr>` columns for each pasta attribute.

Usage:
    OMP_NUM_THREADS=1 .venv/bin/python -m src.eval.manual_label_proxy \\
        --in datasets/manual_label/pasta_gold_250.csv \\
        --out datasets/manual_label/pasta_gold_250_proxy.csv \\
        --model qwen/qwen-2.5-72b-instruct
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

from src.manual_label.schemas_loader import load_pasta_attrs
from src.pipeline.llm_fallback import enrich_product
from src.pipeline.schemas.pasta import PASTA_SCHEMA


INPUT_FIELDS = ["product_name", "brands", "ingredients_text", "quantity"]


def _row_to_partner_fields(row: pd.Series) -> dict:
    """Extract the partner-available fields (the only inputs the cascade sees)."""
    product = {f: (row.get(f, "") or "") for f in INPUT_FIELDS}
    product["code"] = str(row.get("code", ""))
    return product


def _to_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    s = str(v)
    if s.lower() in ("nan", "none"):
        return ""
    return s


def run(in_path: Path, out_path: Path, model: str, limit: int | None) -> None:
    df = pd.read_csv(in_path, dtype=str).fillna("")
    if limit:
        df = df.head(limit)
    attrs = list(load_pasta_attrs().keys())

    out_rows: list[dict] = []
    for i, row in df.iterrows():
        product = _row_to_partner_fields(row)
        try:
            parsed = enrich_product(
                product, PASTA_SCHEMA,
                backend="openrouter", model=model,
                enforce_json=True,
            )
        except Exception as exc:
            print(f"  [{i + 1}/{len(df)}] {product['code']} — LLM error: {exc}")
            parsed = {}

        if not isinstance(parsed, dict):
            parsed = {}

        record = {"code": str(row["code"])}
        for attr in attrs:
            record[f"proxy_{attr}"] = _to_str(parsed.get(attr))
        out_rows.append(record)

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(df)} — checkpoint")
            pd.DataFrame(out_rows).to_csv(out_path, index=False)

    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"wrote {len(out_rows)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="in_path", type=Path, required=True,
                        help="Path to pasta_gold_250.csv")
    parser.add_argument("--out", type=Path, required=True,
                        help="Path for proxy predictions CSV")
    parser.add_argument("--model", default="qwen/qwen-2.5-72b-instruct",
                        help="OpenRouter model slug (default: qwen/qwen-2.5-72b-instruct)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N rows (smoke testing).")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run(args.in_path, args.out, args.model, args.limit)


if __name__ == "__main__":
    main()
