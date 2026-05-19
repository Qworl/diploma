"""Cold-start simulation: partner sends only text, cascade fills attributes from scratch.

Trek A2 measured "recovery on partner-typical masking" (~14% of cells masked).
This script measures the ABSOLUTE enrichment from a true cold-start: ALL
attribute cells masked → cascade runs on (product_name + brand + ingredients_text)
alone → how many attributes does it fill, and with what accuracy?

This is the headline number for the §1 actualnost claim
"система автоматизированного обогащения товарных данных" — what fraction
of attributes does the system deliver when the partner sends minimal data?

Pasta is the default; chocolate/cheeses are supported for E5 cross-domain
replication (Phase 1 thesis-defense fix).

Run (no LLM, pasta):
    python -m src.eval.catalog_completion.cold_start_simulation --no-llm

Run (chocolate / cheeses, no LLM):
    python -m src.eval.catalog_completion.cold_start_simulation \
        --category chocolate --no-llm
    python -m src.eval.catalog_completion.cold_start_simulation \
        --category cheeses --no-llm

Run (with LLM, ~15 min):
    python -m src.eval.catalog_completion.cold_start_simulation
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.catalog_completion.cascade_sim import run_cascade_on_masked, DEFAULT_LLM_MODEL
from src.eval.catalog_completion.metrics import aggregate_metrics, _norm, _values_equal
from src.manual_label.schemas_loader import load_domain_attrs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = "datasets/processed"

# Per-domain canonical attribute order (must match what was used to train the
# XGBoost models, since `category_model_prefix` is the only mapping to disk).
DOMAIN_ATTRS: dict[str, list[str]] = {
    "pasta": ["grain_type", "pasta_shape", "is_filled", "is_organic",
              "is_gluten_free", "is_vegan", "nutri_score_grade", "protein_class"],
    "chocolate": ["chocolate_type", "cocoa_percentage", "contains_nuts",
                  "chocolate_extra", "is_organic", "nutri_score_grade", "protein_class"],
    "cheeses": ["milk_source", "texture", "country_of_origin",
                "fat_class", "is_pdo", "is_organic", "is_ultra_processed"],
}

GOLD_CSV_MAP: dict[str, str] = {
    "pasta": "datasets/manual_label/pasta_gold_250.csv",
    "chocolate": "datasets/manual_label/chocolate_gold_239.csv",
    "cheeses": "datasets/manual_label/cheeses_gold_239.csv",
}

_SCHEMA_MODULES: dict[str, tuple[str, str]] = {
    "pasta": ("src.pipeline.schemas.pasta", "PASTA_SCHEMA"),
    "chocolate": ("src.pipeline.schemas.chocolate", "CHOCOLATE_SCHEMA"),
    "cheeses": ("src.pipeline.schemas.cheeses", "CHEESES_SCHEMA"),
}


def _load_llm_schema(category: str) -> dict:
    """Load the LLM-prompt schema dict for a given domain."""
    mod_path, name = _SCHEMA_MODULES[category]
    mod = importlib.import_module(mod_path)
    return getattr(mod, name)


def _mask_all(silver_sub: pd.DataFrame, attrs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mask EVERY attribute cell. Returns (masked_df, mask_log)."""
    masked = silver_sub.copy()
    mask_log_rows = []
    for _, row in silver_sub.iterrows():
        code = str(row["code"])
        for a in attrs:
            original = row[a] if a in silver_sub.columns else None
            mask_log_rows.append({
                "code": code,
                "attr": a,
                "masked": True,
                "original_value": None if pd.isna(original) else original,
            })
    for a in attrs:
        if a in masked.columns:
            masked[a] = None
    return masked, pd.DataFrame(mask_log_rows)


def _load_gold_audited(gold_csv: Path, attrs: list[str], llm_schema: dict) -> dict:
    """Return {code: {attr: audited_value}} for cells with status in audited set."""
    AUDITED = {"confirmed", "override", "manual_only"}
    out: dict[str, dict] = {}
    with gold_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code = r["code"]
            for a in attrs:
                status = (r.get(f"manual_{a}_status") or "").strip()
                if status not in AUDITED:
                    continue
                v = (r.get(f"manual_{a}") or "").strip()
                if v == "":
                    continue
                # Coerce booleans per schema
                spec = llm_schema.get(a, {})
                if spec.get("type") == "bool":
                    v = True if v.lower() in ("true", "1") else False
                out.setdefault(code, {})[a] = v
    return out


def _compute_metrics(silver_sub: pd.DataFrame, cascade_log: pd.DataFrame,
                     gold_audited: dict, attrs: list[str]) -> dict:
    """Compute cold-start metrics.

    - **fill_rate**: % of cells cascade filled (the headline enrichment number).
    - **accuracy_on_silver**: of filled cells, % that match silver_standard
      (the production-realistic accuracy on a representative product set).
    - **accuracy_on_gold**: of filled cells, % that match audited gold
      (the methodologically-cleanest accuracy — only on the audited cells).
    - **per_layer_breakdown**: which layer (regex/ml/bayes/llm) filled each cell.
    """
    n_total = len(silver_sub) * len(attrs)
    n_filled = 0
    n_silver_correct = 0
    n_silver_compared = 0
    n_gold_correct = 0
    n_gold_compared = 0
    layer_counts: dict[str, int] = {"regex": 0, "ml": 0, "bayes": 0, "llm": 0, "none": 0}
    per_attr: dict[str, dict] = {a: {"n_filled": 0, "n_total": 0,
                                     "n_silver_correct": 0, "n_silver_compared": 0,
                                     "n_gold_correct": 0, "n_gold_compared": 0}
                                 for a in attrs}
    silver_by_code = silver_sub.set_index(silver_sub["code"].astype(str))

    for _, log_row in cascade_log.iterrows():
        code = str(log_row["code"])
        attr = log_row["attr"]
        pred = log_row["cascade_pred"]
        layer = log_row["cascade_layer"]
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        per_attr[attr]["n_total"] += 1
        if _norm(pred) is not None:
            n_filled += 1
            per_attr[attr]["n_filled"] += 1
            # Compare to silver
            silver_row = silver_by_code.loc[code] if code in silver_by_code.index else None
            silver_v = silver_row[attr] if silver_row is not None and attr in silver_by_code.columns else None
            if _norm(silver_v) is not None:
                n_silver_compared += 1
                per_attr[attr]["n_silver_compared"] += 1
                if _values_equal(pred, silver_v):
                    n_silver_correct += 1
                    per_attr[attr]["n_silver_correct"] += 1
            # Compare to gold (audited)
            if code in gold_audited and attr in gold_audited[code]:
                n_gold_compared += 1
                per_attr[attr]["n_gold_compared"] += 1
                if _values_equal(pred, gold_audited[code][attr]):
                    n_gold_correct += 1
                    per_attr[attr]["n_gold_correct"] += 1

    return {
        "n_products": len(silver_sub),
        "n_attrs": len(attrs),
        "n_total_cells": n_total,
        "n_filled": n_filled,
        "fill_rate": n_filled / n_total if n_total else float("nan"),
        "accuracy_on_silver": n_silver_correct / n_silver_compared if n_silver_compared else None,
        "n_silver_compared": n_silver_compared,
        "accuracy_on_gold": n_gold_correct / n_gold_compared if n_gold_compared else None,
        "n_gold_compared": n_gold_compared,
        "per_layer": layer_counts,
        "per_attr": per_attr,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="pasta",
                   choices=["pasta", "chocolate", "cheeses"],
                   help="Domain to run cold-start on; requires audited gold CSV.")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--out", default=None,
                   help="Output JSON path; defaults to "
                        "datasets/processed/cold_start_simulation_{category}.json")
    args = p.parse_args()

    cat = args.category
    attrs = DOMAIN_ATTRS[cat]
    llm_schema = _load_llm_schema(cat)
    if args.out is None:
        args.out = os.path.join(PROCESSED_DIR, f"cold_start_simulation_{cat}.json")
    gold_path = Path(GOLD_CSV_MAP[cat])

    silver_path = os.path.join(PROCESSED_DIR, f"{cat}_stratified_silver_standard.parquet")
    emb_path = os.path.join(PROCESSED_DIR, f"{cat}_stratified_embeddings.npy")

    silver = pd.read_parquet(silver_path).reset_index(drop=True)
    silver["code"] = silver["code"].astype(str)
    emb = np.load(emb_path)

    with gold_path.open() as f:
        gold_codes = {r["code"] for r in csv.DictReader(f)}
    mask = silver["code"].isin(gold_codes).values
    sub = silver.loc[mask].reset_index(drop=True)
    sub_emb = emb[mask]
    logger.info("Cold-start simulation on %d %s products (audited subset)", len(sub), cat)

    masked_sub, mask_log = _mask_all(sub, attrs)
    gold_audited = _load_gold_audited(gold_path, attrs, llm_schema)

    cascade_log = run_cascade_on_masked(
        masked_sub, sub_emb, attrs, f"{cat}_stratified",
        regex_category=cat,
        enable_llm=not args.no_llm,
        llm_schema=llm_schema,
        llm_model=args.llm_model,
    )

    config_tag = "no_llm" if args.no_llm else "with_llm"
    m = _compute_metrics(sub, cascade_log, gold_audited, attrs)
    m["config"] = config_tag
    m["category"] = cat

    print()
    print(f"=== Cold-start simulation ({cat}, {config_tag}, n={m['n_products']}) ===")
    print(f"Total cells:         {m['n_total_cells']}")
    print(f"Cells filled:        {m['n_filled']} ({m['fill_rate']*100:.1f}%)")
    print(f"Accuracy vs silver:  {m['accuracy_on_silver']*100:.1f}% (n={m['n_silver_compared']})" if m['accuracy_on_silver'] is not None else "Accuracy vs silver:  n/a")
    print(f"Accuracy vs gold:    {m['accuracy_on_gold']*100:.1f}% (n={m['n_gold_compared']})" if m['accuracy_on_gold'] is not None else "Accuracy vs gold:    n/a")
    print()
    print("Per-layer breakdown:")
    for layer, n in m["per_layer"].items():
        print(f"  {layer:10s}  {n:>5d}  ({n/m['n_total_cells']*100:.1f}%)")
    print()
    print("Per-attribute fill rate + accuracy_on_gold:")
    print(f"  {'attr':22s} {'fill':>10s} {'silver_acc':>12s} {'gold_acc':>12s}")
    for attr, st in m["per_attr"].items():
        fill = f"{st['n_filled']}/{st['n_total']} ({st['n_filled']/st['n_total']*100:.0f}%)"
        sa = f"{st['n_silver_correct']/st['n_silver_compared']*100:.0f}% (n={st['n_silver_compared']})" if st['n_silver_compared'] else "n/a"
        ga = f"{st['n_gold_correct']/st['n_gold_compared']*100:.0f}% (n={st['n_gold_compared']})" if st['n_gold_compared'] else "n/a"
        print(f"  {attr:22s} {fill:>10s} {sa:>12s} {ga:>12s}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(m, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
