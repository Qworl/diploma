"""Validate the silver-extractor fix on the 239 pasta gold products.

**Caveat — not a clean A/B test.** "Old silver" stored in
`pasta_stratified_silver_extended.parquet` is the project's
*accumulated* silver standard — apply_off_labels plus regex fallback,
LLM enrichment, manual arbitrage. The "new silver" we compute here is
pure `apply_off_labels(row, PASTA_SCHEMA)` — no regex extractor, no
LLM, no manual overlays. So the diff conflates "fix improved
apply_off_labels" with "apply_off_labels alone has less coverage than
the full pipeline".

A clean A/B requires re-running the FULL `src.data.label_silver
--category pasta_stratified` pipeline against both code-base versions
(pre- and post-fix), then comparing the regenerated silver parquets.

What this script IS useful for:
- Spot-checking that the new apply_off_labels handles specific tags
  Opus identified (en:lentil-pasta, en:konjac, en:ravioli, etc.).
- Counting how many cells *changed* between the new apply_off_labels
  output and the accumulated old silver — useful for triage when
  about to re-run silver labelling.

Run:
    python -m src.eval.silver_fix_validation
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from src.manual_label.schemas_loader import load_pasta_attrs
from src.pipeline.off_labels.apply import apply_off_labels
from src.pipeline.schemas.pasta import PASTA_SCHEMA


def _new_silver_value(row: dict, attr: str) -> str:
    """Re-run the FULL silver pipeline (apply_off_labels) for one row.

    Returns the string-coerced silver value, or "" if the pipeline abstains.
    """
    out = apply_off_labels(row, PASTA_SCHEMA)
    v = out.get(attr)
    if v is None:
        return ""
    return str(v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", default="datasets/manual_label/pasta_gold_250.csv", type=Path)
    p.add_argument("--silver-extended",
                   default="datasets/processed/pasta_stratified_silver_extended.parquet",
                   type=Path)
    p.add_argument("--out", default="datasets/processed/silver_fix_validation_pasta.json", type=Path)
    args = p.parse_args()

    # Load gold (the audit truth)
    with args.gold.open(newline="", encoding="utf-8") as f:
        gold_rows = list(csv.DictReader(f))
    gold_by_code = {r["code"]: r for r in gold_rows}

    # Load raw OFF data for the same 239 products
    raw = pd.read_parquet(args.silver_extended)
    raw["code"] = raw["code"].astype(str)
    raw = raw[raw["code"].isin(gold_by_code.keys())]
    print(f"Audited products: {len(gold_by_code)}, matched in silver_extended: {len(raw)}")

    attrs = list(load_pasta_attrs())

    AUDITED_STATUSES = {"confirmed", "override", "manual_only"}
    per_attr: dict[str, dict] = {}
    for attr in attrs:
        n_audited = 0
        old_correct = new_correct = 0
        silver_changed = 0
        fixes_landed = 0
        regressions = 0
        for _, raw_row in raw.iterrows():
            code = str(raw_row["code"])
            gold = gold_by_code.get(code)
            if gold is None:
                continue
            status = (gold.get(f"manual_{attr}_status") or "").strip()
            if status not in AUDITED_STATUSES:
                continue
            mode = (gold.get(f"manual_{attr}_mode") or "").strip()
            if mode not in {"blind", "llm"}:
                continue
            n_audited += 1
            manual = (gold.get(f"manual_{attr}") or "").strip()
            old_silver = (gold.get(f"silver_{attr}") or "").strip()
            new_silver = _new_silver_value(raw_row.to_dict(), attr)

            if old_silver == manual:
                old_correct += 1
            if new_silver == manual:
                new_correct += 1
            if old_silver != new_silver:
                silver_changed += 1
                if new_silver == manual and old_silver != manual:
                    fixes_landed += 1
                elif new_silver != manual and old_silver == manual:
                    regressions += 1
        if n_audited == 0:
            continue
        per_attr[attr] = {
            "n_audited": n_audited,
            "old_silver_acc": old_correct / n_audited,
            "new_silver_acc": new_correct / n_audited,
            "silver_changed_cells": silver_changed,
            "fixes_landed": fixes_landed,
            "regressions": regressions,
            "net_improvement_cells": new_correct - old_correct,
        }

    total_old = sum(p["old_silver_acc"] * p["n_audited"] for p in per_attr.values())
    total_new = sum(p["new_silver_acc"] * p["n_audited"] for p in per_attr.values())
    total_n = sum(p["n_audited"] for p in per_attr.values())

    print()
    print(f"{'attribute':22s} {'n':>4s}  {'old_acc':>8s} {'new_acc':>8s} {'Δ_pp':>6s}  {'fixes':>5s} {'regr':>5s}")
    print("-" * 75)
    for attr, st in per_attr.items():
        delta_pp = (st["new_silver_acc"] - st["old_silver_acc"]) * 100
        print(f"{attr:22s} {st['n_audited']:>4d}  {st['old_silver_acc']*100:>7.1f}% {st['new_silver_acc']*100:>7.1f}% {delta_pp:>+5.1f}p  "
              f"{st['fixes_landed']:>5d} {st['regressions']:>5d}")
    print("-" * 75)
    print(f"{'TOTAL':22s} {total_n:>4d}  {total_old/total_n*100:>7.1f}% {total_new/total_n*100:>7.1f}% "
          f"{(total_new-total_old)/total_n*100:>+5.1f}p")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({
            "per_attr": per_attr,
            "total_n": total_n,
            "total_old_acc": total_old / total_n,
            "total_new_acc": total_new / total_n,
        }, f, indent=2)
    print(f"\nSaved JSON to {args.out}")


if __name__ == "__main__":
    main()
