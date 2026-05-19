"""Quick experiment: does enriched silver (audited gold overlaid) shift A2 coverage_gain?

Builds a "silver_enriched" parquet by replacing silver_<attr> with audited
manual_<attr> values for the 239 pasta_gold products. Runs the SAME A2
simulation against this enriched silver and prints coverage_gain delta.

No LLM calls. ~30 sec.

Run:
    python -m src.eval.catalog_completion.silver_enrichment_experiment
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.catalog_completion.cascade_sim import run_cascade_on_masked
from src.eval.catalog_completion.masking import mask_dataframe
from src.eval.catalog_completion.missingness import compute_missingness_profile
from src.eval.catalog_completion.metrics import aggregate_metrics
from src.pipeline.schemas.pasta import PASTA_SCHEMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PASTA_ATTRS = ["grain_type", "pasta_shape", "is_filled", "is_organic",
               "is_gluten_free", "is_vegan", "nutri_score_grade", "protein_class"]
PROCESSED_DIR = "datasets/processed"
GOLD_PATH = "datasets/manual_label/pasta_gold_250.csv"
AUDITED = {"confirmed", "override", "manual_only"}


def _coerce(v: str, attr: str):
    """Coerce gold CSV string to silver-native type for the given attr."""
    v = (v or "").strip()
    if v == "":
        return None
    spec = PASTA_SCHEMA.get(attr, {})
    if spec.get("type") == "bool":
        if v.lower() in ("true", "1", "yes"):
            return True
        if v.lower() in ("false", "0", "no"):
            return False
        return None
    return v


def build_enriched_silver(silver: pd.DataFrame, gold_csv: Path) -> tuple[pd.DataFrame, dict]:
    """Overlay audited gold values onto silver. Return (enriched_df, delta_stats)."""
    with open(gold_csv, newline="", encoding="utf-8") as f:
        gold_rows = list(csv.DictReader(f))
    gold_by_code = {r["code"]: r for r in gold_rows}
    enriched = silver.copy()
    enriched["code"] = enriched["code"].astype(str)
    delta: dict[str, dict] = {a: {"changed": 0, "filled": 0, "overridden": 0, "untouched_audited": 0}
                              for a in PASTA_ATTRS}
    for idx, row in enriched.iterrows():
        code = row["code"]
        if code not in gold_by_code:
            continue
        g = gold_by_code[code]
        for a in PASTA_ATTRS:
            status = (g.get(f"manual_{a}_status") or "").strip()
            mode = (g.get(f"manual_{a}_mode") or "").strip()
            if status not in AUDITED or mode not in ("blind", "llm"):
                continue
            new_v = _coerce(g.get(f"manual_{a}", ""), a)
            old_v = row[a] if a in enriched.columns else None
            if pd.isna(old_v) and new_v is not None:
                delta[a]["filled"] += 1
                delta[a]["changed"] += 1
                enriched.at[idx, a] = new_v
            elif not pd.isna(old_v) and new_v is not None and old_v != new_v:
                delta[a]["overridden"] += 1
                delta[a]["changed"] += 1
                enriched.at[idx, a] = new_v
            else:
                delta[a]["untouched_audited"] += 1
    return enriched, delta


def run_simulation(silver: pd.DataFrame, emb: np.ndarray, gold_codes: set[str], *, seed: int = 42) -> dict:
    """Mask → cascade → score. Returns coverage_gain_pp + recovery_acc."""
    mask = silver["code"].isin(gold_codes).values
    sub = silver.loc[mask].reset_index(drop=True)
    sub_emb = emb[mask]
    profile = compute_missingness_profile(silver, target_attrs=PASTA_ATTRS)
    masked_sub, mask_log = mask_dataframe(sub, profile, global_seed=seed)
    cascade_log = run_cascade_on_masked(
        masked_sub, sub_emb, PASTA_ATTRS, "pasta_stratified",
        regex_category="pasta",
        enable_llm=False,  # no-LLM only for this experiment
    )
    m = aggregate_metrics(mask_log, cascade_log, n_products=len(sub))
    return {
        "n_products": len(sub),
        "n_total_cells": len(sub) * len(PASTA_ATTRS),
        "n_masked": int(mask_log["masked"].sum()),
        "coverage_gain_pp": m["coverage_gain_pp"],
        "recovery_acc": m.get("recovery_accuracy"),
        "n_recovery": m.get("n_recovery_universe"),
    }


def main() -> None:
    silver = pd.read_parquet(os.path.join(PROCESSED_DIR, "pasta_stratified_silver_standard.parquet")).reset_index(drop=True)
    silver["code"] = silver["code"].astype(str)
    emb = np.load(os.path.join(PROCESSED_DIR, "pasta_stratified_embeddings.npy"))

    with open(GOLD_PATH) as f:
        gold_codes = {r["code"] for r in csv.DictReader(f)}

    logger.info("Baseline (raw silver) — running simulation...")
    base = run_simulation(silver, emb, gold_codes)
    logger.info("baseline coverage_gain_pp=%.2f recovery_acc=%s n_masked=%d n_recovery=%s",
                base["coverage_gain_pp"], base["recovery_acc"], base["n_masked"], base["n_recovery"])

    logger.info("Building enriched silver (overlay audited gold)...")
    enriched, delta = build_enriched_silver(silver, Path(GOLD_PATH))
    total_changed = sum(d["changed"] for d in delta.values())
    total_filled = sum(d["filled"] for d in delta.values())
    total_overridden = sum(d["overridden"] for d in delta.values())
    logger.info("enrichment delta: %d cells changed (%d filled None→value, %d overridden value→value)",
                total_changed, total_filled, total_overridden)
    for a, d in delta.items():
        logger.info("  %22s  filled=%d  overridden=%d  untouched=%d", a, d["filled"], d["overridden"], d["untouched_audited"])

    logger.info("Enriched silver — running simulation...")
    enr = run_simulation(enriched, emb, gold_codes)
    logger.info("enriched coverage_gain_pp=%.2f recovery_acc=%s n_masked=%d n_recovery=%s",
                enr["coverage_gain_pp"], enr["recovery_acc"], enr["n_masked"], enr["n_recovery"])

    print()
    print("=== Silver enrichment experiment (pasta, no-LLM) ===")
    print(f"{'metric':24s} {'baseline':>12s} {'enriched':>12s} {'delta':>10s}")
    print("-" * 65)
    for key in ("coverage_gain_pp", "recovery_acc", "n_masked", "n_recovery"):
        bv, ev = base[key], enr[key]
        if isinstance(bv, (int, float)) and isinstance(ev, (int, float)):
            d = ev - bv
            d_str = f"{d:+.2f}" if isinstance(d, float) and not isinstance(d, bool) else f"{d:+d}"
        else:
            d_str = "n/a"
        bv_str = f"{bv:.4f}" if isinstance(bv, float) else str(bv)
        ev_str = f"{ev:.4f}" if isinstance(ev, float) else str(ev)
        print(f"{key:24s} {bv_str:>12s} {ev_str:>12s} {d_str:>10s}")
    print()
    out_path = f"{PROCESSED_DIR}/silver_enrichment_experiment_pasta.json"
    with open(out_path, "w") as f:
        json.dump({"baseline": base, "enriched": enr, "delta_per_attr": delta}, f, indent=2, default=str)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
