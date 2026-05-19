"""Trek A2 — pasta runner (audited gold).

Usage:
    OMP_NUM_THREADS=1 python -m src.eval.catalog_completion.run_pasta \\
        [--seed 42] [--no-llm] [--gold-only]
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.eval.catalog_completion.cascade_sim import (
    DEFAULT_LLM_MODEL,
    _llm_cost_usd,
    run_cascade_on_masked,
)
from src.eval.catalog_completion.masking import mask_dataframe
from src.eval.catalog_completion.metrics import aggregate_metrics
from src.eval.catalog_completion.missingness import (
    compute_missingness_profile,
    save_profile,
)
from src.eval.cascade_vs_audited_gold import PASTA_ATTRS
from src.pipeline.schemas import PASTA_SCHEMA

logger = logging.getLogger(__name__)

AUDITED_STATUSES = {"confirmed", "override", "manual_only"}


def _load_gold_long(gold_path: str) -> pd.DataFrame:
    """audited gold CSV → long-format (code, attr, manual_value, status)."""
    g = pd.read_csv(gold_path, dtype={"code": str})
    rows = []
    for _, r in g.iterrows():
        code = str(r["code"])
        for a in PASTA_ATTRS:
            rows.append({
                "code": code,
                "attr": a,
                "manual_value": r.get(f"manual_{a}"),
                "manual_status": r.get(f"manual_{a}_status"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gold", default="datasets/manual_label/pasta_gold_250.csv")
    p.add_argument("--no-llm", action="store_true",
                   help="Disable Layer 4 LLM (static-policy baseline run)")
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--out-prefix", default=os.path.join(PROCESSED_DIR, "catalog_completion"))
    args = p.parse_args()

    # 1. Load silver + embeddings, subset to gold codes
    silver = pd.read_parquet(os.path.join(PROCESSED_DIR, "pasta_stratified_silver_standard.parquet")).reset_index(drop=True)
    silver["code"] = silver["code"].astype(str)
    emb = np.load(os.path.join(PROCESSED_DIR, "pasta_stratified_embeddings.npy"))
    gold_long = _load_gold_long(args.gold)
    gold_codes = set(gold_long["code"].unique())
    mask = silver["code"].isin(gold_codes).values
    sub = silver.loc[mask].reset_index(drop=True)
    sub_emb = emb[mask]
    logger.info("Pasta gold-overlap rows: %d", len(sub))

    # 2. Build missingness profile from FULL silver (not the subset) — represents partner-typical
    profile = compute_missingness_profile(silver, target_attrs=PASTA_ATTRS)
    save_profile(profile, f"{args.out_prefix}_missingness_pasta.json")

    # 3. Mask the gold subset
    masked_sub, mask_log = mask_dataframe(sub, profile, global_seed=args.seed)

    # 4. Run cascade
    cascade_log = run_cascade_on_masked(
        masked_sub, sub_emb, PASTA_ATTRS, "pasta_stratified",
        regex_category="pasta",
        enable_llm=not args.no_llm,
        llm_schema=PASTA_SCHEMA,
        llm_model=args.llm_model,
    )

    # 5. Replace mask_log's `original_value` with audited manual_value where available
    audited_lookup = gold_long.set_index(["code", "attr"]).to_dict("index")
    def _override(row):
        rec = audited_lookup.get((row["code"], row["attr"]))
        if rec and rec.get("manual_status") in AUDITED_STATUSES:
            return rec["manual_value"]
        return row["original_value"]
    mask_log["original_value"] = mask_log.apply(_override, axis=1)

    # 6. Save logs + summary
    config_tag = "no_llm" if args.no_llm else "with_llm"
    out_log = f"{args.out_prefix}_log_pasta_{config_tag}.parquet"
    combined = mask_log.merge(cascade_log, on=["code", "attr"], how="left")
    # Ensure original_value is str-typed for parquet compatibility (may contain bools)
    combined["original_value"] = combined["original_value"].apply(
        lambda v: None if (v is None or (isinstance(v, float) and __import__("math").isnan(v))) else str(v)
    )
    combined.to_parquet(out_log, index=False)

    m = aggregate_metrics(mask_log, cascade_log, n_products=len(sub), alpha=0.05)
    m["config"] = config_tag
    m["llm_cost_usd_per_call"] = _llm_cost_usd(args.llm_model) if not args.no_llm else 0.0
    m["llm_cost_usd_per_1000_products"] = m["llm_calls_per_1000_products"] * m["llm_cost_usd_per_call"]
    out_sum = f"{args.out_prefix}_summary_pasta_{config_tag}.json"
    with open(out_sum, "w") as f:
        json.dump(m, f, indent=2, default=str)
    logger.info("Saved %s → %s", config_tag, out_sum)
    logger.info("coverage_gain_pp=%.2f  recovery_accuracy=%s  llm/1k=%.1f",
                m["coverage_gain_pp"], m["recovery_accuracy"], m["llm_calls_per_1000_products"])


if __name__ == "__main__":
    main()
