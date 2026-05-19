"""Trek A2 — electronics runner (smartphones).

Usage:
    OMP_NUM_THREADS=1 python -m src.eval.catalog_completion.run_electronics \\
        [--seed 42] [--no-llm] [--max-products 300]
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
from src.pipeline.schemas import ELECTRONICS_SCHEMA

logger = logging.getLogger(__name__)

# Drop `price_tier` — silver has 0 fill (see prepare.py) → masking is no-op.
ELECTRONICS_ATTRS = [
    "brand", "os", "form_factor",
    "screen_size_class", "ram_class", "storage_class",
    "release_year_class",
]


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--max-products", type=int, default=300)
    p.add_argument("--out-prefix", default=os.path.join(PROCESSED_DIR, "catalog_completion"))
    args = p.parse_args()

    silver_path = os.path.join(PROCESSED_DIR, "electronics_silver_standard.parquet")
    emb_path = os.path.join(PROCESSED_DIR, "electronics_embeddings.npy")
    silver = pd.read_parquet(silver_path).reset_index(drop=True)
    silver["code"] = silver["code"].astype(str)
    emb = np.load(emb_path)

    profile = compute_missingness_profile(silver, target_attrs=ELECTRONICS_ATTRS)
    save_profile(profile, f"{args.out_prefix}_missingness_electronics.json")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(silver))[: args.max_products]
    sub = silver.iloc[perm].reset_index(drop=True)
    sub_emb = emb[perm]

    masked_sub, mask_log = mask_dataframe(sub, profile, global_seed=args.seed)
    cascade_log = run_cascade_on_masked(
        masked_sub, sub_emb, ELECTRONICS_ATTRS, "electronics",
        regex_category="electronics",
        enable_llm=not args.no_llm,
        llm_schema=ELECTRONICS_SCHEMA,
        llm_model=args.llm_model,
    )

    config_tag = "no_llm" if args.no_llm else "with_llm"
    mask_log.merge(cascade_log, on=["code", "attr"], how="left").to_parquet(
        f"{args.out_prefix}_log_electronics_{config_tag}.parquet", index=False)

    m = aggregate_metrics(mask_log, cascade_log, n_products=len(sub), alpha=0.05)
    m["config"] = config_tag
    m["llm_cost_usd_per_call"] = _llm_cost_usd(args.llm_model) if not args.no_llm else 0.0
    m["llm_cost_usd_per_1000_products"] = m["llm_calls_per_1000_products"] * m["llm_cost_usd_per_call"]
    m["recovery_accuracy_note"] = "PhoneDB-derived silver = deterministic ground truth"
    with open(f"{args.out_prefix}_summary_electronics_{config_tag}.json", "w") as f:
        json.dump(m, f, indent=2, default=str)
    logger.info("coverage_gain_pp=%.2f  recovery_acc=%s  llm/1k=%.1f",
                m["coverage_gain_pp"], m["recovery_accuracy"], m["llm_calls_per_1000_products"])


if __name__ == "__main__":
    main()
