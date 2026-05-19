"""Paired McNemar cascade (HYBRID) vs each direct LLM on v2 gold.

Same logic as cascade_vs_llm_v2.py but uses hybrid cascade models
(silver + 5x v2 gold, all data, no hold-out).

WARNING: The hybrid models have seen ALL v2 gold codes during training.
Accuracy numbers here are optimistically biased (train=eval overlap).
Use cascade_vs_llm_v2.parquet (silver cascade) for the honest comparison.
"""
import argparse
import glob
import json
import logging
from pathlib import Path

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.eval.cascade_predict import predict_cascade
from src.eval.cascade_vs_llm_v2 import paired_compare, _llm_parquet_to_long
from src.manual_label.schemas_loader import load_domain_attrs

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--gold",
                   default=str(Path(PROCESSED_DIR) / "consensus_gold_v2_off_grounded.parquet"))
    p.add_argument("--logs-glob",
                   default=str(Path(PROCESSED_DIR) / "direct_llm_v2" / "*_partner.parquet"))
    p.add_argument("--out",
                   default=str(Path(PROCESSED_DIR) / "cascade_vs_llm_v2_hybrid.parquet"))
    args = p.parse_args()

    gold = pd.read_parquet(args.gold)
    gold["code"] = gold["code"].astype(str)
    log_paths = sorted(glob.glob(args.logs_glob))
    models_kept = set()
    for lp in log_paths:
        df = pd.read_parquet(lp)
        if len(df):
            models_kept.add(df["model"].iloc[0])
    k = len(models_kept)
    logger.info("Comparing hybrid cascade vs %d models", k)

    results = []
    # Cache cascade predictions per cat (use_hybrid=True)
    cascade_cache = {}
    for log_path in log_paths:
        df = pd.read_parquet(log_path)
        if len(df) == 0:
            continue
        model = df["model"].iloc[0]
        domain = df["domain"].iloc[0]
        codes = sorted(df["code"].astype(str).unique())
        if domain not in cascade_cache:
            silver = pd.read_parquet(
                Path(PROCESSED_DIR) / f"{domain}_stratified_silver_standard.parquet")
            silver["code"] = silver["code"].astype(str)
            products = silver[silver["code"].isin(codes)].copy()
            cascade_cache[domain] = predict_cascade(
                products, category=f"{domain}_stratified", use_hybrid=True)
        cascade = cascade_cache[domain]
        llm_long = _llm_parquet_to_long(df, domain=domain)
        cat_gold = gold[gold["category"] == domain]
        out_row = paired_compare(cascade, llm_long, cat_gold,
                                 model_name=f"{domain}/{model}")
        results.append(out_row)

    table = pd.concat(results, ignore_index=True)
    table["mcnemar_p_bonferroni"] = (table["mcnemar_p_raw"] * k).clip(upper=1.0)
    table.to_parquet(args.out, index=False)
    print("\n=== Hybrid cascade vs LLM (BIASED — train=eval overlap) ===")
    print(table.to_string(index=False))
    logger.info("Wrote %s (k=%d for Bonferroni)", args.out, k)

    # Also compare to silver cascade vs LLM
    silver_path = Path(PROCESSED_DIR) / "cascade_vs_llm_v2.parquet"
    if silver_path.exists():
        silver_res = pd.read_parquet(silver_path)
        print("\n=== Silver cascade vs LLM (honest, from cascade_vs_llm_v2.py) ===")
        print(silver_res[["model", "cascade_acc", "llm_acc_refusal_as_miss"]].to_string(index=False))
        print("\n=== Delta: hybrid_cascade_acc - silver_cascade_acc ===")
        merged = table.merge(
            silver_res[["model", "cascade_acc"]].rename(columns={"cascade_acc": "silver_cascade_acc"}),
            on="model", how="left",
        )
        merged["delta_pp"] = (merged["cascade_acc"] - merged["silver_cascade_acc"]) * 100
        print(merged[["model", "silver_cascade_acc", "cascade_acc", "llm_acc_refusal_as_miss", "delta_pp"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
