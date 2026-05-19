"""Paired McNemar cascade vs each direct LLM on v2 gold, refusal-as-miss.
Bonferroni-corrected over kept model set (partner-input only)."""
import argparse
import glob
import json
import logging
from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from src.common import PROCESSED_DIR, setup_logging
from src.eval.cascade_predict import predict_cascade
from src.manual_label.schemas_loader import load_domain_attrs

logger = logging.getLogger(__name__)


def _llm_parquet_to_long(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    attrs = list(load_domain_attrs(domain))
    rows = []
    for _, r in df.iterrows():
        try:
            parsed = json.loads(r["parsed_json"]) if r.get("parsed_json") else {}
        except Exception:
            parsed = {}
        for a in attrs:
            v = parsed.get(a)
            rows.append({"code": str(r["code"]), "attr": a,
                         "predicted": None if v is None else str(v)})
    return pd.DataFrame(rows)


def paired_compare(cascade: pd.DataFrame, llm: pd.DataFrame,
                   gold: pd.DataFrame, *, model_name: str) -> pd.DataFrame:
    gold = gold[~gold["gold_is_null"]].copy()
    gold["code"] = gold["code"].astype(str)
    m = gold.merge(cascade[["code", "attr", "predicted"]]
                       .rename(columns={"predicted": "cascade_pred"}),
                   on=["code", "attr"], how="left")
    m = m.merge(llm[["code", "attr", "predicted"]]
                    .rename(columns={"predicted": "llm_pred"}),
                on=["code", "attr"], how="left")
    m["cascade_correct"] = (m["cascade_pred"].astype(object)
                            == m["gold_value"].astype(object)).fillna(False).astype(int)
    m["llm_correct"] = (m["llm_pred"].astype(object)
                        == m["gold_value"].astype(object)).fillna(False).astype(int)
    n = len(m)
    b = int(((m["cascade_correct"] == 1) & (m["llm_correct"] == 0)).sum())
    c = int(((m["cascade_correct"] == 0) & (m["llm_correct"] == 1)).sum())
    table = [[0, b], [c, 0]]
    res = mcnemar(table, exact=(b + c < 25), correction=True)
    return pd.DataFrame([{
        "model": model_name,
        "n_cells": n,
        "cascade_acc": m["cascade_correct"].mean() if n else float("nan"),
        "llm_acc_refusal_as_miss": m["llm_correct"].mean() if n else float("nan"),
        "b_cascade_only_correct": b,
        "c_llm_only_correct": c,
        "mcnemar_p_raw": float(res.pvalue),
    }])


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--gold",
                   default=str(Path(PROCESSED_DIR) / "consensus_gold_v2_off_grounded.parquet"))
    p.add_argument("--logs-glob",
                   default=str(Path(PROCESSED_DIR) / "direct_llm_v2" / "*_partner.parquet"))
    p.add_argument("--out",
                   default=str(Path(PROCESSED_DIR) / "cascade_vs_llm_v2.parquet"))
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
    logger.info("Comparing cascade vs %d models", k)

    results = []
    # Cache cascade predictions per cat (avoid re-running for every model)
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
            cascade_cache[domain] = predict_cascade(products, category=f"{domain}_stratified")
        cascade = cascade_cache[domain]
        llm_long = _llm_parquet_to_long(df, domain=domain)
        cat_gold = gold[gold["category"] == domain]
        out_row = paired_compare(cascade, llm_long, cat_gold,
                                 model_name=f"{domain}/{model}")
        results.append(out_row)

    table = pd.concat(results, ignore_index=True)
    table["mcnemar_p_bonferroni"] = (table["mcnemar_p_raw"] * k).clip(upper=1.0)
    table.to_parquet(args.out, index=False)
    print(table.to_string(index=False))
    logger.info("Wrote %s (k=%d for Bonferroni)", args.out, k)


if __name__ == "__main__":
    main()
