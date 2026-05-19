"""Ceiling-check appendix table: partner_input vs off_grounded vs cascade
for the 2 ceiling-check models (gpt-4o-mini, gemini-2.5-flash).
"""
import argparse
import glob
import json
import logging
from pathlib import Path

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.manual_label.schemas_loader import load_domain_attrs

logger = logging.getLogger(__name__)

CEILING_MODELS = {"openai/gpt-4o-mini", "google/gemini-2.5-flash"}


def _llm_long_per_call(llm_df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Explode each call's parsed_json into (code, attr, predicted) rows."""
    attrs = list(load_domain_attrs(domain))
    rows = []
    for _, r in llm_df.iterrows():
        try:
            parsed = json.loads(r["parsed_json"]) if r.get("parsed_json") else {}
        except Exception:
            parsed = {}
        for a in attrs:
            v = parsed.get(a)
            rows.append({"code": str(r["code"]), "attr": a,
                         "predicted": None if v is None else str(v)})
    return pd.DataFrame(rows)


def _accuracy(llm_long: pd.DataFrame, gold: pd.DataFrame) -> float:
    """Refusal-as-miss accuracy on non-null gold cells."""
    gold = gold[~gold["gold_is_null"]].copy()
    gold["code"] = gold["code"].astype(str)
    m = gold.merge(llm_long, on=["code", "attr"], how="left")
    correct = (m["predicted"].astype(object) == m["gold_value"].astype(object)).fillna(False)
    if len(m) == 0:
        return float("nan")
    return float(correct.sum() / len(m))


def build_table(llm_acc: pd.DataFrame, cascade_acc: dict) -> pd.DataFrame:
    """Compose per-model per-cat ceiling-check rows."""
    rows = []
    pivot = llm_acc.pivot_table(index=["model", "category"],
                                columns="context_mode", values="accuracy")
    for (model, cat), r in pivot.iterrows():
        partner = r.get("partner_input")
        off = r.get("off_grounded")
        casc = cascade_acc.get(cat)
        if partner is None or off is None or casc is None:
            continue
        rows.append({
            "model": model, "category": cat,
            "acc_partner_input": float(partner),
            "acc_off_grounded": float(off),
            "acc_cascade": float(casc),
            "ceiling_delta_pp": (float(off) - float(partner)) * 100,
            "cascade_advantage_vs_partner_pp": (float(casc) - float(partner)) * 100,
        })
    return pd.DataFrame(rows)


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--gold",
                   default=str(Path(PROCESSED_DIR) / "consensus_gold_v2_off_grounded.parquet"))
    p.add_argument("--logs-glob",
                   default=str(Path(PROCESSED_DIR) / "direct_llm_v2" / "*.parquet"))
    p.add_argument("--cascade-headline",
                   default=str(Path(PROCESSED_DIR) / "headline_results_off_grounded.parquet"))
    p.add_argument("--out",
                   default=str(Path(PROCESSED_DIR) / "llm_ceiling_check.parquet"))
    args = p.parse_args()

    gold = pd.read_parquet(args.gold)
    gold["code"] = gold["code"].astype(str)
    headline = pd.read_parquet(args.cascade_headline)
    cascade_acc_by_cat = {
        cat: float(g["n_correct"].sum() / g["n_non_null_gold"].sum())
        for cat, g in headline.groupby("category")
    }

    rows = []
    for log_path in sorted(glob.glob(args.logs_glob)):
        df = pd.read_parquet(log_path)
        if len(df) == 0:
            continue
        model = df["model"].iloc[0]
        if model not in CEILING_MODELS:
            continue
        domain = df["domain"].iloc[0]
        context_mode = df["context_mode"].iloc[0]
        llm_long = _llm_long_per_call(df, domain)
        cat_gold = gold[gold["category"] == domain]
        acc = _accuracy(llm_long, cat_gold)
        rows.append({"model": model, "category": domain,
                     "context_mode": context_mode, "accuracy": acc})

    llm_acc = pd.DataFrame(rows)
    table = build_table(llm_acc, cascade_acc_by_cat)
    table.to_parquet(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
