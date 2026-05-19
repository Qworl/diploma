"""Aggregate per-call LLM logs into per (model, context_mode) cost/latency table."""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import PROCESSED_DIR


def aggregate(per_call: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, mode), g in per_call.groupby(["model", "context_mode"]):
        rows.append({
            "model": model,
            "context_mode": mode,
            "n_calls": len(g),
            "total_cost_usd": float(g["cost_usd"].sum()),
            "cost_per_1k_products_usd": float(g["cost_usd"].sum() / len(g) * 1000),
            "p50_latency_ms": float(np.percentile(g["latency_ms"], 50)),
            "p95_latency_ms": float(np.percentile(g["latency_ms"], 95)),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs-glob",
                   default=str(Path(PROCESSED_DIR) / "direct_llm_v2" / "*.parquet"))
    p.add_argument("--out",
                   default=str(Path(PROCESSED_DIR) / "cost_latency_table.parquet"))
    args = p.parse_args()
    dfs = [pd.read_parquet(f) for f in glob.glob(args.logs_glob)]
    per_call = pd.concat(dfs, ignore_index=True)
    table = aggregate(per_call)
    table.to_parquet(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
