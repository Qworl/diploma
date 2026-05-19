"""CLI: compute per-attribute validation thresholds and save as JSON.

Usage:
    python -m src.pipeline.bayes.calibrate_validator --category pasta_stratified
    python -m src.pipeline.bayes.calibrate_validator --category pasta_stratified --q 0.05
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from pgmpy.inference import VariableElimination

from src.pipeline.bayes.validate import calibrate_thresholds


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True,
                   help="e.g. pasta_stratified, chocolate_stratified, beverages_stratified")
    p.add_argument("--q", type=float, default=0.05,
                   help="percentile (default 0.05 = 5th)")
    args = p.parse_args()

    models_dir = Path("models")
    data_dir = Path("datasets/processed")

    bayes_path = models_dir / f"{args.category}_bayesian.pkl"
    data_path = data_dir / f"{args.category}_silver_standard.parquet"
    out_path = models_dir / f"{args.category}_validation_thresholds.json"

    if not bayes_path.exists():
        print(f"ERROR: {bayes_path} not found", file=sys.stderr)
        return 2
    if not data_path.exists():
        print(f"ERROR: {data_path} not found", file=sys.stderr)
        return 2

    with open(bayes_path, "rb") as f:
        bayes = pickle.load(f)
    df = pd.read_parquet(data_path)
    inference = VariableElimination(bayes)

    print(f"Calibrating {args.category} (q={args.q}) on {len(df)} rows ...")
    thresholds = calibrate_thresholds(bayes, df, inference, q=args.q)

    payload = {
        "category": args.category,
        "q": args.q,
        "n_train_rows": int(len(df)),
        "thresholds": thresholds,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Wrote {out_path}")
    for attr, thr in sorted(thresholds.items()):
        print(f"  {attr}: {thr:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
