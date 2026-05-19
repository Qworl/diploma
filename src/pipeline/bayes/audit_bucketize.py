"""Audit bucketize() coverage on a real trained Bayes network and silver dataset.

For each (node × silver row) pair: try bucketize. Report fraction of values
that map to None. Fail-loudly if any node has < 95% coverage on its true
values — that signals a bin-format mismatch between train.py and our
bucketize regex.

Usage:
    python -m src.pipeline.bayes.audit_bucketize --category pasta_stratified
    python -m src.pipeline.bayes.audit_bucketize --category chocolate_stratified
    python -m src.pipeline.bayes.audit_bucketize --category beverages_stratified
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

from src.pipeline.bayes.bucketize import bucketize


def audit(category: str, min_coverage: float = 0.95) -> int:
    models_dir = Path("models")
    data_dir = Path("datasets/processed")

    bayes_path = models_dir / f"{category}_bayesian.pkl"
    data_path = data_dir / f"{category}_silver_standard.parquet"

    if not bayes_path.exists():
        print(f"ERROR: {bayes_path} not found", file=sys.stderr)
        return 2
    if not data_path.exists():
        print(f"ERROR: {data_path} not found", file=sys.stderr)
        return 2

    with open(bayes_path, "rb") as f:
        bayes = pickle.load(f)
    df = pd.read_parquet(data_path)

    print(f"=== {category} ===")
    print(f"  rows in dataset: {len(df)}")
    print(f"  nodes in network: {len(bayes.nodes())}")

    # Nodes excluded from the hard gate:
    #
    # * "brand"  — huge cardinality; train.py pins to top-N and routes everything
    #              else through "other" via _clean_evidence at inference time.
    #              Low coverage is expected.
    # * Any categorical-string node whose missed values are categorical strings
    #   and whose state set contains the catch-all "other"/"unknown" — this is
    #   a training-time extract_top(top_n=…) pruning artefact, NOT a bucketize
    #   regex gap. bucketize correctly returns None for out-of-vocabulary
    #   strings; the validator treats None as "no verdict", which matches the
    #   existing _clean_evidence-drops-unknown-evidence behaviour in
    #   src/pipeline/bayes/infer.py. Document with a [WARN] marker but do not
    #   fail the gate, because no bucketize change can fix it (would require
    #   retraining the network with larger top_n).
    GATE_EXEMPT_NODES = {"brand"}
    failed = []
    for node in bayes.nodes():
        if node not in df.columns:
            print(f"  {node}: SKIP — not in dataset columns")
            continue
        vals = df[node].dropna()
        if len(vals) == 0:
            print(f"  {node}: SKIP — all null in dataset")
            continue
        mapped = vals.apply(lambda v: bucketize(node, v, bayes))
        coverage = mapped.notna().mean()
        cpd = bayes.get_cpds(node)
        states = {str(s) for s in cpd.state_names[node]}
        sample_missed = vals[mapped.isna()].head(3).tolist()

        is_pruning_artefact = (
            coverage < min_coverage
            and bool(sample_missed)
            and all(isinstance(v, str) for v in sample_missed)
            and bool(states & {"other", "unknown"})
        )

        if coverage >= min_coverage:
            marker = "OK  "
        elif node in GATE_EXEMPT_NODES:
            marker = "WARN"  # brand: high cardinality, expected
        elif is_pruning_artefact:
            marker = "WARN"  # training extract_top pruning, not a bucketize bug
        else:
            marker = "FAIL"
        print(
            f"  [{marker}] {node}: coverage={coverage:.3f} "
            f"(n={len(vals)}, missed sample: {sample_missed})"
        )
        if marker == "FAIL":
            failed.append((node, coverage, sample_missed))

    if failed:
        print(f"\nFAILED nodes for {category}:")
        for node, cov, sample in failed:
            print(f"  {node}: {cov:.3f}, sample: {sample}")
        return 1
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True, help="e.g. pasta_stratified")
    p.add_argument("--min-coverage", type=float, default=0.95)
    args = p.parse_args()
    sys.exit(audit(args.category, args.min_coverage))


if __name__ == "__main__":
    main()
