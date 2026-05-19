"""Merge per-batch Opus decision JSONs into one file, validate non-overlap.

Run:
    python -m src.manual_label.merge_opus_batches \\
        --inputs datasets/manual_label/opus_batches/batch_1_decisions.json \\
                 datasets/manual_label/opus_batches/batch_2_decisions.json \\
                 datasets/manual_label/opus_batches/batch_3_decisions.json \\
                 datasets/manual_label/opus_batches/batch_4_decisions.json \\
        --out datasets/manual_label/opus_batches/opus_decisions_all.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge(inputs: list[Path], out: Path) -> dict:
    merged: dict[str, dict] = {}
    overlaps: list[str] = []
    per_file_counts: dict[str, int] = {}
    for path in inputs:
        with path.open() as f:
            data = json.load(f)
        per_file_counts[str(path)] = len(data)
        for code, attrs in data.items():
            if code in merged:
                overlaps.append(code)
                continue
            merged[code] = attrs
    if overlaps:
        raise ValueError(f"Codes overlap across batches: {overlaps[:5]}... (total {len(overlaps)})")
    with out.open("w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    return {
        "total_codes": len(merged),
        "per_file": per_file_counts,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    res = merge(args.inputs, args.out)
    print(f"Merged {res['total_codes']} codes into {args.out}")
    for path, n in res["per_file"].items():
        print(f"  {path}: {n}")


if __name__ == "__main__":
    main()
