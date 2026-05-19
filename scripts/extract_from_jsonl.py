"""Stream-parse OFF JSONL dump, extract our codes, save to cache.

Usage:
  scripts/extract_from_jsonl.py --jsonl /tmp/openfoodfacts-products.jsonl.gz \\
    --codes-file /tmp/parquet_derived_codes.txt \\
    --cache-dir datasets/manual_label/off_cache
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, type=Path)
    p.add_argument("--codes-file", required=True, type=Path)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing cache files (else skip).")
    args = p.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    with args.codes_file.open() as f:
        target = {l.strip() for l in f if l.strip()}
    print(f"Looking for {len(target)} codes")

    found = 0
    written = 0
    skipped = 0
    n_lines = 0
    t0 = time.monotonic()

    with gzip.open(args.jsonl, "rt", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            if n_lines % 200_000 == 0:
                elapsed = time.monotonic() - t0
                rate = n_lines / max(elapsed, 1)
                print(f"  scanned {n_lines:,} lines ({rate:.0f}/s) — found {found} — written {written}")
            line = line.strip()
            if not line:
                continue
            # Cheap pre-filter — only parse JSON if "code" substring matches one of ours
            try:
                obj = json.loads(line)
            except Exception:
                continue
            code = str(obj.get("code") or "").strip()
            if code not in target:
                continue
            found += 1
            out_path = args.cache_dir / f"{code}.json"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            with out_path.open("w", encoding="utf-8") as g:
                json.dump(obj, g, ensure_ascii=False)
            written += 1
            if found == len(target):
                print(f"  all {len(target)} codes found, early exit")
                break

    elapsed = time.monotonic() - t0
    print(f"DONE: scanned {n_lines:,} lines in {elapsed:.0f}s — "
          f"found={found} written={written} skipped_existing={skipped}")


if __name__ == "__main__":
    main()
