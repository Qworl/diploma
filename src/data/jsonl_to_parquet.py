"""Stream OFF products.jsonl.gz → enriched parquet (no full decompression).

OFF JSONL dump has rich fields not in CSV-derived parquet:
  - images URLs / nutrition images
  - all per-language fields
  - labels_tags, categories_tags fully exploded
  - ingredients hierarchy
  - manufacturing_places, packaging, etc.

We keep only fields useful for downstream cascade enrichment & annotation,
write in 100k-row parquet chunks to control memory.

Output:
  datasets/raw/off_jsonl_enriched.parquet (single concatenated, ~5-10 GB)
  OR datasets/raw/off_jsonl_chunks/part_NNNN.parquet (sharded, easier to stream)
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout, force=True,
)
logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("/tmp/openfoodfacts-products.jsonl.gz")
DEFAULT_OUTDIR = Path("datasets/raw/off_jsonl_chunks")
CHUNK_SIZE = 100_000

KEEP_FIELDS = (
    "code", "product_name", "brands", "ingredients_text", "quantity",
    "categories", "categories_tags", "labels_tags", "ingredients_analysis_tags",
    "traces_tags", "countries_tags", "allergens_tags", "additives_tags",
    "languages", "main_language", "lang", "lc",
    "image_url", "image_small_url", "image_front_url", "image_ingredients_url",
    "manufacturing_places", "manufacturing_places_tags", "packaging",
    "nutriscore_grade", "nova_group", "ecoscore_grade",
    "fat_100g", "saturated-fat_100g", "carbohydrates_100g", "sugars_100g",
    "proteins_100g", "salt_100g", "fiber_100g", "energy-kcal_100g",
    "completeness", "states_tags", "last_modified_t",
)


def extract(row: dict) -> dict:
    """Extract only KEEP_FIELDS from a product JSON, flattening nutriments."""
    out = {}
    for f in KEEP_FIELDS:
        v = row.get(f)
        if isinstance(v, (list, tuple)):
            v = "|".join(str(x) for x in v) if v else ""
        elif v is None:
            v = ""
        else:
            v = str(v)  # force str to avoid mixed-type parquet errors
        out[f] = v
    # Nutriments may be nested
    nut = row.get("nutriments", {}) or {}
    for nk in ("fat_100g", "saturated-fat_100g", "carbohydrates_100g",
               "sugars_100g", "proteins_100g", "salt_100g", "fiber_100g",
               "energy-kcal_100g"):
        if not out.get(nk):
            val = nut.get(nk, "")
            out[nk] = str(val) if val != "" else ""
    return out


def stream_jsonl(input_path: Path, outdir: Path, chunk_size: int = CHUNK_SIZE):
    outdir.mkdir(parents=True, exist_ok=True)
    buf: list[dict] = []
    n_total = 0
    n_chunks = 0
    t_start = time.time()
    skipped_parse = 0
    skipped_no_code = 0

    logger.info("Streaming %s → %s (chunk=%d rows)", input_path, outdir, chunk_size)
    with gzip.open(input_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped_parse += 1
                continue
            row = extract(obj)
            if not row.get("code"):
                skipped_no_code += 1
                continue
            buf.append(row)
            if len(buf) >= chunk_size:
                df = pd.DataFrame(buf)
                out_path = outdir / f"part_{n_chunks:04d}.parquet"
                df.to_parquet(out_path, index=False, compression="zstd")
                n_chunks += 1
                n_total += len(buf)
                elapsed = time.time() - t_start
                rate = n_total / max(elapsed, 1)
                logger.info("chunk %d saved: %d rows total, %.0f rows/sec, "
                            "skipped: %d parse, %d no_code",
                            n_chunks, n_total, rate, skipped_parse, skipped_no_code)
                buf = []
    # Flush remainder
    if buf:
        df = pd.DataFrame(buf)
        out_path = outdir / f"part_{n_chunks:04d}.parquet"
        df.to_parquet(out_path, index=False, compression="zstd")
        n_chunks += 1
        n_total += len(buf)

    elapsed = time.time() - t_start
    logger.info("DONE: %d chunks, %d total rows in %.0f sec (%.0f rows/sec)",
                n_chunks, n_total, elapsed, n_total / max(elapsed, 1))
    logger.info("Skipped: %d parse errors, %d no-code", skipped_parse, skipped_no_code)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input not found: %s. Download first via: "
                     "curl -L https://static.openfoodfacts.org/data/openfoodfacts-products.jsonl.gz "
                     "-o /tmp/openfoodfacts-products.jsonl.gz",
                     input_path)
        sys.exit(1)

    stream_jsonl(input_path, Path(args.outdir), args.chunk_size)


if __name__ == "__main__":
    main()
