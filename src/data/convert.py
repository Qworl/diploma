"""
CSV.gz / CSV → parquet converter for Open Food Facts family.

Поддерживает все 4 проекта семейства: off, obf, opff, opf.
Читает .csv.gz напрямую (без распаковки) через pyarrow, или CSV непосредственно.
Использует pyarrow.csv multi-threaded в C++ и пишет parquet напрямую (~5-10x быстрее чем pandas).

Progress: ~1-2 sec per 100k rows.

Usage:
    python -m src.data.convert --source off
    python -m src.data.convert --source obf
    python -m src.data.convert --source opff --force
    python -m src.data.convert --source off --csv  # конвертировать уже распакованный CSV
"""

import argparse
import gzip
import logging
import os
import time

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from src.common import RAW_DIR, setup_logging

logger = logging.getLogger(__name__)

# Все 4 проекта семейства Open Food Facts
SOURCES = {
    "off":  {"domain": "openfoodfacts.org",  "name": "Open Food Facts"},
    "obf":  {"domain": "openbeautyfacts.org", "name": "Open Beauty Facts"},
    "opff": {"domain": "openpetfoodfacts.org", "name": "Open Pet Food Facts"},
    "opf":  {"domain": "openproductsfacts.org", "name": "Open Products Facts"},
}

# Базовые колонки, общие для всех источников. apply_off_labels() ожидает их же.
BASE_COLUMNS = [
    "code", "product_name", "generic_name", "brands",
    "categories_tags", "labels_tags",
    "ingredients_text", "ingredients_tags", "ingredients_analysis_tags",
    "allergens_tags", "traces_tags",
    "quantity", "serving_size",
    "completeness", "data_quality_tags",
    "countries_tags", "image_url",
]

# Дополнительные колонки per-source
SOURCE_EXTRA_COLUMNS = {
    "off": [
        "fat_100g", "proteins_100g", "carbohydrates_100g",
        "energy-kcal_100g", "sugars_100g", "fiber_100g",
        "salt_100g", "sodium_100g", "alcohol_100g",
        "nutriscore_grade", "nova_group",
    ],
    "obf": [
        "periods_after_opening_tags",
    ],
    "opff": [
        "fat_100g", "proteins_100g", "carbohydrates_100g",
        "energy-kcal_100g", "fiber_100g",
    ],
    "opf": [],
}


def get_input_paths(source: str) -> tuple[str | None, str | None]:
    """Return (csv_gz_path, csv_path) for given source. At least one must exist."""
    domain = SOURCES[source]["domain"]
    csv_gz_filename = f"en.{domain}.products.csv.gz"
    csv_filename = f"en.{domain}.products.csv"
    csv_gz_path = os.path.join(RAW_DIR, csv_gz_filename)
    csv_path = os.path.join(RAW_DIR, csv_filename)
    return (csv_gz_path if os.path.exists(csv_gz_path) else None,
            csv_path if os.path.exists(csv_path) else None)


def get_output_path(source: str) -> str:
    """Parquet выход."""
    domain = SOURCES[source]["domain"]
    filename = f"en.{domain}.products.parquet"
    return os.path.join(RAW_DIR, filename)


def discover_header(csv_path: str, is_gzip: bool = False) -> list[str]:
    """Прочитать первую строку CSV / CSV.gz."""
    if is_gzip:
        with gzip.open(csv_path, "rt", encoding="utf-8") as f:
            return f.readline().rstrip("\n").split("\t")
    else:
        with open(csv_path, "r", encoding="utf-8") as f:
            return f.readline().rstrip("\n").split("\t")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, choices=list(SOURCES.keys()),
                        help="Источник данных (off/obf/opff/opf)")
    parser.add_argument("--force", action="store_true",
                        help="Перезаписать parquet, даже если он существует")
    parser.add_argument("--csv", action="store_true",
                        help="Использовать уже распакованный CSV вместо CSV.gz")
    args = parser.parse_args()

    setup_logging()

    csv_gz_path, csv_path = get_input_paths(args.source)
    parquet_path = get_output_path(args.source)

    # Decide which input to use
    if args.csv and csv_path:
        input_path = csv_path
        is_gzip = False
        logger.info("Using CSV: %s", input_path)
    elif csv_gz_path:
        input_path = csv_gz_path
        is_gzip = True
        logger.info("Using CSV.gz: %s", input_path)
    elif csv_path:
        input_path = csv_path
        is_gzip = False
        logger.info("Using CSV: %s", input_path)
    else:
        raise FileNotFoundError(
            f"No source file for {args.source} in {RAW_DIR}. "
            f"Expected either en.{SOURCES[args.source]['domain']}.products.csv.gz "
            f"or en.{SOURCES[args.source]['domain']}.products.csv"
        )

    if os.path.exists(parquet_path) and not args.force:
        size_mb = os.path.getsize(parquet_path) / 1024 ** 2
        logger.info("Parquet уже существует: %s (%.0f MB). Используй --force для перегенерации.",
                    parquet_path, size_mb)
        return

    input_size_gb = os.path.getsize(input_path) / 1024 ** 3
    logger.info("Source: %s (%s)", SOURCES[args.source]["name"], args.source)
    logger.info("Input: %.2f GB → Output: %s", input_size_gb, parquet_path)

    keep_columns = BASE_COLUMNS + SOURCE_EXTRA_COLUMNS[args.source]

    # Discover header
    header = discover_header(input_path, is_gzip=is_gzip)
    available_cols = [c for c in keep_columns if c in header]
    missing = [c for c in keep_columns if c not in header]
    if missing:
        logger.warning("Колонок нет в %s header (пропущены): %s", args.source, missing)
    logger.info("Будет сохранено %d/%d колонок", len(available_cols), len(keep_columns))

    read_options = pcsv.ReadOptions(block_size=64 * 1024 * 1024)
    parse_options = pcsv.ParseOptions(delimiter="\t", invalid_row_handler=lambda r: "skip")
    convert_options = pcsv.ConvertOptions(
        include_columns=available_cols,
        include_missing_columns=False,
        column_types={c: pa.string() for c in available_cols},
    )

    t0 = time.time()
    logger.info("Открываю CSV reader (pyarrow auto-detects gzip by extension)...")
    reader = pcsv.open_csv(
        input_path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )

    writer = None
    total_rows = 0
    batch_idx = 0

    try:
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break

            if writer is None:
                writer = pq.ParquetWriter(parquet_path, batch.schema, compression="snappy")

            writer.write_batch(batch)
            total_rows += batch.num_rows
            batch_idx += 1
            if batch_idx % 5 == 0:
                elapsed = time.time() - t0
                rate = total_rows / elapsed if elapsed > 0 else 0
                logger.info("  batch %d: %d rows total, %.0f rows/s, %.0fs elapsed",
                            batch_idx, total_rows, rate, elapsed)
    finally:
        if writer is not None:
            writer.close()
        reader.close()

    elapsed = time.time() - t0
    out_size_mb = os.path.getsize(parquet_path) / 1024 ** 2
    logger.info("Готово. %d строк за %.1fs (%.0f rows/s)",
                total_rows, elapsed, total_rows / elapsed if elapsed > 0 else 0)
    logger.info("Parquet: %.0f MB", out_size_mb)


if __name__ == "__main__":
    main()
