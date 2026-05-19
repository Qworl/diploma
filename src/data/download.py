"""
Download OFF / OBF / OPFF / OPF dumps to datasets/raw/.

Поддерживаемые источники:
    off  — Open Food Facts (https://world.openfoodfacts.org)
    obf  — Open Beauty Facts (https://world.openbeautyfacts.org)
    opff — Open Pet Food Facts (https://world.openpetfoodfacts.org)
    opf  — Open Products Facts (https://world.openproductsfacts.org)

Usage:
    python -m src.data.download --source off
    python -m src.data.download --source obf
    python -m src.data.download --source opff --format jsonl
    python -m src.data.download --source off --format csv
"""

import argparse
import logging
import os
import subprocess
import sys

from src.common import RAW_DIR, setup_logging

logger = logging.getLogger(__name__)

# Все 4 проекта семейства Open Food Facts следуют одинаковому паттерну URL:
#   https://static.{domain}/data/en.{domain}.products.csv.gz
SOURCES = {
    "off":  {"domain": "openfoodfacts.org",  "name": "Open Food Facts"},
    "obf":  {"domain": "openbeautyfacts.org", "name": "Open Beauty Facts"},
    "opff": {"domain": "openpetfoodfacts.org", "name": "Open Pet Food Facts"},
    "opf":  {"domain": "openproductsfacts.org", "name": "Open Products Facts"},
}

FORMATS = {
    "csv":   "en.{domain}.products.csv.gz",
    "jsonl": "{base}-products.jsonl.gz",  # base = openfoodfacts/openbeautyfacts/...
}


def build_url(source: str, fmt: str) -> tuple[str, str]:
    """Возвращает (url, output_filename) для заданного источника и формата."""
    domain = SOURCES[source]["domain"]
    base = domain.split(".")[0]  # openfoodfacts из openfoodfacts.org

    if fmt == "csv":
        filename = f"en.{domain}.products.csv.gz"
    elif fmt == "jsonl":
        filename = f"{base}-products.jsonl.gz"
    else:
        raise ValueError(f"Unknown format: {fmt}")

    url = f"https://static.{domain}/data/{filename}"
    return url, filename


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, choices=list(SOURCES.keys()),
                        help="Источник данных")
    parser.add_argument("--format", default="csv", choices=list(FORMATS.keys()),
                        help="Формат скачивания (default: csv)")
    parser.add_argument("--force", action="store_true",
                        help="Перекачать, даже если файл уже существует")
    args = parser.parse_args()

    setup_logging()

    url, filename = build_url(args.source, args.format)
    output_path = os.path.join(RAW_DIR, filename)

    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(output_path) and not args.force:
        size_gb = os.path.getsize(output_path) / (1024 ** 3)
        logger.info("Файл уже существует: %s (%.2f GB). Пропускаю. Используй --force чтобы перекачать.",
                    output_path, size_gb)
        return

    logger.info("Скачиваю %s (%s, %s)...", SOURCES[args.source]["name"], args.source, args.format)
    logger.info("URL: %s", url)
    logger.info("Output: %s", output_path)

    cmd = ["curl", "-L", "-o", output_path, "--progress-bar", url]
    subprocess.run(cmd, check=True)

    size_gb = os.path.getsize(output_path) / (1024 ** 3)
    logger.info("Готово! Файл сохранён в %s (%.2f GB)", output_path, size_gb)


if __name__ == "__main__":
    main()
