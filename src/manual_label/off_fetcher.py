"""Fetch Open Food Facts product JSON via public REST API, cache locally.

Public endpoint: https://world.openfoodfacts.org/api/v2/product/{code}.json
Returns the `product` sub-dict on success. Raises OFFFetchError on HTTP error,
404, or status:0 (product not in DB).

Cache: one JSON file per product code under `cache_dir`. Subsequent calls for
the same code skip HTTP and read from cache.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import requests

OFF_API_URL = "https://world.openfoodfacts.org/api/v2/product/{code}.json"
DEFAULT_RATE_LIMIT_SEC = 2.0  # 0.5 req/s, safe for OFF public API
# OFF requires a meaningful User-Agent; default python-requests UA returns 403.
DEFAULT_USER_AGENT = (
    "ai-attributes-thesis/1.0 "
    "(frolovmika@gmail.com; https://github.com/MikhailFrolov/ai-attributes-thesis)"
)


class OFFFetchError(Exception):
    """OFF API returned an error or product not found."""


def fetch_off_product(
    code: str,
    *,
    cache_dir: Path,
    rate_limit_sec: float = DEFAULT_RATE_LIMIT_SEC,
    timeout_sec: int = 30,
) -> dict:
    """Fetch product data from OFF, returning the `product` dict.

    Cached on disk: subsequent calls for the same code skip HTTP.
    Raises OFFFetchError on 404 or status:0.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{code}.json"

    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            cached = json.load(f)
        return cached

    url = OFF_API_URL.format(code=code)
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    # 429 retry: honor Retry-After header, else exp backoff 5/10/20s (max 3 tries).
    backoffs = [5, 10, 20]
    resp = requests.get(url, timeout=timeout_sec, headers=headers)
    for attempt, backoff in enumerate(backoffs):
        if resp.status_code != 429:
            break
        retry_after = resp.headers.get("Retry-After")
        wait = int(retry_after) if retry_after and retry_after.isdigit() else backoff
        time.sleep(wait)
        resp = requests.get(url, timeout=timeout_sec, headers=headers)
    if resp.status_code == 404:
        raise OFFFetchError(f"404 for code {code}")
    if resp.status_code != 200:
        raise OFFFetchError(f"HTTP {resp.status_code} for code {code}")

    body = resp.json()
    if body.get("status") != 1:
        raise OFFFetchError(
            f"OFF status {body.get('status')} for code {code}: "
            f"{body.get('status_verbose', 'not found')}"
        )

    product = body.get("product", {})
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(product, f, ensure_ascii=False)

    if rate_limit_sec > 0:
        time.sleep(rate_limit_sec)

    return product


def main() -> None:
    import argparse

    try:
        from src.common import setup_logging
        setup_logging()
    except Exception:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Populate OFF product cache from a codes file.")
    parser.add_argument("--codes-file", required=True, help="Text file (one code per line) or CSV with 'code' column")
    parser.add_argument("--cache-dir", default="datasets/manual_label/off_cache", help="Directory for cached JSON files")
    parser.add_argument("--rate-limit-sec", type=float, default=DEFAULT_RATE_LIMIT_SEC, help="Seconds between OFF API requests")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="User-Agent header for OFF API")
    args = parser.parse_args()

    codes_path = Path(args.codes_file)
    cache_dir = Path(args.cache_dir)

    # Read codes — CSV or plain text
    codes: list[str] = []
    seen: set[str] = set()
    if codes_path.suffix.lower() == ".csv":
        with codes_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                c = str(row.get("code") or "").strip()
                if c and c not in seen:
                    codes.append(c)
                    seen.add(c)
    else:
        for line in codes_path.read_text(encoding="utf-8").splitlines():
            c = line.strip()
            if c and c not in seen:
                codes.append(c)
                seen.add(c)

    logger.info("Loaded %d unique codes from %s", len(codes), codes_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_cached = 0
    n_skipped = 0
    n_failed = 0

    for i, code in enumerate(codes, 1):
        cache_path = cache_dir / f"{code}.json"
        if cache_path.exists():
            n_skipped += 1
            if i % 50 == 0:
                logger.info("Progress %d/%d — cached=%d skipped=%d failed=%d", i, len(codes), n_cached, n_skipped, n_failed)
            continue

        try:
            fetch_off_product(code, cache_dir=cache_dir, rate_limit_sec=args.rate_limit_sec)
            n_cached += 1
        except OFFFetchError as exc:
            logger.warning("OFF fetch error for %s: %s", code, exc)
            n_failed += 1
        except Exception as exc:
            logger.warning("Unexpected error for %s: %s — sleeping 10s", code, exc)
            n_failed += 1
            time.sleep(10)

        if i % 50 == 0:
            logger.info("Progress %d/%d — cached=%d skipped=%d failed=%d", i, len(codes), n_cached, n_skipped, n_failed)

    print(f"Done. cached={n_cached} skipped_existing={n_skipped} failed={n_failed}")


if __name__ == "__main__":
    main()
