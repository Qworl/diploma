"""P2-EXP10 Phase 1: Download product images from OFF for all gold codes.

Reads off_cache/{code}.json → extracts image_url → downloads to
datasets/raw/off_images/{code}.jpg.  Skips if file already exists.

Rate limit: 10 req/sec (gentle; 5 is slow for ~2500 images).
Timeout per request: 10s.  Retries: 2.

Usage:
  python scripts/download_off_images.py
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WORKTREE_ROOT = Path(__file__).parent.parent
CACHE_DIR = WORKTREE_ROOT / "datasets" / "manual_label" / "off_cache"
OUT_DIR = WORKTREE_ROOT / "datasets" / "raw" / "off_images"
GOLD_PATH = WORKTREE_ROOT / "datasets" / "processed" / "consensus_gold_v2_expanded.parquet"

RATE_LIMIT_RPS = 10   # requests per second
TIMEOUT = 10          # seconds per request
RETRIES = 2

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ai_attributes_thesis/1.0 (research)"})


def _get_image_url(code: str) -> str | None:
    """Read off_cache JSON and return image_url or None."""
    fpath = CACHE_DIR / f"{code}.json"
    if not fpath.exists():
        return None
    try:
        with open(fpath, encoding="utf-8") as f:
            d = json.load(f)
        return d.get("image_url") or d.get("image_front_url")
    except Exception:
        return None


def _download(url: str, dest: Path) -> bool:
    """Download URL to dest. Returns True on success."""
    for attempt in range(RETRIES + 1):
        try:
            resp = SESSION.get(url, timeout=TIMEOUT, stream=True)
            if resp.status_code == 200:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            elif resp.status_code == 404:
                return False  # no point retrying
            else:
                logger.warning("HTTP %d for %s (attempt %d)", resp.status_code, url, attempt + 1)
        except requests.RequestException as e:
            logger.warning("Request error for %s: %s (attempt %d)", url, e, attempt + 1)
        if attempt < RETRIES:
            time.sleep(0.5)
    return False


def main() -> None:
    import pandas as pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)
    all_codes = sorted(gold["code"].unique().tolist())
    logger.info("Total unique codes in gold: %d", len(all_codes))

    # Collect (code, url) pairs to download
    to_download: list[tuple[str, str]] = []
    skipped_no_url = 0
    skipped_exists = 0

    for code in all_codes:
        dest = OUT_DIR / f"{code}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            skipped_exists += 1
            continue
        url = _get_image_url(code)
        if url and url.startswith("http"):
            to_download.append((code, url))
        else:
            skipped_no_url += 1

    logger.info(
        "To download: %d | already exists: %d | no URL: %d",
        len(to_download), skipped_exists, skipped_no_url,
    )

    min_interval = 1.0 / RATE_LIMIT_RPS
    ok = 0
    fail = 0
    t_start = time.time()

    for code, url in tqdm(to_download, desc="Downloading images"):
        t0 = time.time()
        dest = OUT_DIR / f"{code}.jpg"
        if _download(url, dest):
            ok += 1
        else:
            fail += 1
        elapsed = time.time() - t0
        wait = min_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    wall = time.time() - t_start
    total_attempted = ok + fail
    logger.info(
        "Done: %d/%d succeeded, %d failed. Wall clock: %.1fs (%.1f img/s)",
        ok, total_attempted, fail, wall, total_attempted / max(wall, 1),
    )

    # Summary
    downloaded = sum(1 for c in all_codes if (OUT_DIR / f"{c}.jpg").exists())
    print(f"\nFinal coverage: {downloaded}/{len(all_codes)} codes have image files "
          f"({100*downloaded/len(all_codes):.1f}%)")


if __name__ == "__main__":
    main()
