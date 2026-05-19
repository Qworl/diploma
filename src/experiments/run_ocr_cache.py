"""Standalone OCR runner — extracts text from product images, saves to cache JSON.

Run as:
  OMP_NUM_THREADS=2 python3 src/experiments/run_ocr_cache.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from glob import glob
from pathlib import Path

# Configure logging to both stdout and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

WORKTREE_ROOT = Path(__file__).parent.parent.parent
IMAGES_DIR = WORKTREE_ROOT / "datasets" / "raw" / "off_images"
CACHE_PATH = WORKTREE_ROOT / "datasets" / "processed" / "ocr_text_cache.json"
MAX_SIZE = 5 * 1024 * 1024
CONF_THRESH = 0.3


def main() -> None:
    img_files = sorted(glob(str(IMAGES_DIR / "*.jpg")))
    logger.info("Total images: %d in %s", len(img_files), IMAGES_DIR)

    if not img_files:
        logger.error("No images found. EXP10 must download images first.")
        sys.exit(1)

    # Load partial cache if exists
    cache: dict[str, str] = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        logger.info("Loaded partial cache: %d entries", len(cache))

    import easyocr  # noqa: PLC0415 — lazy import
    reader = easyocr.Reader(["en", "fr", "de", "it", "es"], gpu=False, verbose=False)
    logger.info("EasyOCR reader ready")

    t0 = time.time()
    processed = 0
    skipped_cached = 0
    skipped_size = 0

    for i, img_path in enumerate(img_files, 1):
        code = Path(img_path).stem

        if code in cache:
            skipped_cached += 1
            continue

        file_size = os.path.getsize(img_path)
        if file_size > MAX_SIZE:
            logger.debug("Skip (size): %s (%.1f MB)", code, file_size / 1e6)
            cache[code] = ""
            skipped_size += 1
            continue

        try:
            results = reader.readtext(img_path, detail=1)
            ocr_text = " ".join(t for _, t, c in results if c >= CONF_THRESH)
            cache[code] = ocr_text.strip()
            processed += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR failed %s: %s", code, exc)
            cache[code] = ""

        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(img_files) - i) / rate if rate > 0 else 0
            logger.info(
                "  %d/%d  processed=%d cached=%d size_skip=%d  rate=%.1f/s  ETA=%.0fs",
                i, len(img_files), processed, skipped_cached, skipped_size, rate, eta,
            )
            # Incremental save every 50
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)

    # Final save
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    elapsed = time.time() - t0
    rate = processed / elapsed if elapsed > 0 else 0
    non_empty = sum(1 for v in cache.values() if v.strip())
    logger.info(
        "Done: %d processed, %d from cache, %d size-skipped, %.1fs total (%.2f/s)",
        processed, skipped_cached, skipped_size, elapsed, rate,
    )
    logger.info("Non-empty OCR text: %d / %d total cache entries", non_empty, len(cache))

    # Sample outputs
    items_by_len = sorted(
        [(k, v) for k, v in cache.items() if v.strip()],
        key=lambda x: len(x[1]),
        reverse=True,
    )
    print("\n=== Top OCR outputs (longest) ===")
    for code, text in items_by_len[:5]:
        print(f"  [{code}] {text[:120]!r}")
    print("\n=== Shortest OCR outputs ===")
    for code, text in items_by_len[-3:]:
        print(f"  [{code}] {text!r}")


if __name__ == "__main__":
    main()
