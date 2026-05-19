#!/bin/bash
# Background pipeline: download OFF JSONL.gz (~12GB) → stream to parquet chunks
# Logs to /tmp/off_pipeline.log; updates progress every chunk
# Usage: bash scripts/off_pipeline.sh &
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=/tmp/off_pipeline.log
JSONL=/tmp/openfoodfacts-products.jsonl.gz

echo "=== $(date +%H:%M:%S) START OFF pipeline ===" | tee -a "$LOG"

TARGET_BYTES=12000000000  # ~12 GB; resume until at least this size
ACTUAL=$(stat -f%z "$JSONL" 2>/dev/null || stat -c%s "$JSONL" 2>/dev/null || echo 0)
echo "Current size: $ACTUAL bytes (target >= $TARGET_BYTES)" | tee -a "$LOG"
if [ "$ACTUAL" -lt "$TARGET_BYTES" ]; then
  echo "Downloading (resume if partial)..." | tee -a "$LOG"
  # -C - resumes from where left off if partial; --retry handles transient errors
  curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors \
    "https://static.openfoodfacts.org/data/openfoodfacts-products.jsonl.gz" \
    -o "$JSONL" 2>&1 | tail -200 >> "$LOG"
fi

ACTUAL=$(stat -f%z "$JSONL" 2>/dev/null || stat -c%s "$JSONL" 2>/dev/null)
echo "Final size: $ACTUAL bytes" | tee -a "$LOG"

mkdir -p datasets/raw/off_jsonl_chunks

echo "=== $(date +%H:%M:%S) STREAM to parquet chunks ===" | tee -a "$LOG"
.venv/bin/python -u -m src.data.jsonl_to_parquet \
  --input "$JSONL" \
  --outdir datasets/raw/off_jsonl_chunks \
  --chunk-size 100000 \
  2>&1 | tee -a "$LOG"

echo "=== $(date +%H:%M:%S) DONE ===" | tee -a "$LOG"
ls -la datasets/raw/off_jsonl_chunks/ | tail -5 | tee -a "$LOG"
