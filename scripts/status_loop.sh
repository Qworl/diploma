#!/bin/bash
# Periodic status snapshot. Writes /tmp/phase2_status.txt every 90 sec.
OUT=/tmp/phase2_status.txt
cd "$(dirname "$0")/.."

while true; do
  SNAP=$(timeout 15 .venv/bin/python scripts/_status_snapshot.py 2>&1)
  {
    echo "=== $(date +%H:%M:%S) ==="
    echo "$SNAP"
  } > "$OUT"

  # Stop when no relevant background process is running
  if ! pgrep -fl "off_fetcher" >/dev/null 2>&1 \
     && ! pgrep -fl "direct_llm_v2 " >/dev/null 2>&1 \
     && ! pgrep -fl "train_hybrid" >/dev/null 2>&1 \
     && ! pgrep -fl "eval_v2_expanded" >/dev/null 2>&1 \
     && ! pgrep -fl "build_expanded_gold" >/dev/null 2>&1 \
     && ! pgrep -fl "gold_vs_silver" >/dev/null 2>&1; then
    echo "" >> "$OUT"
    echo "*** ALL BACKGROUND JOBS DONE ***" >> "$OUT"
    break
  fi
  sleep 90
done
