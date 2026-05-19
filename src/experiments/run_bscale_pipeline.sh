#!/usr/bin/env bash
# Full B-scale gold annotation pipeline:
# Step 3: Annotate with gpt-5.5 (parallel across cats)
# Step 4: Build expanded gold + retrain hybrid
# Step 5: Compare numbers
# Run after OFF fetcher is done.

set -euo pipefail

WDIR="/Users/miafrolov/Desktop/stuff/ai_attributes/.worktrees/phase2-recomputes-on-v2-gold"
VENV="${WDIR}/.venv/bin/python"
LOG_DIR="/tmp/bscale_logs"
mkdir -p "$LOG_DIR"
mkdir -p "${WDIR}/datasets/processed/gpt55_gold"

cd "$WDIR"

echo "=== Step 3: Annotate with gpt-5.5 OFF-grounded ==="
echo "Starting 3 parallel annotation processes..."

pids=()
for cat in pasta chocolate cheeses; do
    csv="datasets/manual_label/${cat}_b_scale_650.csv"
    out="datasets/processed/gpt55_gold/${cat}_gpt55_gold.parquet"
    log="${LOG_DIR}/bscale_${cat}.log"
    echo "  [${cat}] starting → ${out}"
    OMP_NUM_THREADS=1 "${VENV}" -m src.eval.direct_llm_v2 \
        --gold-codes "$csv" \
        --domain "$cat" \
        --model openai/gpt-5.5 \
        --context-mode off_grounded \
        --off-cache-dir datasets/manual_label/off_cache \
        --out "$out" \
        --max-cost-usd 32.0 > "$log" 2>&1 &
    pids+=($!)
    echo "  [${cat}] PID=${pids[-1]}"
done

echo ""
echo "Waiting for all 3 annotation processes..."
failed=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then
        echo "  PID $pid: OK"
    else
        echo "  PID $pid: FAILED (exit $?)"
        failed=$((failed+1))
    fi
done

if [[ $failed -gt 0 ]]; then
    echo "WARNING: $failed annotation process(es) failed. Proceeding with what we have."
fi

echo ""
echo "=== Step 4a: Build expanded gold parquet ==="
OMP_NUM_THREADS=1 "${VENV}" -m src.experiments.build_expanded_gold
echo "Done: consensus_gold_v2_expanded.parquet"

echo ""
echo "=== Step 4b: Retrain hybrid cascade on expanded gold ==="
OMP_NUM_THREADS=2 "${VENV}" -m src.experiments.train_hybrid_cascade \
    --cats pasta chocolate cheeses \
    --gold-path datasets/processed/consensus_gold_v2_expanded.parquet
echo "Done: hybrid models retrained"

echo ""
echo "=== Step 4c: Re-run headline with expanded hybrid ==="
OMP_NUM_THREADS=1 "${VENV}" -m src.eval.headline_v2_hybrid \
    --gold datasets/processed/consensus_gold_v2_expanded.parquet \
    --out datasets/processed/headline_results_off_grounded_hybrid_v2.parquet
echo "Done: headline_results_off_grounded_hybrid_v2.parquet"

echo ""
echo "=== Step 5: Compare versions ==="
OMP_NUM_THREADS=1 "${VENV}" -m src.experiments.compare_gold_versions

echo ""
echo "=== All done! ==="
