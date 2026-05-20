#!/bin/bash
# Run direct LLM eval on brand-disjoint test (239 codes/cat) for 3 LLMs.
# Total expected cost: ~$25-30.

set -e
cd /Users/miafrolov/Desktop/stuff/ai_attributes
source .venv/bin/activate
# shellcheck disable=SC2046
export $(grep -v '^#' .env | xargs)

LOG=/tmp/llm_eval_v3.log
echo "=== LLM eval v3 (brand-disjoint test) starting $(date) ===" > "$LOG"

for MODEL in "google/gemini-2.5-flash" "openai/gpt-4o" "anthropic/claude-sonnet-4.5"; do
    MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||;s|-2.5||;s|-4.5||;s|/.*||;s|-|_|g')
    case "$MODEL" in
        *gemini*) MODEL_SHORT=gemini25flash_v3;;
        *gpt-4o*) MODEL_SHORT=gpt4o_v3;;
        *sonnet*) MODEL_SHORT=sonnet45_v3;;
    esac
    for CAT in pasta chocolate cheeses; do
        OUT="datasets/processed/direct_llm_eval_${CAT}_stratified_${MODEL_SHORT}.parquet"
        if [ -f "$OUT" ]; then
            echo "[$(date +%H:%M:%S)] $MODEL $CAT — output exists, will resume from there" >> "$LOG"
        fi
        echo "[$(date +%H:%M:%S)] === $MODEL × $CAT → $OUT ===" >> "$LOG"
        OMP_NUM_THREADS=1 python -u -m src.eval.direct_llm_v2 \
            --gold-codes "datasets/processed/${CAT}_test_codes.csv" \
            --domain "$CAT" \
            --model "$MODEL" \
            --context-mode partner_input \
            --out "$OUT" \
            --max-cost-usd 6.0 \
            --sleep 0.05 >> "$LOG" 2>&1 \
            || echo "[$(date +%H:%M:%S)] FAILED $MODEL $CAT, continuing" >> "$LOG"
    done
done

echo "=== LLM eval v3 DONE $(date) ===" >> "$LOG"
