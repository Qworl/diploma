#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — AI Attributes thesis pipeline orchestrator
#
# Re-runs all main analysis pipelines in dependency order so that defense
# reviewers can clone + set up the venv + run one command to get all key
# numbers.
#
# Usage:
#   bash reproduce.sh [--quick] [--skip-llm] [--skip-train]
#
#   --quick       Only run stages 3-5 (assumes silver + models already exist).
#   --skip-llm    Skip LLM-gated stages (6 + LLM cold-start branch of 9).
#                 Default: no-LLM-only path.
#   --skip-train  Skip stage 2 (ML training) if model files already exist.
#
# Requirements:
#   - Python 3.14 venv in .venv/
#   - pip install -r requirements.txt (including sentence-transformers, xgboost)
#   - brew install libomp  (macOS, needed by xgboost)
#   - Optional: OPENROUTER_API_KEY in .env (only for stages 6 and LLM cold-start)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve project root (works regardless of CWD)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${ROOT}/.venv/bin/python"
PROCESSED="${ROOT}/datasets/processed"
FIGURES="${ROOT}/docs/thesis/figures"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
SKIP_LLM=1          # default: no LLM
SKIP_TRAIN=0
QUICK=0

for arg in "$@"; do
  case "$arg" in
    --skip-llm)   SKIP_LLM=1  ;;
    --skip-train) SKIP_TRAIN=1 ;;
    --quick)      QUICK=1      ;;
    *)
      echo "Unknown flag: $arg"
      echo "Usage: bash reproduce.sh [--quick] [--skip-llm] [--skip-train]"
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Source .env for OPENROUTER_API_KEY if present
# ---------------------------------------------------------------------------
if [[ -f "${ROOT}/.env" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${ROOT}/.env"
  set +a
  echo "[env] Loaded ${ROOT}/.env"
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run_stage() {
  local stage_num=$1; shift
  local name=$1; shift
  local cmd="$*"
  echo ""
  echo "=== STAGE ${stage_num}: ${name} ==="
  echo "[$(date +%H:%M:%S)] Running: OMP_NUM_THREADS=1 ${cmd}"
  OMP_NUM_THREADS=1 eval "${cmd}"
  echo "[$(date +%H:%M:%S)] Done."
}

check_output() {
  local stage_num=$1
  local name=$2
  local path=$3
  if [[ ! -e "${path}" ]]; then
    echo ""
    echo "ERROR: Stage ${stage_num} (${name}) — expected output not found:"
    echo "  ${path}"
    exit 1
  fi
  echo "[check] Stage ${stage_num} output OK: $(basename "${path}")"
}

skip_stage() {
  local stage_num=$1
  local name=$2
  local reason=$3
  echo ""
  echo "=== STAGE ${stage_num}: ${name} === [SKIPPED — ${reason}]"
}

# ---------------------------------------------------------------------------
# STAGE 1: Silver labelling
# ---------------------------------------------------------------------------
if [[ ${QUICK} -eq 0 ]]; then
  run_stage 1 "Silver labelling (pasta_stratified)" \
    "${PYTHON} -m src.data.label_silver --category pasta_stratified --no-backup"
  check_output 1 "Silver labelling" \
    "${PROCESSED}/pasta_stratified_silver_standard.parquet"
else
  skip_stage 1 "Silver labelling" "--quick mode"
fi

# ---------------------------------------------------------------------------
# STAGE 2: ML training
# ---------------------------------------------------------------------------
if [[ ${QUICK} -eq 0 ]]; then
  _model_check="${ROOT}/models/pasta_stratified_thresholds.pkl"
  if [[ ${SKIP_TRAIN} -eq 1 && -f "${_model_check}" ]]; then
    skip_stage 2 "ML training" "--skip-train and models exist"
  else
    run_stage 2 "ML training (pasta_stratified, ~5 min)" \
      "${PYTHON} -m src.pipeline.ml.train --category pasta_stratified"
    check_output 2 "ML training (thresholds)" \
      "${ROOT}/models/pasta_stratified_thresholds.pkl"
  fi
else
  skip_stage 2 "ML training" "--quick mode"
fi

# ---------------------------------------------------------------------------
# STAGE 3: Cascade evaluation on audited gold
# ---------------------------------------------------------------------------
run_stage 3 "Cascade vs audited gold" \
  "${PYTHON} -m src.eval.cascade_vs_audited_gold"
check_output 3 "Cascade vs audited gold" \
  "${PROCESSED}/cascade_vs_audited_gold_pasta.json"

# ---------------------------------------------------------------------------
# STAGE 4: Trek A1 — validator comparison (3 sub-steps)
# ---------------------------------------------------------------------------
run_stage "4a" "Trek A1 — validator_comparison (signals + inference)" \
  "${PYTHON} -m src.eval.validator_comparison"
check_output "4a" "validator_comparison" \
  "${PROCESSED}/validator_comparison_pasta.parquet"

run_stage "4b" "Trek A1 — validator_report (summary JSON)" \
  "${PYTHON} -m src.eval.validator_report"
check_output "4b" "validator_report" \
  "${PROCESSED}/validator_comparison_pasta_summary.json"

run_stage "4c" "Trek A1 — validator_pareto (pareto PNG)" \
  "${PYTHON} -m src.eval.validator_pareto"
check_output "4c" "validator_pareto" \
  "${PROCESSED}/validator_pareto_pasta.png"

# ---------------------------------------------------------------------------
# STAGE 5: Trek A2 — catalog completion, no-LLM (3 categories)
# ---------------------------------------------------------------------------
run_stage "5a" "Trek A2 — catalog completion (pasta, no-LLM, ~30 sec)" \
  "${PYTHON} -m src.eval.catalog_completion.run_pasta --no-llm"
check_output "5a" "run_pasta no-LLM" \
  "${PROCESSED}/catalog_completion_summary_pasta_no_llm.json"

run_stage "5b" "Trek A2 — catalog completion (cheeses, no-LLM, ~30 sec)" \
  "${PYTHON} -m src.eval.catalog_completion.run_cheeses --no-llm"
check_output "5b" "run_cheeses no-LLM" \
  "${PROCESSED}/catalog_completion_summary_cheeses_no_llm.json"

run_stage "5c" "Trek A2 — catalog completion (electronics, no-LLM, ~30 sec)" \
  "${PYTHON} -m src.eval.catalog_completion.run_electronics --no-llm"
check_output "5c" "run_electronics no-LLM" \
  "${PROCESSED}/catalog_completion_summary_electronics_no_llm.json"

# ---------------------------------------------------------------------------
# STAGE 6: Trek A2 — pasta with LLM (optional, ~15 min, needs API key)
# ---------------------------------------------------------------------------
if [[ ${SKIP_LLM} -eq 1 ]]; then
  skip_stage 6 "Trek A2 pasta with-LLM" "--skip-llm flag"
else
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo ""
    echo "WARNING: OPENROUTER_API_KEY not set — skipping stage 6 (pasta with-LLM)"
    skip_stage 6 "Trek A2 pasta with-LLM" "no OPENROUTER_API_KEY"
  else
    run_stage 6 "Trek A2 — catalog completion (pasta, with-LLM, ~15 min)" \
      "${PYTHON} -m src.eval.catalog_completion.run_pasta"
    check_output 6 "run_pasta with-LLM" \
      "${PROCESSED}/catalog_completion_summary_pasta_with_llm.json"
  fi
fi

# ---------------------------------------------------------------------------
# STAGE 7: Trek A2 — hypothesis tests + report
# ---------------------------------------------------------------------------
if [[ ${SKIP_LLM} -eq 1 ]]; then
  run_stage 7 "Trek A2 — hypothesis tests (no_llm config)" \
    "${PYTHON} -m src.eval.catalog_completion.hypotheses --config-tag no_llm"
else
  run_stage 7 "Trek A2 — hypothesis tests (with_llm config)" \
    "${PYTHON} -m src.eval.catalog_completion.hypotheses"
fi
check_output 7 "catalog_completion hypotheses" \
  "${PROCESSED}/catalog_completion_hypotheses.json"

# ---------------------------------------------------------------------------
# STAGE 8: Trek A2 — Pareto plot
# ---------------------------------------------------------------------------
mkdir -p "${FIGURES}"
run_stage 8 "Trek A2 — Pareto plot (PNG per category)" \
  "${PYTHON} -m src.eval.catalog_completion.pareto_plot"
check_output 8 "Trek A2 pareto plot (pasta)" \
  "${FIGURES}/trek_a2_pareto_pasta.png"

# ---------------------------------------------------------------------------
# STAGE 9: Cold-start simulation
# ---------------------------------------------------------------------------
if [[ ${SKIP_LLM} -eq 1 ]]; then
  run_stage 9 "Cold-start simulation (no-LLM, ~30 sec)" \
    "${PYTHON} -m src.eval.catalog_completion.cold_start_simulation --no-llm"
else
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    run_stage 9 "Cold-start simulation (no-LLM fallback — no API key)" \
      "${PYTHON} -m src.eval.catalog_completion.cold_start_simulation --no-llm"
  else
    run_stage 9 "Cold-start simulation (with-LLM, ~15 min)" \
      "${PYTHON} -m src.eval.catalog_completion.cold_start_simulation"
  fi
fi
check_output 9 "cold-start simulation" \
  "${PROCESSED}/cold_start_simulation_pasta.json"

# ---------------------------------------------------------------------------
# STAGE 10: Silver-fix validation (optional, ~30 sec)
# ---------------------------------------------------------------------------
run_stage 10 "Silver-fix / enrichment experiment (~30 sec)" \
  "${PYTHON} -m src.eval.catalog_completion.silver_enrichment_experiment"
check_output 10 "silver enrichment experiment" \
  "${PROCESSED}/silver_enrichment_experiment_pasta.json"

# ---------------------------------------------------------------------------
# STAGE 11: Trek E — cross-domain audit (chocolate + cheeses)
# ---------------------------------------------------------------------------
# Stage 11a: cascade vs audited gold for chocolate
if [[ -f "${ROOT}/datasets/manual_label/chocolate_gold_239.csv" ]]; then
  run_stage "11a" "Trek E — cascade vs audited gold (chocolate)" \
    "${PYTHON} -m src.eval.cascade_vs_audited_gold --domain chocolate"
  check_output "11a" "chocolate cascade-vs-gold" \
    "${PROCESSED}/cascade_vs_audited_gold_chocolate.json"
else
  skip_stage "11a" "Trek E chocolate cascade-vs-gold" \
    "chocolate_gold_239.csv missing — run sample_domain_gold first"
fi

# Stage 11b: cascade vs audited gold for cheeses
if [[ -f "${ROOT}/datasets/manual_label/cheeses_gold_239.csv" ]]; then
  run_stage "11b" "Trek E — cascade vs audited gold (cheeses)" \
    "${PYTHON} -m src.eval.cascade_vs_audited_gold --domain cheeses"
  check_output "11b" "cheeses cascade-vs-gold" \
    "${PROCESSED}/cascade_vs_audited_gold_cheeses.json"
else
  skip_stage "11b" "Trek E cheeses cascade-vs-gold" \
    "cheeses_gold_239.csv missing"
fi

# Stage 11c: cross-domain summary
run_stage "11c" "Trek E — cross-domain summary (3-row table)" \
  "${PYTHON} -m src.eval.cross_domain_summary"
check_output "11c" "cross_domain_summary" \
  "${PROCESSED}/cross_domain_audit_summary.json"

# ---------------------------------------------------------------------------
# STAGE 12: Test suite
# ---------------------------------------------------------------------------
run_stage 12 "Test suite (pytest, ~10 sec)" \
  "${ROOT}/.venv/bin/pytest ${ROOT}/tests/ -q \
    --ignore=${ROOT}/tests/test_cascade_validator.py \
    --tb=short"

# ---------------------------------------------------------------------------
# DONE — print headline numbers
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "All stages complete. Headline numbers:"
echo "========================================================================"

_jq() {
  "${PYTHON}" -c "import json,sys; d=json.load(open('$1')); print($2)" 2>/dev/null || echo "(parse error)"
}

echo ""
echo "--- Cascade vs audited gold (pasta) ---"
_jq "${PROCESSED}/cascade_vs_audited_gold_pasta.json" \
  "'  acc_on_audited   : ' + str(round(d['metrics']['all_audited']['overall']['acc_on_audited'], 4))"
_jq "${PROCESSED}/cascade_vs_audited_gold_pasta.json" \
  "'  coverage         : ' + str(round(d['metrics']['all_audited']['overall']['coverage'], 4))"

echo ""
echo "--- Catalog completion — pasta no-LLM ---"
_jq "${PROCESSED}/catalog_completion_summary_pasta_no_llm.json" \
  "'  coverage_gain_pp : ' + str(d.get('coverage_gain_pp', 'n/a'))"
_jq "${PROCESSED}/catalog_completion_summary_pasta_no_llm.json" \
  "'  recovery_accuracy: ' + str(d.get('recovery_accuracy', 'n/a'))"

echo ""
echo "--- Trek A2 hypotheses ---"
_jq "${PROCESSED}/catalog_completion_hypotheses.json" \
  "'  H1 (coverage gain)   : ' + str(d.get('H1',{}).get('decision','n/a'))"
_jq "${PROCESSED}/catalog_completion_hypotheses.json" \
  "'  H2 (recovery acc)    : ' + str(d.get('H2',{}).get('decision','n/a'))"
_jq "${PROCESSED}/catalog_completion_hypotheses.json" \
  "'  H3 (electronics ceil): ' + str(d.get('H3',{}).get('decision','n/a'))"

echo ""
echo "--- Validator comparison summary ---"
_jq "${PROCESSED}/validator_comparison_pasta_summary.json" \
  "'  n_cells  : ' + str(d.get('n_cells','n/a'))"
_jq "${PROCESSED}/validator_comparison_pasta_summary.json" \
  "'  n_errors : ' + str(d.get('n_errors','n/a'))"

echo ""
echo "--- Cold-start simulation ---"
_jq "${PROCESSED}/cold_start_simulation_pasta.json" \
  "'  result: ' + str({k: d[k] for k in list(d)[:4]})"

echo ""
echo "--- Cross-domain audit summary (Trek E §6.18) ---"
if [[ -f "${PROCESSED}/cross_domain_audit_summary.md" ]]; then
  sed 's/^/  /' "${PROCESSED}/cross_domain_audit_summary.md"
fi

echo ""
echo "Output files:"
echo "  ${PROCESSED}/cascade_vs_audited_gold_pasta.json"
echo "  ${PROCESSED}/cascade_vs_audited_gold_chocolate.json   (Trek E)"
echo "  ${PROCESSED}/cascade_vs_audited_gold_cheeses.json     (Trek E)"
echo "  ${PROCESSED}/cross_domain_audit_summary.{json,md}      (Trek E §6.18)"
echo "  ${PROCESSED}/catalog_completion_summary_pasta_no_llm.json"
echo "  ${PROCESSED}/catalog_completion_hypotheses.json"
echo "  ${PROCESSED}/validator_comparison_pasta_summary.json"
echo "  ${PROCESSED}/cold_start_simulation_pasta.json"
echo "  ${PROCESSED}/silver_enrichment_experiment_pasta.json"
echo "  ${FIGURES}/trek_a2_pareto_pasta.png"
echo ""
