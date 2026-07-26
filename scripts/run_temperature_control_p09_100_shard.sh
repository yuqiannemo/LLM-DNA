#!/usr/bin/env bash
# Fixed-top-p temperature experiment over the exact primary 100-model cohort.
# Temperatures run sequentially from low to high; each has four generation seeds.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {rtx3090|a100}" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SHARD="$1"
case "$SHARD" in
  rtx3090)
    HF_JSONL="configs/rand_chinese_stratified_100_rtx3090.jsonl"
    EXPECTED_MODELS=77
    export MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-4}"
    ;;
  a100)
    HF_JSONL="configs/rand_chinese_stratified_100_a100.jsonl"
    EXPECTED_MODELS=23
    export MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-2}"
    ;;
  *)
    echo "Unknown shard '$SHARD'; expected rtx3090 or a100." >&2
    exit 2
    ;;
esac

if [[ ! -f "$HF_JSONL" ]]; then
  echo "Missing fixed shard file: $HF_JSONL" >&2
  exit 1
fi
ACTUAL_MODELS="$(wc -l < "$HF_JSONL")"
if [[ "$ACTUAL_MODELS" -ne "$EXPECTED_MODELS" ]]; then
  echo "Invalid $SHARD shard: expected $EXPECTED_MODELS rows, found $ACTUAL_MODELS." >&2
  exit 1
fi

EXPERIMENT_NAME="temperature_control_p09_100"
export HF_JSONL
export OUTPUT_DIR="out/$EXPERIMENT_NAME"
export DATASET="rand_chinese"
export SWEEP_NAME="${EXPERIMENT_NAME}_${SHARD}"
export LOG_DIR="${LOG_DIR:-logs/${EXPERIMENT_NAME}_${SHARD}_$(date +%Y%m%d-%H%M%S)}"
export GLOBAL_STATUS_JOURNAL="$OUTPUT_DIR/dna_global_status_${SHARD}.jsonl"
export GLOBAL_STATE="$OUTPUT_DIR/dna_global_state_${SHARD}.json"

# Experimental variables and controls. The sweep runner preserves this order.
export TEMPERATURES="0.2 0.3 0.5 0.7"
export TOP_PS="0.9"
export REPEATS="${REPEATS:-1 2 3 4}"
export GENERATION_SEED_BASE="${GENERATION_SEED_BASE:-42000}"
export INCLUDE_T00="0"

export GPUS="${GPUS:-}"
export LIMIT="0"
export RESUME_MODE="all"
export TRY_VLLM="${TRY_VLLM:-1}"
export CACHE_EVICT_FINISHED="${CACHE_EVICT_FINISHED:-success}"

echo "Experiment: $EXPERIMENT_NAME"
echo "Shard: $SHARD ($ACTUAL_MODELS models)"
echo "Fixed top_p: $TOP_PS"
echo "Temperatures (execution order): $TEMPERATURES"
echo "Repeats: $REPEATS"
echo "Generation seeds: 42001 42002 42003 42004"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"

exec bash scripts/run_dna_temperature_top_p_sweep.sh
