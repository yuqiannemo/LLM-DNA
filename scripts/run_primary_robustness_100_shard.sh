#!/usr/bin/env bash
# Run one hardware shard of the fixed 100-model primary experiment.
#
# Both shards intentionally share OUTPUT_DIR so the completed dataset is
# analysis-ready. Status/state journals and scheduler run names are isolated to
# avoid concurrent writers clobbering each other.

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

# These values are deliberately forced so stale exports from the superseded
# 300-model experiment cannot redirect a shard into the old cohort or outputs.
export HF_JSONL
export OUTPUT_DIR="out/primary_robustness_100"
export SWEEP_NAME="primary_robustness_100_${SHARD}"
export LOG_DIR="${LOG_DIR:-logs/primary_robustness_100_${SHARD}_$(date +%Y%m%d-%H%M%S)}"
export GLOBAL_STATUS_JOURNAL="$OUTPUT_DIR/dna_global_status_${SHARD}.jsonl"
export GLOBAL_STATE="$OUTPUT_DIR/dna_global_state_${SHARD}.json"

exec bash scripts/run_primary_robustness_experiment.sh
