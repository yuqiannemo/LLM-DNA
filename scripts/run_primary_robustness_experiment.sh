#!/usr/bin/env bash
# Primary exact-model robustness dataset: 100 models, a complete factorial
# decoding grid, and four independent generation seeds. T=0 is emitted once.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_JSONL="${HF_JSONL:-configs/rand_chinese_stratified_100.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-out/primary_robustness_100}"
export DATASET="${DATASET:-rand_chinese}"
export SWEEP_NAME="${SWEEP_NAME:-primary_robustness_100}"
export LOG_DIR="${LOG_DIR:-logs/primary_robustness_100_$(date +%Y%m%d-%H%M%S)}"

export TEMPERATURES="0.2 0.3 0.5 0.7"
export TOP_PS="0.8 0.9 1.0"
export REPEATS="${REPEATS:-1 2 3 4}"
export INCLUDE_T00="1"
export GENERATION_SEED_BASE="${GENERATION_SEED_BASE:-42000}"

# Empty GPUS means all visible cards are dynamically scheduled.  The worker
# cap can be raised without touching Python; 10 matches the available cluster.
export GPUS="${GPUS:-}"
export MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-10}"
export LIMIT="0"
export RESUME_MODE="all"
export TRY_VLLM="${TRY_VLLM:-1}"
export CACHE_EVICT_FINISHED="${CACHE_EVICT_FINISHED:-success}"

exec bash scripts/run_dna_temperature_top_p_sweep.sh
