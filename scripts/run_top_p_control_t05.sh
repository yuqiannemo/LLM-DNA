#!/usr/bin/env bash
# Controlled top-p experiment: hold temperature and every other pipeline option
# fixed while sweeping top_p from 0.0 through 1.0. Two repeats quantify sampling
# variation. All outputs, status files, manifests, and logs are isolated.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-out/top_p_control_t05_validated100_20260719}"

export OUTPUT_DIR="$EXPERIMENT_ROOT"
export LOG_DIR="${LOG_DIR:-$EXPERIMENT_ROOT/logs}"
export MANIFEST="${MANIFEST:-$LOG_DIR/sweep_manifest.tsv}"
export SWEEP_NAME="${SWEEP_NAME:-top_p_control_t05}"
export HF_JSONL="${HF_JSONL:-configs/rand_chinese_0p3b_7b_rerun.jsonl}"
export PROVIDERS="huggingface"
export DATASET="rand_chinese"

# Experimental controls.
export TEMPERATURES="0.5"
# top_p=0.0 and 0.1 outputs are retained from earlier runs. Continue the
# controlled sweep at 0.2; incomplete/failed models will be backfilled later.
export TOP_PS="0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
export REPEATS="${REPEATS:-1 2}"
export INCLUDE_T00="0"
export LIMIT="0"
export RESUME_MODE="all"

# Restrict this user's experiment to the currently free cards. The scheduler
# still waits if another process begins using one of these GPUs.
export GPUS="${GPUS:-}"
export MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-10}"
export MIN_GPU_FREE_GB="${MIN_GPU_FREE_GB:-4.0}"
export GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-15.0}"

# Resilient completion settings: four attempts total for transient failures.
export MODEL_TIMEOUT_SECONDS="${MODEL_TIMEOUT_SECONDS:-43200}"
export MAX_RETRIES="${MAX_RETRIES:-3}"
export OOM_RETRIES="${OOM_RETRIES:-3}"
export RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-300}"
export TRY_VLLM="${TRY_VLLM:-1}"
export CACHE_EVICT_FINISHED="${CACHE_EVICT_FINISHED:-success}"

exec bash scripts/run_dna_temperature_top_p_sweep.sh
