#!/usr/bin/env bash
# Rerun only the 72 failed slots from the validated-100 two-round primary grid.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

PLAN="${PLAN:-configs/primary_robustness_failed_slots_20260805.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-out/primary_robustness_recovery_20260810}"
LOG_DIR="${LOG_DIR:-logs/primary_robustness_recovery_20260810}"
GENERATED_CONFIG_DIR="$LOG_DIR/generated_configs"
GPUS="${GPUS:-0}"
MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-1}"
MIN_GPU_FREE_GB="${MIN_GPU_FREE_GB:-4}"
MODEL_TIMEOUT_SECONDS="${MODEL_TIMEOUT_SECONDS:-43200}"
MAX_RETRIES="${MAX_RETRIES:-3}"
OOM_RETRIES="${OOM_RETRIES:-3}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-300}"
CACHE_EVICT_FINISHED="${CACHE_EVICT_FINISHED:-success}"
TRY_VLLM="${TRY_VLLM:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$GENERATED_CONFIG_DIR"

LOCK_PATH="$LOG_DIR/recovery.lock"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "Another recovery process already holds $LOCK_PATH" >&2
  exit 1
fi

if [[ ! -f "$PLAN" ]]; then
  echo "Missing recovery plan: $PLAN" >&2
  exit 1
fi
if [[ ! -f data/rand/rand_dataset_chinese.json ]]; then
  echo "Missing prompt dataset: data/rand/rand_dataset_chinese.json" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/build_primary_recovery_configs.py \
  --plan "$PLAN" \
  --output-dir "$GENERATED_CONFIG_DIR" \
  --expected-slots 72

GLOBAL_STATUS="$OUTPUT_DIR/dna_global_status.jsonl"
GLOBAL_STATE="$OUTPUT_DIR/dna_global_state.json"
MASTER_LOG="$LOG_DIR/recovery_master.log"

run_phase() {
  local cell="$1"
  local temperature="$2"
  local top_p="$3"
  local repeat="$4"
  local generation_seed="$5"
  local task_count="$6"
  local config_path="$7"
  local mode="$8"
  local status_path="$OUTPUT_DIR/primary_recovery_${cell}/status.jsonl"
  local cell_log="$LOG_DIR/primary_recovery_${cell}_${mode}.log"
  local command=(
    "$PYTHON_BIN" scripts/run_hf_dna_pipeline.py
    --providers huggingface
    --hf-jsonl "$config_path"
    --dataset rand_chinese
    --max-samples 100
    --output-dir "$OUTPUT_DIR"
    --output-suffix "_${cell}"
    --run-name "primary_recovery_${cell}"
    --global-status-journal "$GLOBAL_STATUS"
    --global-state "$GLOBAL_STATE"
    --temperature "$temperature"
    --top-p "$top_p"
    --generation-seed "$generation_seed"
    --ignore-response-cache
    --gpus "$GPUS"
    --max-concurrent-gpus "$MAX_CONCURRENT_GPUS"
    --min-gpu-free-gb "$MIN_GPU_FREE_GB"
    --gpu-poll-seconds 15
    --model-timeout-seconds "$MODEL_TIMEOUT_SECONDS"
    --max-retries "$MAX_RETRIES"
    --oom-retries "$OOM_RETRIES"
    --retry-delay-seconds "$RETRY_DELAY_SECONDS"
    --gpu-memory-per-billion-gb 2.2
    --gpu-memory-headroom-gb 2
    --cache-evict-finished "$CACHE_EVICT_FINISHED"
    --resume-mode "$mode"
  )

  if [[ -f "$status_path" ]]; then
    command+=(--resume-status "$status_path")
  fi
  if [[ "$temperature" == "0.0" ]]; then
    command+=(--no-do-sample)
  else
    command+=(--do-sample)
  fi
  if [[ "$TRY_VLLM" == "1" ]]; then
    command+=(--try-vllm)
  fi

  {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] cell=$cell tasks=$task_count mode=$mode"
    printf "command:"
    printf " %q" "${command[@]}"
    printf "\n"
  } | tee -a "$MASTER_LOG"

  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  if "${command[@]}" 2>&1 | tee "$cell_log"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] completed $cell mode=$mode" | tee -a "$MASTER_LOG"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] retained failures in $cell mode=$mode" | tee -a "$MASTER_LOG" >&2
  fi
}

tail -n +2 "$GENERATED_CONFIG_DIR/cells.tsv" |
while IFS=$'\t' read -r cell temperature top_p repeat generation_seed task_count config_path; do
  status_path="$OUTPUT_DIR/primary_recovery_${cell}/status.jsonl"
  if [[ -f "$status_path" ]]; then
    # Continue an interrupted cell without repeating successes, then retry its
    # terminal failures. A completed successful cell queues zero models.
    run_phase "$cell" "$temperature" "$top_p" "$repeat" \
      "$generation_seed" "$task_count" "$config_path" continue
    run_phase "$cell" "$temperature" "$top_p" "$repeat" \
      "$generation_seed" "$task_count" "$config_path" retry-failed
  else
    run_phase "$cell" "$temperature" "$top_p" "$repeat" \
      "$generation_seed" "$task_count" "$config_path" all
  fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] recovery pass complete" | tee -a "$MASTER_LOG"
echo "Recovery artifacts: $OUTPUT_DIR"
echo "Copy that directory back under this repository's out/ directory for auditing."
