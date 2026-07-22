#!/usr/bin/env bash
# Sequentially run LLM-DNA temperature/top_p sweeps.
#
# Launch once with nohup, for example:
#   nohup bash scripts/run_dna_temperature_top_p_sweep.sh > logs/dna_sweep/master_$(date +%Y%m%d-%H%M%S).log 2>&1 &
#
# The script runs one setting at a time. Each setting gets a clear output suffix,
# run name, child log, and manifest row.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

HF_JSONL="${HF_JSONL:-configs/rand_chinese_0p3b_7b_rerun.jsonl}"
DATASET="${DATASET:-rand_chinese}"
OUTPUT_DIR="${OUTPUT_DIR:-out}"
PROVIDERS="${PROVIDERS:-huggingface}"
GLOBAL_STATUS_JOURNAL="${GLOBAL_STATUS_JOURNAL:-$OUTPUT_DIR/dna_global_status.jsonl}"
GLOBAL_STATE="${GLOBAL_STATE:-$OUTPUT_DIR/dna_global_state.json}"

# Default sweep. Override with env vars, e.g.:
#   TEMPERATURES="0.2 0.3" TOP_PS="0.9" REPEATS="1 2"
TEMPERATURES="${TEMPERATURES:-0.0 0.1 0.2 0.3 0.5 0.7 0.9}"
TOP_PS="${TOP_PS:-0.8 0.85 0.9 0.95 1.0}"
REPEATS="${REPEATS:-1 2}"
GENERATION_SEED_BASE="${GENERATION_SEED_BASE:-42000}"

MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-10}"
MIN_GPU_FREE_GB="${MIN_GPU_FREE_GB:-4.0}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-15.0}"
GPU_MEMORY_PER_BILLION_GB="${GPU_MEMORY_PER_BILLION_GB:-2.2}"
GPU_MEMORY_HEADROOM_GB="${GPU_MEMORY_HEADROOM_GB:-2.0}"
MODEL_TIMEOUT_SECONDS="${MODEL_TIMEOUT_SECONDS:-43200}"
MAX_RETRIES="${MAX_RETRIES:-3}"
OOM_RETRIES="${OOM_RETRIES:-3}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-300}"
GPUS="${GPUS:-}"
TRY_VLLM="${TRY_VLLM:-1}"
RESUME_MODE="${RESUME_MODE:-all}"
CACHE_EVICT_FINISHED="${CACHE_EVICT_FINISHED:-success}"

# Set LIMIT=50 for a cheaper pilot. LIMIT=0 means no limit.
LIMIT="${LIMIT:-0}"

# Use 1 to include deterministic baselines named _t00_p10_r1/_t00_p10_r2.
# Existing _2_run/_3_run already cover deterministic baselines, so default is off.
INCLUDE_T00="${INCLUDE_T00:-0}"

SWEEP_NAME="${SWEEP_NAME:-rand_chinese_temp_top_p_sweep}"
LOG_DIR="${LOG_DIR:-logs/dna_sweep/${SWEEP_NAME}_$(date +%Y%m%d-%H%M%S)}"
MANIFEST="${MANIFEST:-$LOG_DIR/sweep_manifest.tsv}"

mkdir -p "$LOG_DIR"

format_decimal() {
  local value="$1"
  "$PYTHON_BIN" - "$value" <<'PY'
import sys
value = float(sys.argv[1])
print(f"{int(round(value * 10)):02d}")
PY
}

write_manifest_header() {
  printf "timestamp\tsuffix\trun_name\ttemperature\ttop_p\trepeat\tstatus\tlog_path\n" > "$MANIFEST"
}

append_manifest() {
  local suffix="$1"
  local run_name="$2"
  local temperature="$3"
  local top_p="$4"
  local repeat="$5"
  local status="$6"
  local log_path="$7"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$suffix" "$run_name" "$temperature" "$top_p" "$repeat" "$status" "$log_path" >> "$MANIFEST"
}

run_one() {
  local temperature="$1"
  local top_p="$2"
  local repeat="$3"
  local t_code
  local p_code
  t_code="$(format_decimal "$temperature")"
  p_code="$(format_decimal "$top_p")"

  local suffix="_t${t_code}_p${p_code}_r${repeat}"
  local generation_seed=$((GENERATION_SEED_BASE + repeat))
  local run_name="${SWEEP_NAME}${suffix}"
  local log_path="$LOG_DIR/${run_name}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START suffix=$suffix temperature=$temperature top_p=$top_p repeat=$repeat"
  append_manifest "$suffix" "$run_name" "$temperature" "$top_p" "$repeat" "started" "$log_path"

  local command=(
    "$PYTHON_BIN" scripts/run_hf_dna_pipeline.py
    --providers "$PROVIDERS"
    --hf-jsonl "$HF_JSONL"
    --dataset "$DATASET"
    --output-dir "$OUTPUT_DIR"
    --global-status-journal "$GLOBAL_STATUS_JOURNAL"
    --global-state "$GLOBAL_STATE"
    --output-suffix "$suffix"
    --ignore-response-cache
    --resume-mode "$RESUME_MODE"
    --max-concurrent-gpus "$MAX_CONCURRENT_GPUS"
    --min-gpu-free-gb "$MIN_GPU_FREE_GB"
    --gpu-poll-seconds "$GPU_POLL_SECONDS"
    --gpu-memory-per-billion-gb "$GPU_MEMORY_PER_BILLION_GB"
    --gpu-memory-headroom-gb "$GPU_MEMORY_HEADROOM_GB"
    --model-timeout-seconds "$MODEL_TIMEOUT_SECONDS"
    --max-retries "$MAX_RETRIES"
    --oom-retries "$OOM_RETRIES"
    --retry-delay-seconds "$RETRY_DELAY_SECONDS"
    --cache-evict-finished "$CACHE_EVICT_FINISHED"
    --temperature "$temperature"
    --top-p "$top_p"
    --generation-seed "$generation_seed"
    --run-name "$run_name"
  )

  # vLLM requires top_p > 0. Transformers supports top_p=0 by retaining only
  # the highest-probability token, so use the Transformers backend for p=0.
  if [[ "$TRY_VLLM" == "1" ]] && "$PYTHON_BIN" - "$top_p" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0.0 else 1)
PY
  then
    command+=(--try-vllm)
  fi
  if [[ -n "$GPUS" ]]; then
    command+=(--gpus "$GPUS")
  fi
  if [[ "$LIMIT" != "0" ]]; then
    command+=(--limit "$LIMIT")
  fi

  # temperature=0 should stay deterministic; otherwise enable sampling.
  if "$PYTHON_BIN" - "$temperature" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0.0 else 1)
PY
  then
    command+=(--do-sample)
  else
    command+=(--no-do-sample)
  fi

  {
    echo "cwd=$ROOT"
    echo "python=$PYTHON_BIN"
    echo "suffix=$suffix"
    echo "run_name=$run_name"
    echo "temperature=$temperature"
    echo "top_p=$top_p"
    echo "repeat=$repeat"
    echo "generation_seed=$generation_seed"
    printf "command:"
    printf " %q" "${command[@]}"
    printf "\n\n"
  } > "$log_path"

  local status="success"
  if "${command[@]}" >> "$log_path" 2>&1; then
    status="success"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  suffix=$suffix"
  else
    status="failed"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL  suffix=$suffix log=$log_path"
  fi
  append_manifest "$suffix" "$run_name" "$temperature" "$top_p" "$repeat" "$status" "$log_path"
}

write_manifest_header

echo "Sweep manifest: $MANIFEST"
echo "Logs: $LOG_DIR"
echo "HF_JSONL=$HF_JSONL"
echo "TEMPERATURES=$TEMPERATURES"
echo "TOP_PS=$TOP_PS"
echo "REPEATS=$REPEATS"
echo "LIMIT=$LIMIT"

if [[ "$INCLUDE_T00" == "1" ]]; then
  for repeat in $REPEATS; do
    run_one "0.0" "1.0" "$repeat"
  done
fi

for temperature in $TEMPERATURES; do
  # T=0 is a single deterministic condition: top-p is inactive when sampling
  # is disabled and must not be counted as multiple factorial cells.
  if "$PYTHON_BIN" - "$temperature" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) == 0.0 else 1)
PY
  then
    continue
  fi
  for top_p in $TOP_PS; do
    for repeat in $REPEATS; do
      run_one "$temperature" "$top_p" "$repeat"
    done
  done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] sweep complete"
