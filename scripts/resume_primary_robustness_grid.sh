#!/usr/bin/env bash
# Resume the revised validated-100 primary grid without regenerating models
# already marked successful in the canonical per-cell status journals.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/venv/bin/python}"
HF_JSONL="${HF_JSONL:-configs/rand_chinese_validated100_latest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-out/primary_robustness_validated100_20260730}"
LOG_DIR="${LOG_DIR:-logs/primary_robustness_validated100_20260730}"
GPUS="${GPUS:-0,1,2,4}"
MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-4}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

LOCK_PATH="$LOG_DIR/resume_grid.lock"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "Another primary-grid resume process already holds $LOCK_PATH" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$HF_JSONL" ]]; then
  echo "Missing validated cohort: $HF_JSONL" >&2
  exit 1
fi
if [[ "$(wc -l < "$HF_JSONL")" -ne 100 ]]; then
  echo "Validated cohort must contain exactly 100 rows: $HF_JSONL" >&2
  exit 1
fi

cell_code() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
print(f"{int(round(float(sys.argv[1]) * 10)):02d}")
PY
}

run_phase() {
  local temperature="$1"
  local top_p="$2"
  local repeat="$3"
  local generation_seed="$4"
  local run_name="$5"
  local resume_status="$6"
  local resume_mode="$7"
  local t_code
  local p_code
  t_code="$(cell_code "$temperature")"
  p_code="$(cell_code "$top_p")"
  local suffix="_t${t_code}_p${p_code}_r${repeat}"

  local command=(
    "$PYTHON_BIN" scripts/run_hf_dna_pipeline.py
    --providers huggingface
    --hf-jsonl "$HF_JSONL"
    --dataset rand_chinese
    --max-samples 100
    --output-dir "$OUTPUT_DIR"
    --output-suffix "$suffix"
    --run-name "$run_name"
    --global-status-journal "$OUTPUT_DIR/dna_global_status.jsonl"
    --global-state "$OUTPUT_DIR/dna_global_state.json"
    --temperature "$temperature"
    --top-p "$top_p"
    --generation-seed "$generation_seed"
    --ignore-response-cache
    --try-vllm
    --gpus "$GPUS"
    --max-concurrent-gpus "$MAX_CONCURRENT_GPUS"
    --min-gpu-free-gb 4
    --gpu-poll-seconds 15
    --model-timeout-seconds 43200
    --max-retries 3
    --oom-retries 3
    --retry-delay-seconds 300
    --gpu-memory-per-billion-gb 2.2
    --gpu-memory-headroom-gb 2
    --cache-evict-finished success
    --resume-mode "$resume_mode"
  )

  if [[ "$temperature" == "0.0" ]]; then
    command+=(--no-do-sample)
  else
    command+=(--do-sample)
  fi
  if [[ -n "$resume_status" ]]; then
    command+=(--resume-status "$resume_status")
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] phase=$resume_mode cell=($temperature,$top_p,r$repeat) seed=$generation_seed"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "command:"
    printf " %q" "${command[@]}"
    printf "\n"
    return 0
  fi
  if "${command[@]}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] phase complete: $run_name $resume_mode"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] phase retained failures: $run_name $resume_mode" >&2
  fi
}

run_cell() {
  local temperature="$1"
  local top_p="$2"
  local repeat="$3"
  local generation_seed="$4"
  local existing_run_dir="${5:-}"
  local t_code
  local p_code
  t_code="$(cell_code "$temperature")"
  p_code="$(cell_code "$top_p")"
  local run_name="primary_grid_backfill_20260730_t${t_code}_p${p_code}_r${repeat}"
  local resume_status=""

  if [[ -n "$existing_run_dir" && -f "$existing_run_dir/status.jsonl" ]]; then
    resume_status="$existing_run_dir/status.jsonl"
    # First add cohort models that have no terminal status, then revisit all
    # failed records. Successful models are never enqueued by either phase.
    run_phase "$temperature" "$top_p" "$repeat" "$generation_seed" \
      "$run_name" "$resume_status" "continue"
    run_phase "$temperature" "$top_p" "$repeat" "$generation_seed" \
      "$run_name" "$resume_status" "retry-failed"
  else
    # A genuinely absent repeat has no successes to preserve.
    run_phase "$temperature" "$top_p" "$repeat" "$generation_seed" \
      "$run_name" "" "all"
  fi
}

echo "Validated cohort: $HF_JSONL"
echo "Pinned GPUs: $GPUS"
echo "Output root for new artifacts: $OUTPUT_DIR"
echo "Priority: deterministic control, T=0.7, T=0.5, then T=0.2/0.3 backfills"

# Deterministic control.
run_cell 0.0 1.0 1 42001 out/rand_chinese_temp_top_p_sweep_t00_p10_r1
run_cell 0.0 1.0 2 42002

# Largest stochastic gaps.
run_cell 0.7 0.8 1 42001 out/rand_chinese_temp_top_p_sweep_t07_p08_r1
run_cell 0.7 0.8 2 42002
run_cell 0.7 0.9 1 42001 out/rand_chinese_0p3b_7b_temp07
run_cell 0.7 0.9 2 42002
run_cell 0.7 1.0 1 42001 out/rand_chinese_temp_top_p_sweep_t07_p10_r1
run_cell 0.7 1.0 2 42002

# T=0.5 remains compatible with the validated top-p control (seed 42). The
# separate independence audit decides whether a distinct-seed repeat is needed.
run_cell 0.5 0.8 1 42 out/rand_chinese_temp_top_p_sweep_resume_t05_t07_t05_p08_r1
run_cell 0.5 0.8 2 42
run_cell 0.5 0.9 1 42 out/rand_chinese_temp_top_p_sweep_t05_p09_r1
run_cell 0.5 0.9 2 42 out/rand_chinese_temp_top_p_sweep_t05_p09_r2
run_cell 0.5 1.0 1 42 out/rand_chinese_temp_top_p_sweep_t05_p10_r1
run_cell 0.5 1.0 2 42

# Smaller historical gaps.
run_cell 0.2 0.8 1 42001 out/rand_chinese_temp_top_p_sweep_t02_p08_r1
run_cell 0.2 0.8 2 42002 out/rand_chinese_temp_top_p_sweep_t02_p08_r2
run_cell 0.2 0.9 1 42001 out/rand_chinese_temp_top_p_sweep_t02_p09_r1
run_cell 0.2 0.9 2 42002 out/rand_chinese_temp_top_p_sweep_t02_p09_r2
run_cell 0.2 1.0 1 42001 out/rand_chinese_temp_top_p_sweep_t02_p10_r1
run_cell 0.2 1.0 2 42002 out/rand_chinese_temp_top_p_sweep_t02_p10_r2
run_cell 0.3 0.8 1 42001 out/rand_chinese_temp_top_p_sweep_t03_p08_r1
run_cell 0.3 0.8 2 42002 out/rand_chinese_temp_top_p_sweep_t03_p08_r2
run_cell 0.3 0.9 1 42001 out/rand_chinese_temp_top_p_sweep_t03_p09_r1
run_cell 0.3 0.9 2 42002 out/rand_chinese_temp_top_p_sweep_t03_p09_r2
run_cell 0.3 1.0 1 42001 out/rand_chinese_temp_top_p_sweep_t03_p10_r1
run_cell 0.3 1.0 2 42002 out/rand_chinese_temp_top_p_sweep_resume_t03p10r2_t03_p10_r2

echo "[$(date '+%Y-%m-%d %H:%M:%S')] primary grid resume sequence complete"
