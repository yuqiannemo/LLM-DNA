#!/usr/bin/env bash
# Resume p=0.6/r2, then continue the validated-100 top-p control through 1.0.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/venv/bin/python}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-out/top_p_control_t05_validated100_20260719}"
CANONICAL_MANIFEST="$EXPERIMENT_ROOT/top_p_control_t05_t05_p06_r2/manifest.json"
HF_JSONL="${HF_JSONL:-configs/rand_chinese_validated100_latest.jsonl}"
GENERATION_SEED="${GENERATION_SEED:-42}"
GPUS="${GPUS:-0,1,2,4}"
MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-4}"
DRY_RUN="${DRY_RUN:-0}"

"$PYTHON_BIN" scripts/build_validated100_config.py \
  --manifest "$CANONICAL_MANIFEST" \
  --output "$HF_JSONL"

run_cell() {
  local top_p="$1"
  local p_code="$2"
  local repeat="$3"
  local suffix="_t05_p${p_code}_r${repeat}"
  local run_name="top_p_control_t05${suffix}"
  local run_dir="$EXPERIMENT_ROOT/$run_name"
  local status_path="$run_dir/status.jsonl"
  local resume_mode="all"

  local command=(
    "$PYTHON_BIN" scripts/run_hf_dna_pipeline.py
    --providers huggingface
    --hf-jsonl "$HF_JSONL"
    --dataset rand_chinese
    --max-samples 100
    --output-dir "$EXPERIMENT_ROOT"
    --output-suffix "$suffix"
    --run-name "$run_name"
    --global-status-journal "$EXPERIMENT_ROOT/dna_global_status.jsonl"
    --global-state "$EXPERIMENT_ROOT/dna_global_state.json"
    --temperature 0.5
    --top-p "$top_p"
    --generation-seed "$GENERATION_SEED"
    --do-sample
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
  )

  if [[ -f "$status_path" ]]; then
    resume_mode="continue"
    command+=(--resume-status "$status_path")
  fi
  command+=(--resume-mode "$resume_mode")

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $run_name mode=$resume_mode seed=$GENERATION_SEED"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "command:"
    printf " %q" "${command[@]}"
    printf "\n"
    return
  fi
  "${command[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  $run_name"
}

# Preserve the 24 completed p=0.6/r2 models and enqueue its remaining 76.
run_cell "0.6" "06" "2"

# Continue from low to high. Both rounds intentionally use seed 42 so these
# cells remain compatible with the explicit-seed portion of the control sweep.
for cell in "0.7 07" "0.8 08" "0.9 09" "1.0 10"; do
  read -r top_p p_code <<< "$cell"
  run_cell "$top_p" "$p_code" "1"
  run_cell "$top_p" "$p_code" "2"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] validated top-p control complete"
