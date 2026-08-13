#!/usr/bin/env bash
# Generate only missing explicit-seed artifacts for the 10-model DistDNA pilot.
# Historical unknown-seed outputs are never used as evidence and never touched.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/venv/bin/python}"
COHORT="${COHORT:-configs/distdna_pilot_10models.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-out/distdna_four_seed_pilot_t07_p10_20260813}"
LOG_DIR="${LOG_DIR:-logs/distdna_four_seed_pilot_t07_p10_20260813}"
GPUS="${GPUS:-6,7}"
MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-2}"
SEEDS="${SEEDS:-42001 42002 42003 42004}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR/generated_configs" "$LOG_DIR/inventory"

LOCK_PATH="$LOG_DIR/four_seed_pilot.lock"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "Another DistDNA pilot process already holds $LOCK_PATH" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment: $PYTHON_BIN" >&2
  exit 1
fi

run_seed() {
  local seed="$1"
  local queue="$LOG_DIR/generated_configs/seed_${seed}_missing.jsonl"
  local inventory="$LOG_DIR/inventory/seed_${seed}.json"
  local suffix="_t07_p10_seed${seed}"
  local run_name="distdna_pilot_t07_p10_seed${seed}"
  local run_log="$LOG_DIR/${run_name}.log"

  "$PYTHON_BIN" scripts/build_distdna_pilot_seed_queue.py \
    --cohort "$COHORT" \
    --seed "$seed" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --missing-jsonl "$queue" \
    --inventory-json "$inventory"

  local missing_count
  missing_count="$(wc -l < "$queue")"
  if [[ "$missing_count" -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] seed=$seed already complete; skipped"
    return 0
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] seed=$seed starting $missing_count missing models on GPUs $GPUS"
  "$PYTHON_BIN" scripts/run_hf_dna_pipeline.py \
    --providers huggingface \
    --hf-jsonl "$queue" \
    --dataset rand_chinese \
    --max-samples 100 \
    --output-dir "$OUTPUT_DIR" \
    --output-suffix "$suffix" \
    --run-name "$run_name" \
    --global-status-journal "$OUTPUT_DIR/dna_global_status.jsonl" \
    --global-state "$OUTPUT_DIR/dna_global_state.json" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --generation-seed "$seed" \
    --ignore-response-cache \
    --try-vllm \
    --gpus "$GPUS" \
    --max-concurrent-gpus "$MAX_CONCURRENT_GPUS" \
    --min-gpu-free-gb 4 \
    --gpu-poll-seconds 15 \
    --model-timeout-seconds 43200 \
    --max-retries 3 \
    --oom-retries 3 \
    --retry-delay-seconds 300 \
    --gpu-memory-per-billion-gb 2.2 \
    --gpu-memory-headroom-gb 2 \
    --cache-evict-finished success \
    --resume-mode all \
    --do-sample 2>&1 | tee "$run_log"
}

echo "DistDNA four-seed pilot: T=$TEMPERATURE top_p=$TOP_P cohort=$COHORT"
echo "New explicit-seed output root: $OUTPUT_DIR"
echo "Historical output roots are read only for explicit-seed reuse."

for seed in $SEEDS; do
  run_seed "$seed"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DistDNA four-seed pilot generation complete"
