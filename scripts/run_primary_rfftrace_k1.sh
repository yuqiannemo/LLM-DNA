#!/usr/bin/env bash
set -euo pipefail

# Formal RFFTrace/DistDNA K=1 diagnostic on the validated-100 robustness grid.
# K=1 is an implementation and point-kernel comparison only; it is not evidence
# for a multi-sample advantage. A valid K=2 run needs independent rounds 1/3
# for reference and 2/4 for query.

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENCODER_DEVICE="${ENCODER_DEVICE:-cpu}"
OUTPUT_DIR="${OUTPUT_DIR:-out/rfftrace_primary_k1}"
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-out/response_embedding_cache}"
MINIMUM_MODELS="${MINIMUM_MODELS:-80}"
RFF_DIMENSION="${RFF_DIMENSION:-1024}"
COMPACT_DIMENSION="${COMPACT_DIMENSION:-128}"
SEED="${SEED:-42}"

normalize_args=()
if [[ "${NORMALIZE_EMBEDDINGS:-0}" == "1" ]]; then
  normalize_args+=(--normalize-embeddings)
fi

"$PYTHON_BIN" scripts/run_rfftrace_experiment.py \
  --data-dir out/primary_robustness_validated100_20260730/rand_chinese \
  --data-dir out/primary_robustness_recovery_20260810 \
  --output-dir "$OUTPUT_DIR" \
  --cohort-file configs/rand_chinese_validated100_latest.jsonl \
  --reference-setting 0.2:0.8 \
  --query-setting 0.0:1.0 \
  --query-setting 0.2:0.8 \
  --query-setting 0.2:0.9 \
  --query-setting 0.2:1.0 \
  --query-setting 0.3:0.8 \
  --query-setting 0.3:0.9 \
  --query-setting 0.3:1.0 \
  --query-setting 0.5:0.8 \
  --query-setting 0.5:0.9 \
  --query-setting 0.5:1.0 \
  --query-setting 0.7:0.8 \
  --query-setting 0.7:0.9 \
  --query-setting 0.7:1.0 \
  --reference-repeats 1 \
  --query-repeats 2 \
  --samples-per-side 1 \
  --minimum-models "$MINIMUM_MODELS" \
  --encoder all-mpnet-base-v2 \
  --encoder-device "$ENCODER_DEVICE" \
  --embedding-cache-dir "$EMBEDDING_CACHE_DIR" \
  --rff-dimension "$RFF_DIMENSION" \
  --compact-dimension "$COMPACT_DIMENSION" \
  --seed "$SEED" \
  "${normalize_args[@]}"
