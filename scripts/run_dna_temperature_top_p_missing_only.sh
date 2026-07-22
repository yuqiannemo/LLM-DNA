#!/usr/bin/env bash
# Run only the missing temperature/top_p pairs for the current rand_chinese sweep.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MASTER_LOG_DIR="${MASTER_LOG_DIR:-logs/dna_sweep/rand_chinese_temp_top_p_sweep_missing_$(date +%Y%m%d-%H%M%S)}"
REPEATS="${REPEATS:-1 2}"
SWEEP_NAME="${SWEEP_NAME:-rand_chinese_temp_top_p_sweep}"

mkdir -p "$MASTER_LOG_DIR"

run_pair() {
  local temperature="$1"
  local top_p="$2"
  local t_code
  local p_code
  t_code="$(printf '%.1f' "$temperature" | awk '{printf "%02d", int($1 * 10 + 0.5)}')"
  p_code="$(printf '%.1f' "$top_p" | awk '{printf "%02d", int($1 * 10 + 0.5)}')"
  local pair_name="${SWEEP_NAME}_t${t_code}_p${p_code}"
  local pair_log_dir="$MASTER_LOG_DIR/$pair_name"

  mkdir -p "$pair_log_dir"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START pair=$temperature:$top_p log_dir=$pair_log_dir"

  env \
    TEMPERATURES="$temperature" \
    TOP_PS="$top_p" \
    REPEATS="$REPEATS" \
    LOG_DIR="$pair_log_dir" \
    SWEEP_NAME="$SWEEP_NAME" \
    bash scripts/run_dna_temperature_top_p_sweep.sh

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  pair=$temperature:$top_p"
}

run_pair 0.5 0.9
run_pair 0.5 1.0
run_pair 0.7 0.8
run_pair 0.7 1.0

echo "[$(date '+%Y-%m-%d %H:%M:%S')] all missing pairs complete"