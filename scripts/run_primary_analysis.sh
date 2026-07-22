#!/usr/bin/env bash
# Analyze the completed primary sweep.  Repeats 1-2 are training/gallery data;
# repeats 3-4 are held-out queries.  The command refuses cohorts below 250.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-out/primary_robustness_300/rand_chinese}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-out/primary_robustness_300_analysis}"
MINIMUM_MODELS="${MINIMUM_MODELS:-250}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

settings=("0.2 0.8" "0.2 0.9" "0.2 1.0" "0.3 0.8" "0.3 0.9" "0.3 1.0" "0.5 0.8" "0.5 0.9" "0.5 1.0" "0.7 0.8" "0.7 0.9" "0.7 1.0")

code() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
print(f"{int(round(float(sys.argv[1]) * 10)):02d}")
PY
}

exact_args=(
  --dataset-dir "$DATA_DIR"
  --minimum-models "$MINIMUM_MODELS"
  --output-dir "$ANALYSIS_ROOT/exact"
  --temperature-bound 0.2 --temperature-bound 0.3
  --temperature-bound 0.5 --temperature-bound 0.7
)
tree_args=(
  --dataset-dir "$DATA_DIR"
  --reference-suffix _t00_p10_r1
  --minimum-models "$MINIMUM_MODELS"
  --output-dir "$ANALYSIS_ROOT/tree"
)

for repeat in 1 2; do
  exact_args+=(--train-run "t00_r${repeat}:_t00_p10_r${repeat}:0.0:1.0:${repeat}")
done
for repeat in 3 4; do
  exact_args+=(--test-run "t00_r${repeat}:_t00_p10_r${repeat}:0.0:1.0:${repeat}")
  tree_args+=(--run "t00_r${repeat}:_t00_p10_r${repeat}:0.0:1.0:${repeat}")
done

for setting in "${settings[@]}"; do
  read -r temperature top_p <<< "$setting"
  t_code="$(code "$temperature")"
  p_code="$(code "$top_p")"
  for repeat in 1 2; do
    exact_args+=(--train-run "t${t_code}_p${p_code}_r${repeat}:_t${t_code}_p${p_code}_r${repeat}:${temperature}:${top_p}:${repeat}")
  done
  for repeat in 3 4; do
    spec="t${t_code}_p${p_code}_r${repeat}:_t${t_code}_p${p_code}_r${repeat}:${temperature}:${top_p}:${repeat}"
    exact_args+=(--test-run "$spec")
    tree_args+=(--run "$spec")
  done
done

"$PYTHON_BIN" scripts/evaluate_exact_dna_classification.py "${exact_args[@]}"
"$PYTHON_BIN" scripts/analyze_tree_aware_errors.py "${tree_args[@]}"
