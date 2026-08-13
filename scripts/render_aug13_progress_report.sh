#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/docs/primary_robustness_experiment_full_updated_2026-08-13.tex"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/docs}"
TECTONIC_BIN="${TECTONIC_BIN:-$(command -v tectonic || true)}"

if [[ -z "$TECTONIC_BIN" || ! -x "$TECTONIC_BIN" ]]; then
  echo "tectonic is required; set TECTONIC_BIN to the executable path" >&2
  exit 1
fi

if [[ -z "${BUILD_DIR:-}" ]]; then
  BUILD_DIR="$(mktemp -d)"
  CREATED_BUILD_DIR=1
else
  mkdir -p "$BUILD_DIR"
  CREATED_BUILD_DIR=0
fi

cd "$ROOT"
trap 'if [[ "$CREATED_BUILD_DIR" == "1" ]]; then rm -rf "$BUILD_DIR"; fi' EXIT
"$TECTONIC_BIN" --outdir "$BUILD_DIR" "$SOURCE"
mkdir -p "$OUTPUT_DIR"
cp "$BUILD_DIR/primary_robustness_experiment_full_updated_2026-08-13.pdf" "$OUTPUT_DIR/"

echo "Rendered $OUTPUT_DIR/primary_robustness_experiment_full_updated_2026-08-13.pdf"
