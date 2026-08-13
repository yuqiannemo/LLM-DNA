#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/docs/primary_robustness_experiment_full_updated_2026-08-13.html"
OUTPUT="$ROOT/docs/primary_robustness_experiment_full_updated_2026-08-13.pdf"

chromium \
  --headless \
  --no-sandbox \
  --disable-gpu \
  --allow-file-access-from-files \
  --no-pdf-header-footer \
  --print-to-pdf="$OUTPUT" \
  "file://$SOURCE"

echo "Rendered $OUTPUT"
