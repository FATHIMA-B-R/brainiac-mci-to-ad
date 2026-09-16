#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
FEATURES=${1:?Usage: bash scripts/run_mci_experiments.sh FEATURES.csv [OUTPUT_ROOT] [--dev-only]}
OUTPUT_ROOT=${2:-results/mci_linear_probe}
EXTRA=()
if [[ ${3:-} == --dev-only ]]; then
  EXTRA+=(--dev-only)
elif [[ -n ${3:-} ]]; then
  echo 'Third argument must be --dev-only when supplied' >&2
  exit 1
fi
EXPERIMENTS=(88_88_split21 88_88_split42 88_132_split21 88_132_split42 88_251)
# Check all paths before starting any experiment.
for EXP in "${EXPERIMENTS[@]}"; do
  test -f "comparison_splits/splits_${EXP}.csv"
  if [[ -e "$OUTPUT_ROOT/$EXP" ]]; then
    echo "Output already exists: $OUTPUT_ROOT/$EXP; choose a new output root." >&2
    exit 1
  fi
done
for EXP in "${EXPERIMENTS[@]}"; do
  python scripts/mci_conversion.py evaluate \
    --features "$FEATURES" \
    --split "comparison_splits/splits_${EXP}.csv" \
    --output "$OUTPUT_ROOT/$EXP" "${EXTRA[@]}"
done
