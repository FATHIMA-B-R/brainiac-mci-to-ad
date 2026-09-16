#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TOKENS=${1:?Usage: bash scripts/run_attentive_experiments.sh TOKENS.pt NEUROVFM_ROOT OUTPUT_ROOT [probe options]}
NV=${2:?Provide NeuroVFM repository path}
OUT=${3:?Provide a new output root}
shift 3
for EXP in 88_88_split21 88_88_split42 88_132_split21 88_132_split42 88_251; do
  test -f "comparison_splits/splits_${EXP}.csv"
  if [[ -e "$OUT/$EXP" ]]; then
    echo "Output already exists: $OUT/$EXP" >&2
    exit 1
  fi
done
for EXP in 88_88_split21 88_88_split42 88_132_split21 88_132_split42 88_251; do
  SEED=0
  case "$EXP" in
    *_split21) SEED=21 ;;
    *_split42) SEED=42 ;;
  esac
  python scripts/train_brainiac_attentive.py \
    --tokens "$TOKENS" --neurovfm-root "$NV" \
    --split "comparison_splits/splits_${EXP}.csv" \
    --output "$OUT/$EXP" --split-seed "$SEED" "$@"
done
