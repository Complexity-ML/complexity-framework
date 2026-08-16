#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

FOLDS="${FOLDS:-fold_a fold_b fold_c}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/tr_hash_cuhkx_sensor_fusion_v12_cv}"

for fold in $FOLDS; do
  output="$OUTPUT_ROOT/$fold"
  if [[ -f "$output/training_complete.json" ]]; then
    echo "[skip] completed cross-subject run: $fold"
    continue
  fi

  resume="$(python scripts/latest_sensor_fusion_checkpoint.py "$output")"
  if [[ -n "$resume" ]]; then
    echo "[resume] $fold from $resume"
    FOLD_ID="$fold" OUTPUT="$output" RESUME_CHECKPOINT="$resume" \
      scripts/vast_train_cuhkx_sensor_fusion_v12_fold.sh
  else
    echo "[start] $fold"
    FOLD_ID="$fold" OUTPUT="$output" \
      scripts/vast_train_cuhkx_sensor_fusion_v12_fold.sh
  fi
done
