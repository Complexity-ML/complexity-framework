#!/bin/bash
set -euo pipefail

cd "${REPO_ROOT:-/workspace/complexity-framework}"

OUTPUT="${OUTPUT:-artifacts/detector_coco_v06_native}"
EXPECTED_EPOCHS="${EPOCHS:-245}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python
fi

if checkpoint=$("$PYTHON_BIN" scripts/detector_checkpoint_status.py \
  "$OUTPUT" --expected-epochs "$EXPECTED_EPOCHS"); then
  printf '[skip] native COCO training complete: %s\n' "$checkpoint"
  exit 0
else
  status=$?
fi

case "$status" in
  10)
    printf '[resume] native COCO detector: %s\n' "$checkpoint"
    OUTPUT="$OUTPUT" EPOCHS="$EXPECTED_EPOCHS" RESUME_CHECKPOINT="$checkpoint" \
      scripts/vast_train_detector_specialized_coco.sh
    ;;
  20)
    printf '[start] native COCO detector from random initialization\n'
    OUTPUT="$OUTPUT" EPOCHS="$EXPECTED_EPOCHS" \
      scripts/vast_train_detector_specialized_coco.sh
    ;;
  *)
    printf '[error] incompatible native COCO checkpoint: %s\n' "$checkpoint" >&2
    exit "$status"
    ;;
esac
