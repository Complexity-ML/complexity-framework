#!/bin/bash
set -euo pipefail

cd "${REPO_ROOT:-/workspace/complexity-framework}"

OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/ablations/detector_coco_v06_native}"
EXPECTED_EPOCHS="${EPOCHS:-50}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ARMS=(baseline adapters hash-gate weighting auxiliary full)
if [[ -n "${ABLATION_ARMS:-}" ]]; then
  read -r -a ARMS <<<"$ABLATION_ARMS"
fi

for arm in "${ARMS[@]}"; do
  arm_output="$OUTPUT_ROOT/$arm"
  if checkpoint=$("$PYTHON_BIN" scripts/detector_checkpoint_status.py \
    "$arm_output" --expected-epochs "$EXPECTED_EPOCHS"); then
    printf '[skip] completed specialization arm: %s (%s)\n' "$arm" "$checkpoint"
    continue
  else
    status=$?
  fi

  case "$status" in
    10)
      printf '[resume] specialization arm: %s (%s)\n' "$arm" "$checkpoint"
      ABLATION="$arm" OUTPUT="$arm_output" RESUME_CHECKPOINT="$checkpoint" \
        EPOCHS="$EXPECTED_EPOCHS" scripts/vast_train_detector_specialized_coco.sh
      ;;
    20)
      printf '[start] specialization arm: %s\n' "$arm"
      ABLATION="$arm" OUTPUT="$arm_output" EPOCHS="$EXPECTED_EPOCHS" \
        scripts/vast_train_detector_specialized_coco.sh
      ;;
    *)
      printf '[error] incompatible checkpoint for specialization arm: %s\n' \
        "$arm" >&2
      exit "$status"
      ;;
  esac
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"$PYTHON_BIN" scripts/collect_detector_specialization_ablations.py \
  --root "$OUTPUT_ROOT" \
  --expected-epochs "$EXPECTED_EPOCHS"
