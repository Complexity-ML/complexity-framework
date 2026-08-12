#!/bin/bash
set -euo pipefail

cd "${WORKSPACE:-/workspace/complexity-framework}"

DATASET="${DATASET:-voc}"
if [[ "$DATASET" != "voc" && "$DATASET" != "coco" ]]; then
  echo "unknown dataset: $DATASET (expected voc or coco)" >&2
  exit 2
fi
DEFAULT_ROOT="artifacts/ablations/detector_v06_${DATASET}"
if [[ "$DATASET" == "voc" ]]; then
  DEFAULT_REFERENCE="artifacts/detector_voc_v06_imagenet1k_nmsfree"
  EXPECTED_EPOCHS="${EPOCHS:-50}"
else
  DEFAULT_REFERENCE="artifacts/detector_coco_v06"
  EXPECTED_EPOCHS="${EPOCHS:-100}"
fi

# The existing full PAN/P2/STAL/O2O run is the reference. These arms each
# remove or replace one component while keeping data, seed and compute fixed.
ARMS=(o2m-only no-stal no-p2 fpn no-neck)
if [[ -n "${ABLATION_ARMS:-}" ]]; then
  read -r -a ARMS <<<"$ABLATION_ARMS"
fi

for arm in "${ARMS[@]}"; do
  output_root="${OUTPUT_ROOT:-$DEFAULT_ROOT}"
  arm_output="$output_root/$arm"
  if checkpoint=$(python scripts/detector_checkpoint_status.py \
    "$arm_output" --expected-epochs "$EXPECTED_EPOCHS"); then
    echo "[skip] completed arm: $arm ($checkpoint)"
    continue
  else
    status=$?
  fi

  case "$status" in
    10)
      echo "[resume] incomplete arm: $arm ($checkpoint)"
      OUTPUT="$arm_output" RESUME_CHECKPOINT="$checkpoint" \
        bash scripts/vast_ablate_detector_v06.sh "$arm"
      ;;
    20)
      echo "[start] new arm: $arm"
      OUTPUT="$arm_output" bash scripts/vast_ablate_detector_v06.sh "$arm"
      ;;
    *)
      echo "[error] incompatible checkpoint state for arm: $arm" >&2
      exit "$status"
      ;;
  esac
done

python scripts/collect_detector_v06_ablations.py \
  --root "${OUTPUT_ROOT:-$DEFAULT_ROOT}" \
  --reference "${REFERENCE:-$DEFAULT_REFERENCE}"
