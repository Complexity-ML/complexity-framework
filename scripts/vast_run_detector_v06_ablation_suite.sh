#!/bin/bash
set -euo pipefail

cd /workspace/complexity-framework

DATASET="${DATASET:-voc}"
if [[ "$DATASET" != "voc" && "$DATASET" != "coco" ]]; then
  echo "unknown dataset: $DATASET (expected voc or coco)" >&2
  exit 2
fi
DEFAULT_ROOT="artifacts/ablations/detector_v06_${DATASET}"
if [[ "$DATASET" == "voc" ]]; then
  DEFAULT_REFERENCE="artifacts/detector_voc_v06_imagenet1k_nmsfree"
else
  DEFAULT_REFERENCE="artifacts/detector_coco_v06"
fi

# The existing full PAN/P2/STAL/O2O run is the reference. These arms each
# remove or replace one component while keeping data, seed and compute fixed.
ARMS=(o2m-only no-stal no-p2 fpn no-neck)
if [[ -n "${ABLATION_ARMS:-}" ]]; then
  read -r -a ARMS <<<"$ABLATION_ARMS"
fi

for arm in "${ARMS[@]}"; do
  output_root="${OUTPUT_ROOT:-$DEFAULT_ROOT}"
  if [[ -f "$output_root/$arm/best/validation.json" ]]; then
    echo "[skip] completed arm: $arm"
    continue
  fi
  bash scripts/vast_ablate_detector_v06.sh "$arm"
done

python scripts/collect_detector_v06_ablations.py \
  --root "${OUTPUT_ROOT:-$DEFAULT_ROOT}" \
  --reference "${REFERENCE:-$DEFAULT_REFERENCE}"
