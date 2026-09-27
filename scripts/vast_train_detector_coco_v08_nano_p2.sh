#!/bin/bash
set -euo pipefail

# Controlled P2 ablation: identical v8 recipe, adding only the fine prediction level.
export OUTPUT="${OUTPUT:-artifacts/detector_coco_v08_nano_p2_o2m}"
export P2_HEAD=1
exec "${REPO_ROOT:-/workspace/complexity-framework}/scripts/vast_train_detector_coco_v08_nano.sh"
