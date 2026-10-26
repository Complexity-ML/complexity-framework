#!/bin/bash
set -euo pipefail

# Source67 v8 P2 entry point; inherits its parent routing profile.
export OUTPUT="${OUTPUT:-artifacts/detector_coco_v08_nano_p2_source67_o2m}"
export P2_HEAD=1
exec "${REPO_ROOT:-/workspace/complexity-framework}/scripts/vast_train_detector_coco_v08_nano.sh"
