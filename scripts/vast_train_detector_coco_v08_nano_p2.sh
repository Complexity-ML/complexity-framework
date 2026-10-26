#!/bin/bash
set -euo pipefail

# TR-HASH v8 P2 entry point.
export OUTPUT="${OUTPUT:-artifacts/detector_coco_v08_nano_p2_o2m}"
export P2_HEAD=1
exec "${REPO_ROOT:-/workspace/complexity-framework}/scripts/vast_train_detector_coco_v08_nano.sh"
