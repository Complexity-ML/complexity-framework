#!/bin/bash
set -euo pipefail

# Competitive v8 baseline: narrow 8-way MoE over a dominant shared branch,
# extra depth, a progressive conv stem, relative position bias, and P2 on by
# default (~2.4M parameters).
export OUTPUT="${OUTPUT:-artifacts/detector_coco_v08_nano_o2m}"
export ARCHITECTURE_VERSION=8
export VISION_HIDDEN_SIZE=128
export VISION_LAYERS=7
export VISION_STAGE_DEPTHS="2 2 3"
export VISION_HEADS=4
export VISION_NUM_EXPERTS=8
export VISION_TOP_K=2
export VISION_SHARED_WIDTH=216
export VISION_EXPERT_WIDTH=27
export HEAD_HIDDEN_SIZE=96
export P2_HEAD="${P2_HEAD:-1}"
export END_TO_END=0
export NECK_NORMALIZED_FUSION=1
export NECK_REPEATS=2
export HEAD_SPATIAL_MIXING=1
export REGRESSION_LOGIT_SCALE=1

# Optimization recipe is expressed in optimizer-step units, independently of GPU count.
export NOMINAL_BATCH_SIZE="${NOMINAL_BATCH_SIZE:-64}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-3.0}"
export BOX_LOSS_WEIGHT="${BOX_LOSS_WEIGHT:-7.5}"
export DFL_LOSS_WEIGHT="${DFL_LOSS_WEIGHT:-1.5}"
export QUALITY_LOSS_WEIGHT="${QUALITY_LOSS_WEIGHT:-0.75}"
export MOSAIC="${MOSAIC:-1.0}"
export MOSAIC_TILES="${MOSAIC_TILES:-16}"
export MOSAIC_CANVAS_SIZE="${MOSAIC_CANVAS_SIZE:-1280}"
export MIXUP="${MIXUP:-0.012}"
export COPY_PASTE="${COPY_PASTE:-0.0}"
export CLOSE_MOSAIC_EPOCHS="${CLOSE_MOSAIC_EPOCHS:-10}"
export MOSAIC_PACKED_EPOCH="${MOSAIC_PACKED_EPOCH:-1}"
export PACKED_EPOCHS="${PACKED_EPOCHS:-2}"

exec "${REPO_ROOT:-/workspace/complexity-framework}/scripts/vast_train_detector_specialized_coco.sh"
