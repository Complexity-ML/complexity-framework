#!/bin/bash
set -euo pipefail

# Vision recipe exception: its existing clean-image phase is the refinement
# component of the detector's pretraining lineage. Do not insert the generic
# non-Vision refinement stage before this phase. It transfers model/EMA weights
# and intentionally resets optimizer, scheduler, epoch, step, and RNG state.
export TRAINING_PURPOSE=vision-supervised-finetuning
export DETECTOR_CHECKPOINT="${DETECTOR_CHECKPOINT:-artifacts/detector_coco_v08_nano_o2m/best}"
export OUTPUT="${OUTPUT:-artifacts/detector_coco_v08_nano_sft}"

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
export P2_HEAD=1
export END_TO_END=0
export NECK_NORMALIZED_FUSION=1
export NECK_REPEATS=2
export HEAD_SPATIAL_MIXING=1
export REGRESSION_LOGIT_SCALE=1

# Low-LR, clean-image refinement. No Mosaic/MixUp/Copy-Paste and no packed
# epoch shortcut: every epoch sees the complete COCO training split.
export EPOCHS="${EPOCHS:-30}"
export LR="${LR:-5.4e-4}"
export BACKBONE_LR_MULTIPLIER="${BACKBONE_LR_MULTIPLIER:-1.0}"
export EXPERT_LR_MULTIPLIER="${EXPERT_LR_MULTIPLIER:-1.0}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-1.0}"
export NOMINAL_BATCH_SIZE="${NOMINAL_BATCH_SIZE:-64}"
export AUGMENTATION="${AUGMENTATION:-light}"
export MOSAIC=0.0
export MIXUP=0.0
export COPY_PASTE=0.0
export RANDOM_ERASING=0.0
export CLOSE_MOSAIC_EPOCHS=0
export PACKED_EPOCHS=1
export EVAL_EVERY="${EVAL_EVERY:-1}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export BOX_LOSS_WEIGHT="${BOX_LOSS_WEIGHT:-7.5}"
export DFL_LOSS_WEIGHT="${DFL_LOSS_WEIGHT:-1.5}"
export QUALITY_LOSS_WEIGHT="${QUALITY_LOSS_WEIGHT:-0.75}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for required in config.json provenance.json validation.json; do
    if [[ ! -f "$DETECTOR_CHECKPOINT/$required" ]]; then
      echo "[error] missing $DETECTOR_CHECKPOINT/$required" >&2
      exit 2
    fi
  done
  if [[ ! -f "$DETECTOR_CHECKPOINT/ema.safetensors" && \
        ! -f "$DETECTOR_CHECKPOINT/model.safetensors" ]]; then
    echo "[error] detector checkpoint has no EMA or model weights" >&2
    exit 2
  fi
fi

exec "${REPO_ROOT:-/workspace/complexity-framework}/scripts/vast_train_detector_specialized_coco.sh"
