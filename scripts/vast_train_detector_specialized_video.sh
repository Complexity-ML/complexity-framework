#!/bin/bash
set -euo pipefail

source "${VENV_ACTIVATE:-/venv/main/bin/activate}"
cd "${REPO_ROOT:-/workspace/complexity-framework}"

: "${VIDEO_ANNOTATIONS:?set VIDEO_ANNOTATIONS to a COCO-Video train JSON}"
: "${VIDEO_IMAGES:?set VIDEO_IMAGES to the video frame root}"
: "${VALIDATION_VIDEO_ANNOTATIONS:?set VALIDATION_VIDEO_ANNOTATIONS}"
: "${VALIDATION_VIDEO_IMAGES:?set VALIDATION_VIDEO_IMAGES}"

OUTPUT="${OUTPUT:-artifacts/detector_video_v06_specialized}"
INTERMEDIATE="${INTERMEDIATE:-artifacts/detector_coco_v06_native/best}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-8}"

exec torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
  -m complexity.generative.detection.training \
  --output "$OUTPUT" \
  --detector-checkpoint "$INTERMEDIATE" \
  --video-annotations "$VIDEO_ANNOTATIONS" \
  --video-images "$VIDEO_IMAGES" \
  --validation-video-annotations "$VALIDATION_VIDEO_ANNOTATIONS" \
  --validation-video-images "$VALIDATION_VIDEO_IMAGES" \
  --video-clip-frames "${VIDEO_CLIP_FRAMES:-5}" \
  --video-frame-stride "${VIDEO_FRAME_STRIDE:-1}" \
  --image-size 640 \
  --patch-size 8 \
  --vision-hidden-size 128 \
  --vision-layers 4 \
  --vision-stage-depths 1 1 2 \
  --vision-window-size 8 \
  --vision-heads 4 \
  --vision-num-experts 4 \
  --vision-top-k 2 \
  --vision-shared-width 96 \
  --vision-expert-width 48 \
  --neck-mode pan \
  --p2-head \
  --end-to-end \
  --video-motion \
  --video-motion-hidden-size 64 \
  --level-adapters \
  --class-level-hash-gate \
  --object-weighting \
  --level-aux-loss-weight 0.10 \
  --gate-calibration-loss-weight 0.10 \
  --object-contrastive-loss-weight 0.05 \
  --augmentation strong \
  --multi-scale-min 512 \
  --multi-scale-max 640 \
  --multi-scale-step 32 \
  --ema-decay 0.9999 \
  --optimizer musgd \
  --epochs "${EPOCHS:-100}" \
  --batch-size "$BATCH_SIZE_PER_GPU" \
  --eval-batch-size 4 \
  --workers 6 \
  --lr "${LR:-5.4e-3}" \
  --backbone-lr-multiplier 0.1 \
  --momentum 0.947 \
  --weight-decay 6.4e-4 \
  --expert-lr-multiplier 1.5 \
  --warmup-steps "${WARMUP_STEPS:-1000}" \
  --eval-every 5 \
  --save-steps 2000 \
  --require-triton \
  --device cuda \
  --seed 3
