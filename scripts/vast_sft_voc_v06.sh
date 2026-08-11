#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

OUTPUT="${OUTPUT:-artifacts/detector_voc_v06}"
SOURCE="${SOURCE:-artifacts/detector_coco_v06/best}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
INITIALIZATION=(--detector-checkpoint "$SOURCE")
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  INITIALIZATION=(--resume "$RESUME_CHECKPOINT")
fi

exec torchrun \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  -m complexity.generative.detection.training \
  --output "$OUTPUT" \
  "${INITIALIZATION[@]}" \
  --yolo-images artifacts/VOC/images/train \
  --yolo-labels artifacts/VOC/labels/train \
  --validation-yolo-images artifacts/VOC/images/val \
  --validation-yolo-labels artifacts/VOC/labels/val \
  --architecture-version 6 \
  --image-size 640 \
  --patch-size 8 \
  --vision-hidden-size 128 \
  --vision-layers 4 \
  --vision-stage-depths 1 1 2 \
  --vision-window-size 8 \
  --vision-heads 4 \
  --vision-num-experts 4 \
  --vision-top-k 2 \
  --vision-expert-width 48 \
  --neck-mode pan \
  --p2-head \
  --end-to-end \
  --one-to-one-loss-weight 0.5 \
  --augmentation strong \
  --mosaic 0.7 \
  --mixup 0.10 \
  --copy-paste 0.10 \
  --random-erasing 0.10 \
  --close-mosaic-epochs 10 \
  --multi-scale-min 512 \
  --multi-scale-max 640 \
  --multi-scale-step 32 \
  --ema-decay 0.9999 \
  --epochs 50 \
  --batch-size "$BATCH_SIZE_PER_GPU" \
  --eval-batch-size 8 \
  --eval-every 5 \
  --eval-max-detections 300 \
  --workers 6 \
  --lr 5e-3 \
  --momentum 0.937 \
  --weight-decay 5e-4 \
  --expert-lr-multiplier 1.5 \
  --warmup-steps 500 \
  --log-steps 20 \
  --save-steps 1000 \
  --eval-confidence 0.05 \
  --require-triton \
  --device cuda \
  --seed 3
