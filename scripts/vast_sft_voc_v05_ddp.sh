#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-64}"
OUTPUT="${OUTPUT:-artifacts/detector_voc_v05_pan_ddp}"
LEARNING_RATE="${LEARNING_RATE:-1e-2}"

exec torchrun \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  -m complexity.generative.detection.training \
  --output "$OUTPUT" \
  --backbone-checkpoint artifacts/tr_hash_vision_imagenet100/best \
  --yolo-images artifacts/VOC/images/train \
  --yolo-labels artifacts/VOC/labels/train \
  --validation-yolo-images artifacts/VOC/images/val \
  --validation-yolo-labels artifacts/VOC/labels/val \
  --architecture-version 5 \
  --neck-mode pan \
  --image-size 224 \
  --patch-size 8 \
  --vision-hidden-size 128 \
  --vision-layers 4 \
  --vision-heads 4 \
  --vision-num-experts 4 \
  --vision-top-k 2 \
  --vision-expert-width 48 \
  --p2-head \
  --reg-max 16 \
  --dfl-loss-weight 0.5 \
  --quality-focal-beta 2.0 \
  --augmentation strong \
  --epochs 50 \
  --batch-size "$BATCH_SIZE_PER_GPU" \
  --eval-batch-size "$BATCH_SIZE_PER_GPU" \
  --eval-every 5 \
  --eval-max-detections 100 \
  --workers 4 \
  --lr "$LEARNING_RATE" \
  --momentum 0.937 \
  --weight-decay 5e-4 \
  --expert-lr-multiplier 1.5 \
  --warmup-steps 40 \
  --log-steps 20 \
  --save-steps 500 \
  --eval-confidence 0.10 \
  --seed 3 \
  --device cuda
