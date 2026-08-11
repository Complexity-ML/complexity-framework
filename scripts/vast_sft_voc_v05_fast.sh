#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-1e-2}"
WARMUP_STEPS="${WARMUP_STEPS:-150}"

exec python -u -m complexity.generative.detection.training \
  --output artifacts/detector_voc_5090_v05_pan_fast \
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
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-every 5 \
  --eval-max-detections 100 \
  --workers 8 \
  --lr "$LEARNING_RATE" \
  --momentum 0.937 \
  --weight-decay 5e-4 \
  --expert-lr-multiplier 1.5 \
  --warmup-steps "$WARMUP_STEPS" \
  --log-steps 20 \
  --save-steps 1000 \
  --eval-confidence 0.10 \
  --seed 3 \
  --device cuda
