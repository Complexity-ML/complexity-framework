#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

exec python -u -m complexity.generative.detection.training \
  --output artifacts/detector_voc_5090_linear \
  --backbone-checkpoint artifacts/tr_hash_vision_imagenet100/best \
  --yolo-images artifacts/VOC/images/train \
  --yolo-labels artifacts/VOC/labels/train \
  --validation-yolo-images artifacts/VOC/images/val \
  --validation-yolo-labels artifacts/VOC/labels/val \
  --image-size 224 \
  --patch-size 8 \
  --vision-hidden-size 128 \
  --vision-layers 4 \
  --vision-heads 4 \
  --vision-expert-width 48 \
  --epochs 25 \
  --batch-size 128 \
  --workers 8 \
  --lr 3e-4 \
  --expert-lr-multiplier 1.5 \
  --warmup-steps 200 \
  --log-steps 10 \
  --save-steps 500 \
  --eval-confidence 0.10 \
  --seed 3 \
  --device cuda
