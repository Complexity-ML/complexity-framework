#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

exec python -u -m complexity.generative.detection.training \
  --output artifacts/detector_voc_5090_v03 \
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
  --p2-head \
  --epochs 50 \
  --batch-size 64 \
  --workers 8 \
  --optimizer sgd \
  --lr 1e-2 \
  --momentum 0.937 \
  --weight-decay 5e-4 \
  --expert-lr-multiplier 1.5 \
  --warmup-steps 300 \
  --log-steps 20 \
  --save-steps 1000 \
  --eval-confidence 0.10 \
  --seed 3 \
  --device cuda
