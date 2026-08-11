#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

exec python -u -m complexity.generative.detection.training \
  --output artifacts/detector_voc_5090_o2m_recovery \
  --detector-checkpoint artifacts/detector_voc_5090_v03/best \
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
  --epochs 20 \
  --batch-size 64 \
  --workers 8 \
  --optimizer sgd \
  --lr 2e-3 \
  --momentum 0.937 \
  --weight-decay 5e-4 \
  --expert-lr-multiplier 1.5 \
  --warmup-steps 100 \
  --no-progressive-loss \
  --log-steps 20 \
  --save-steps 1000 \
  --eval-confidence 0.10 \
  --seed 3 \
  --device cuda
