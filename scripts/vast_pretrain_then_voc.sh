#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

export HF_HOME=/workspace/complexity-framework/artifacts/hf-cache
export HF_XET_HIGH_PERFORMANCE=1

PRETRAIN_OUTPUT=artifacts/tr_hash_vision_imagenet100

if [[ ! -f "$PRETRAIN_OUTPUT/last/tower.safetensors" ]]; then
  python -u -m complexity.generative.vision_language.pretraining \
    --hf-dataset clane9/imagenet-100 \
    --data-root artifacts/hf-cache \
    --output "$PRETRAIN_OUTPUT" \
    --image-size 224 \
    --patch-size 8 \
    --hidden-size 128 \
    --layers 4 \
    --heads 4 \
    --num-experts 4 \
    --top-k 2 \
    --expert-width 48 \
    --epochs 30 \
    --batch-size 512 \
    --workers 8 \
    --lr 5e-4 \
    --expert-lr-multiplier 1.5 \
    --warmup-steps 200 \
    --log-steps 25 \
    --seed 3 \
    --device cuda
fi

exec python -u -m complexity.generative.detection.training \
  --output artifacts/detector_voc_5090_imagenet100 \
  --backbone-checkpoint "$PRETRAIN_OUTPUT/best" \
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
  --epochs 50 \
  --batch-size 256 \
  --workers 8 \
  --lr 5e-4 \
  --expert-lr-multiplier 1.5 \
  --warmup-steps 100 \
  --log-steps 10 \
  --save-steps 500 \
  --eval-confidence 0.10 \
  --seed 3 \
  --device cuda
