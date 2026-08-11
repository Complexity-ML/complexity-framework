#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

OUTPUT="${OUTPUT:-artifacts/detector_coco_v06}"
BACKBONE="${BACKBONE:-artifacts/tr_hash_vision_v06_imagenet1k/best}"
INITIALIZATION=(--backbone-checkpoint "$BACKBONE")
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  INITIALIZATION=(--resume "$RESUME_CHECKPOINT")
fi

exec python -u -m complexity.generative.detection.training \
  --output "$OUTPUT" \
  "${INITIALIZATION[@]}" \
  --annotations artifacts/COCO/annotations/instances_train2017.json \
  --images artifacts/COCO/images/train2017 \
  --validation-annotations artifacts/COCO/annotations/instances_val2017.json \
  --validation-images artifacts/COCO/images/val2017 \
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
  --mosaic 0.8 \
  --mixup 0.15 \
  --copy-paste 0.10 \
  --random-erasing 0.10 \
  --close-mosaic-epochs 10 \
  --multi-scale-min 512 \
  --multi-scale-max 640 \
  --multi-scale-step 32 \
  --ema-decay 0.9999 \
  --epochs 100 \
  --batch-size 16 \
  --workers 12 \
  --lr 1e-2 \
  --expert-lr-multiplier 1.5 \
  --warmup-steps 1000 \
  --eval-every 5 \
  --save-steps 2000 \
  --require-triton \
  --device cuda \
  --seed 3
