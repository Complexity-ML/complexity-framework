#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

OUTPUT="${OUTPUT:-artifacts/detector_coco_v06}"
BACKBONE="${BACKBONE:-artifacts/tr_hash_vision_v06_imagenet1k/best}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
BACKBONE_LR_MULTIPLIER="${BACKBONE_LR_MULTIPLIER:-0.1}"
INITIALIZATION=(--backbone-checkpoint "$BACKBONE")
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  INITIALIZATION=(--resume "$RESUME_CHECKPOINT")
fi

exec torchrun \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  -m complexity.generative.detection.training \
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
  --optimizer musgd \
  --epochs 100 \
  --batch-size "$BATCH_SIZE_PER_GPU" \
  --eval-batch-size 8 \
  --workers 6 \
  --lr 5.4e-3 \
  --backbone-lr-multiplier "$BACKBONE_LR_MULTIPLIER" \
  --momentum 0.947 \
  --weight-decay 6.4e-4 \
  --expert-lr-multiplier 1.5 \
  --warmup-steps "$WARMUP_STEPS" \
  --eval-every 5 \
  --save-steps 2000 \
  --require-triton \
  --device cuda \
  --seed 3
