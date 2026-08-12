#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

ARM="${1:-}"
if [[ -z "$ARM" ]]; then
  echo "usage: $0 {full|o2m-only|no-stal|no-p2|fpn|no-neck}" >&2
  exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DATASET="${DATASET:-voc}"
BACKBONE="${BACKBONE:-artifacts/tr_hash_vision_v06_imagenet1k/best}"

case "$DATASET" in
  voc)
    DATA_ARGS=(
      --yolo-images artifacts/VOC/images/train
      --yolo-labels artifacts/VOC/labels/train
      --validation-yolo-images artifacts/VOC/images/val
      --validation-yolo-labels artifacts/VOC/labels/val
    )
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/ablations/detector_v06_voc}"
    EPOCHS="${EPOCHS:-50}"
    ONE_TO_ONE_LOSS_WEIGHT="${ONE_TO_ONE_LOSS_WEIGHT:-1.0}"
    MOSAIC="${MOSAIC:-0.7}"
    MIXUP="${MIXUP:-0.10}"
    LR="${LR:-5e-3}"
    MOMENTUM="${MOMENTUM:-0.937}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    ;;
  coco)
    DATA_ARGS=(
      --annotations artifacts/COCO/annotations/instances_train2017.json
      --images artifacts/COCO/images/train2017
      --validation-annotations artifacts/COCO/annotations/instances_val2017.json
      --validation-images artifacts/COCO/images/val2017
    )
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/ablations/detector_v06_coco}"
    EPOCHS="${EPOCHS:-100}"
    ONE_TO_ONE_LOSS_WEIGHT="${ONE_TO_ONE_LOSS_WEIGHT:-0.5}"
    MOSAIC="${MOSAIC:-0.8}"
    MIXUP="${MIXUP:-0.15}"
    LR="${LR:-5.4e-3}"
    MOMENTUM="${MOMENTUM:-0.947}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-6.4e-4}"
    WARMUP_STEPS="${WARMUP_STEPS:-1000}"
    ;;
  *)
    echo "unknown dataset: $DATASET (expected voc or coco)" >&2
    exit 2
    ;;
esac

if (( GLOBAL_BATCH_SIZE % NPROC_PER_NODE != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by NPROC_PER_NODE" >&2
  exit 2
fi
BATCH_SIZE_PER_GPU=$((GLOBAL_BATCH_SIZE / NPROC_PER_NODE))
OUTPUT="${OUTPUT:-${OUTPUT_ROOT}/${ARM}}"

ARCHITECTURE=(--neck-mode pan --p2-head --end-to-end)
case "$ARM" in
  full)
    ;;
  o2m-only)
    ARCHITECTURE=(--neck-mode pan --p2-head --no-end-to-end)
    ;;
  no-stal)
    ARCHITECTURE+=(--no-stal)
    ;;
  no-p2)
    ARCHITECTURE=(--neck-mode pan --no-p2-head --end-to-end)
    ;;
  fpn)
    ARCHITECTURE=(--neck-mode fpn --p2-head --end-to-end)
    ;;
  no-neck)
    ARCHITECTURE=(--neck-mode baseline --p2-head --end-to-end)
    ;;
  *)
    echo "unknown ablation arm: $ARM" >&2
    exit 2
    ;;
esac

COMMAND=(
  torchrun --standalone --nproc_per_node "$NPROC_PER_NODE"
  -m complexity.generative.detection.training
  --output "$OUTPUT"
  --backbone-checkpoint "$BACKBONE"
  "${DATA_ARGS[@]}"
  --architecture-version 6
  --image-size 640
  --patch-size 8
  --vision-hidden-size 128
  --vision-layers 4
  --vision-stage-depths 1 1 2
  --vision-window-size 8
  --vision-heads 4
  --vision-num-experts 4
  --vision-top-k 2
  --vision-expert-width 48
  "${ARCHITECTURE[@]}"
  --one-to-one-loss-weight "$ONE_TO_ONE_LOSS_WEIGHT"
  --one-to-one-loss-start 0.25
  --one-to-one-shared-gradient-scale 0.25
  --one-to-one-lr-multiplier 1.5
  --augmentation strong
  --mosaic "$MOSAIC"
  --mixup "$MIXUP"
  --copy-paste 0.10
  --random-erasing 0.10
  --close-mosaic-epochs 10
  --multi-scale-min 512
  --multi-scale-max 640
  --multi-scale-step 32
  --ema-decay 0.9999
  --optimizer musgd
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE_PER_GPU"
  --eval-batch-size 8
  --eval-every 5
  --eval-max-detections 300
  --workers 6
  --lr "$LR"
  --backbone-lr-multiplier 0.1
  --momentum "$MOMENTUM"
  --weight-decay "$WEIGHT_DECAY"
  --expert-lr-multiplier 1.5
  --warmup-steps "$WARMUP_STEPS"
  --log-steps 20
  --save-steps 1000
  --eval-confidence 0.05
  --require-triton
  --device cuda
  --seed 3
)

printf 'TR-Hash Vision v6 ablation: dataset=%s arm=%s global_batch=%s output=%s\n' \
  "$DATASET" "$ARM" "$GLOBAL_BATCH_SIZE" "$OUTPUT"
printf '%q ' "${COMMAND[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
exec "${COMMAND[@]}"
