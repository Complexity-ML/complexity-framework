#!/bin/bash
set -euo pipefail

source "${VENV_ACTIVATE:-/venv/main/bin/activate}"
cd "${REPO_ROOT:-/workspace/complexity-framework}"

ABLATION="${ABLATION:-full}"
OUTPUT="${OUTPUT:-artifacts/detector_coco_v06_native}"
TRAINING_PURPOSE="${TRAINING_PURPOSE:-detector-pretraining}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
TRAIN_ANNOTATIONS="${TRAIN_ANNOTATIONS:-artifacts/COCO/annotations/instances_train2017.json}"
TRAIN_IMAGES="${TRAIN_IMAGES:-artifacts/COCO/images/train2017}"
VALIDATION_ANNOTATIONS="${VALIDATION_ANNOTATIONS:-artifacts/COCO/annotations/instances_val2017.json}"
VALIDATION_IMAGES="${VALIDATION_IMAGES:-artifacts/COCO/images/val2017}"
YOLO_TRAIN_IMAGES="${YOLO_TRAIN_IMAGES:-datasets/coco/images/train2017}"
YOLO_TRAIN_LABELS="${YOLO_TRAIN_LABELS:-datasets/coco/labels/train2017}"
YOLO_VALIDATION_IMAGES="${YOLO_VALIDATION_IMAGES:-datasets/coco/images/val2017}"
YOLO_VALIDATION_LABELS="${YOLO_VALIDATION_LABELS:-datasets/coco/labels/val2017}"
WORKERS="${WORKERS:-6}"
ARCHITECTURE_VERSION="${ARCHITECTURE_VERSION:-6}"
VISION_HIDDEN_SIZE="${VISION_HIDDEN_SIZE:-128}"
VISION_LAYERS="${VISION_LAYERS:-4}"
VISION_STAGE_DEPTHS="${VISION_STAGE_DEPTHS:-1 1 2}"
VISION_HEADS="${VISION_HEADS:-4}"
VISION_NUM_EXPERTS="${VISION_NUM_EXPERTS:-4}"
VISION_TOP_K="${VISION_TOP_K:-2}"
VISION_SHARED_WIDTH="${VISION_SHARED_WIDTH:-96}"
VISION_EXPERT_WIDTH="${VISION_EXPERT_WIDTH:-48}"
HEAD_HIDDEN_SIZE="${HEAD_HIDDEN_SIZE:-0}"
P2_HEAD="${P2_HEAD:-1}"
END_TO_END="${END_TO_END:-1}"
NECK_NORMALIZED_FUSION="${NECK_NORMALIZED_FUSION:-0}"
NECK_REPEATS="${NECK_REPEATS:-1}"
HEAD_SPATIAL_MIXING="${HEAD_SPATIAL_MIXING:-0}"
REGRESSION_LOGIT_SCALE="${REGRESSION_LOGIT_SCALE:-0}"
MOSAIC_VALUE="${MOSAIC:-}"
read -r -a VISION_STAGE_DEPTH_ARGS <<<"$VISION_STAGE_DEPTHS"

case "$TRAINING_PURPOSE" in
  detector-pretraining)
    MOSAIC_VALUE="${MOSAIC_VALUE:-0.909}"
    if [[ "$MOSAIC_VALUE" =~ ^0*([.]0*)?$ ]]; then
      echo "MOSAIC must be greater than zero; Mosaic-free detector pretraining is forbidden" >&2
      exit 64
    fi
    # Native pretraining cannot import external weights. Exact resume is the
    # only permitted continuation after provenance validation.
    for forbidden in INTERMEDIATE INITIALIZATION_MODE DETECTOR_CHECKPOINT BACKBONE_CHECKPOINT; do
      if [[ -n "${!forbidden:-}" ]]; then
        echo "[error] $forbidden is forbidden by the native COCO recipe" >&2
        exit 2
      fi
    done
    ;;
  vision-supervised-finetuning)
    MOSAIC_VALUE="${MOSAIC_VALUE:-0.0}"
    if [[ -z "${DETECTOR_CHECKPOINT:-}" ]]; then
      echo "[error] vision SFT requires DETECTOR_CHECKPOINT" >&2
      exit 2
    fi
    for forbidden in RESUME_CHECKPOINT BACKBONE_CHECKPOINT; do
      if [[ -n "${!forbidden:-}" ]]; then
        echo "[error] $forbidden is forbidden during vision SFT; use detector weight transfer" >&2
        exit 2
      fi
    done
    ;;
  *)
    echo "[error] unsupported TRAINING_PURPOSE=$TRAINING_PURPOSE" >&2
    exit 2
    ;;
esac

DATA_FORMAT="${DATA_FORMAT:-auto}"
if [[ "$DATA_FORMAT" == "auto" ]]; then
  if [[ -f "$TRAIN_ANNOTATIONS" && -d "$TRAIN_IMAGES" && \
        -f "$VALIDATION_ANNOTATIONS" && -d "$VALIDATION_IMAGES" ]]; then
    DATA_FORMAT=coco-json
  elif [[ -d "$YOLO_TRAIN_IMAGES" && -d "$YOLO_TRAIN_LABELS" && \
          -d "$YOLO_VALIDATION_IMAGES" && -d "$YOLO_VALIDATION_LABELS" ]]; then
    DATA_FORMAT=yolo
  elif [[ "${DRY_RUN:-0}" == "1" ]]; then
    DATA_FORMAT=coco-json
  else
    echo "[error] COCO 2017 annotations were not found" >&2
    exit 2
  fi
fi
case "$DATA_FORMAT" in
  coco-json)
    DATA_ARGS=(
      --annotations "$TRAIN_ANNOTATIONS"
      --images "$TRAIN_IMAGES"
      --validation-annotations "$VALIDATION_ANNOTATIONS"
      --validation-images "$VALIDATION_IMAGES"
    )
    ;;
  yolo)
    DATA_ARGS=(
      --yolo-images "$YOLO_TRAIN_IMAGES"
      --yolo-labels "$YOLO_TRAIN_LABELS"
      --validation-yolo-images "$YOLO_VALIDATION_IMAGES"
      --validation-yolo-labels "$YOLO_VALIDATION_LABELS"
      --num-classes 80
    )
    ;;
  *)
    echo "[error] unsupported DATA_FORMAT=$DATA_FORMAT" >&2
    exit 2
    ;;
esac

FEATURES=()
case "$ABLATION" in
  baseline)
    ;;
  adapters)
    FEATURES+=(--level-adapters --level-adapter-ratio 0.25)
    ;;
  hash-gate)
    FEATURES+=(
      --level-adapters --level-adapter-ratio 0.25
      --class-level-hash-gate --class-level-gate-temperature 1.0
    )
    ;;
  weighting)
    FEATURES+=(
      --level-adapters --level-adapter-ratio 0.25
      --class-level-hash-gate --object-weighting
    )
    ;;
  auxiliary)
    FEATURES+=(
      --level-adapters --level-adapter-ratio 0.25
      --class-level-hash-gate
      --object-weighting
      --level-aux-loss-weight 0.10
      --gate-calibration-loss-weight 0.10
    )
    ;;
  full)
    FEATURES+=(
      --level-adapters --level-adapter-ratio 0.25
      --class-level-hash-gate
      --class-level-gate-temperature 1.0
      --object-weighting
      --object-weighting-beta 0.999
      --object-weighting-max 4.0
      --level-aux-loss-weight 0.10
      --gate-calibration-loss-weight 0.10
      --object-contrastive-loss-weight 0.05
      --object-contrastive-temperature 0.10
    )
    ;;
  *)
    echo "unknown ABLATION=$ABLATION" >&2
    exit 2
    ;;
esac

COMMAND=(torchrun --standalone --nproc_per_node "$NPROC_PER_NODE"
  -m complexity.generative.detection.training
  --output "$OUTPUT"
  "${DATA_ARGS[@]}"
  --training-purpose "$TRAINING_PURPOSE"
  --provenance-dataset coco-2017
  --architecture-version "$ARCHITECTURE_VERSION"
  --image-size 640
  --patch-size 8
  --vision-hidden-size "$VISION_HIDDEN_SIZE"
  --vision-layers "$VISION_LAYERS"
  --vision-stage-depths "${VISION_STAGE_DEPTH_ARGS[@]}"
  --vision-window-size 8
  --vision-heads "$VISION_HEADS"
  --vision-num-experts "$VISION_NUM_EXPERTS"
  --vision-top-k "$VISION_TOP_K"
  --vision-shared-width "$VISION_SHARED_WIDTH"
  --vision-expert-width "$VISION_EXPERT_WIDTH"
  --head-hidden-size "$HEAD_HIDDEN_SIZE"
  --neck-mode pan
  --neck-repeats "$NECK_REPEATS"
  --assignment-top-k 8
  --augmentation "${AUGMENTATION:-strong}"
  --augmentation-backend "${AUGMENTATION_BACKEND:-albumentations}"
  --image-backend "${IMAGE_BACKEND:-opencv}"
  --mosaic "$MOSAIC_VALUE"
  --mosaic-tiles "${MOSAIC_TILES:-4}"
  --mosaic-canvas-size "${MOSAIC_CANVAS_SIZE:-0}"
  --mixup "${MIXUP:-0.012}"
  --copy-paste "${COPY_PASTE:-0.075}"
  --random-erasing "${RANDOM_ERASING:-0.0}"
  --close-mosaic-epochs "${CLOSE_MOSAIC_EPOCHS:-10}"
  --packed-epochs "${PACKED_EPOCHS:-1}"
  --multi-scale-min 512
  --multi-scale-max 640
  --multi-scale-step 32
  --ema-decay 0.9999
  --optimizer musgd
  --musgd-muon-weight 0.528
  --musgd-sgd-weight 0.674
  --epochs "${EPOCHS:-245}"
  --batch-size "$BATCH_SIZE_PER_GPU"
  --eval-batch-size "${EVAL_BATCH_SIZE:-8}"
  --workers "$WORKERS"
  --lr "${LR:-5.4e-3}"
  --backbone-lr-multiplier "${BACKBONE_LR_MULTIPLIER:-1.0}"
  --momentum 0.947
  --weight-decay 6.4e-4
  --expert-lr-multiplier "${EXPERT_LR_MULTIPLIER:-1.5}"
  --warmup-steps "${WARMUP_STEPS:-906}"
  --min-lr-ratio 0.04952
  --eval-confidence 0.001
  --eval-backend "${EVAL_BACKEND:-auto}"
  --eval-max-detections 100
  --eval-every "${EVAL_EVERY:-5}"
  --save-steps "${SAVE_STEPS:-2000}"
  --require-triton
  --device cuda
  --seed 3)

if [[ "$NECK_NORMALIZED_FUSION" == "1" ]]; then
  COMMAND+=(--neck-normalized-fusion)
fi
if [[ "$HEAD_SPATIAL_MIXING" == "1" ]]; then
  COMMAND+=(--head-spatial-mixing)
fi
if [[ "$REGRESSION_LOGIT_SCALE" == "1" ]]; then
  COMMAND+=(--regression-logit-scale)
fi
if [[ -n "${NOMINAL_BATCH_SIZE:-}" ]]; then
  COMMAND+=(--nominal-batch-size "$NOMINAL_BATCH_SIZE")
fi
if [[ -n "${ACCUMULATION_STEPS:-}" ]]; then
  COMMAND+=(--accumulation-steps "$ACCUMULATION_STEPS")
fi
if [[ -n "${WARMUP_EPOCHS:-}" ]]; then
  COMMAND+=(--warmup-epochs "$WARMUP_EPOCHS")
fi
COMMAND+=(
  --box-loss-weight "${BOX_LOSS_WEIGHT:-5.0}"
  --dfl-loss-weight "${DFL_LOSS_WEIGHT:-0.5}"
  --quality-loss-weight "${QUALITY_LOSS_WEIGHT:-1.0}"
)

if [[ "$P2_HEAD" == "1" ]]; then
  COMMAND+=(--p2-head)
else
  COMMAND+=(--no-p2-head)
fi
if [[ "$END_TO_END" == "1" ]]; then
  COMMAND+=(--end-to-end --one-to-one-loss-weight "${ONE_TO_ONE_LOSS_WEIGHT:-0.5}")
else
  COMMAND+=(--no-end-to-end)
fi

if [[ "$TRAINING_PURPOSE" == "detector-pretraining" ]]; then
  COMMAND+=(--require-random-init)
  if [[ "${MOSAIC_PACKED_EPOCH:-1}" != "1" ]]; then
    echo "MOSAIC_PACKED_EPOCH must be 1; unpacked detector pretraining is forbidden" >&2
    exit 64
  fi
  COMMAND+=(--mosaic-packed-epoch)
else
  COMMAND+=(--detector-checkpoint "$DETECTOR_CHECKPOINT")
fi

if (( ${#FEATURES[@]} )); then
  COMMAND+=("${FEATURES[@]}")
fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  COMMAND+=(--resume "$RESUME_CHECKPOINT")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
exec "${COMMAND[@]}"
