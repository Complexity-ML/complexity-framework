#!/bin/bash
set -euo pipefail

source "${VENV_ACTIVATE:-/venv/main/bin/activate}"
cd "${REPO_ROOT:-/workspace/complexity-framework}"

DATA_ROOT="${DATA_ROOT:-/workspace/datasets/CUHK-X/extracted}"
OUTPUT="${OUTPUT:-artifacts/tr_hash_robot_perception}"
MANIFEST="${MANIFEST:-artifacts/tr_hash_robot_perception/cuhkx_manifest.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
VALIDATION_USERS="${VALIDATION_USERS:-8 9 23 24}"
IMAGE_SIZE="${IMAGE_SIZE:-112}"
CLIP_FRAMES="${CLIP_FRAMES:-16}"
EPOCHS="${EPOCHS:-100}"
EVAL_EVERY="${EVAL_EVERY:-2}"
PREPROCESSING_VERSION="${PREPROCESSING_VERSION:-3}"
CLASS_BALANCE="${CLASS_BALANCE:-inverse-sqrt}"
CLASS_SAMPLING="${CLASS_SAMPLING:-none}"
SAMPLE_LOSS_WEIGHTING="${SAMPLE_LOSS_WEIGHTING:-none}"
SAMPLE_LOSS_WEIGHT_MIN="${SAMPLE_LOSS_WEIGHT_MIN:-0.0}"
SAMPLE_LOSS_WEIGHT_MAX="${SAMPLE_LOSS_WEIGHT_MAX:-0.0}"
LATE_FUSION_WEIGHT="${LATE_FUSION_WEIGHT:-0.5}"
AUXILIARY_LOSS_WEIGHT="${AUXILIARY_LOSS_WEIGHT:-0.0}"
GATE_CALIBRATION_LOSS_WEIGHT="${GATE_CALIBRATION_LOSS_WEIGHT:-0.0}"
GATE_QUALITY_TEMPERATURE="${GATE_QUALITY_TEMPERATURE:-1.0}"
GATE_TARGET_SMOOTHING="${GATE_TARGET_SMOOTHING:-0.1}"
MIXUP_ALPHA="${MIXUP_ALPHA:-0.2}"
CONTRASTIVE_LOSS_WEIGHT="${CONTRASTIVE_LOSS_WEIGHT:-0.0}"
CONTRASTIVE_TEMPERATURE="${CONTRASTIVE_TEMPERATURE:-0.1}"
SUBJECT_ADVERSARIAL_WEIGHT="${SUBJECT_ADVERSARIAL_WEIGHT:-0.0}"
SUBJECT_ADVERSARIAL_WARMUP_EPOCHS="${SUBJECT_ADVERSARIAL_WARMUP_EPOCHS:-10}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-8}"
SHARED_WIDTH="${SHARED_WIDTH:-256}"
EXPERT_WIDTH="${EXPERT_WIDTH:-128}"
CLASS_HASH_SHARED_WIDTH="${CLASS_HASH_SHARED_WIDTH:-96}"
CLASS_HASH_EXPERT_WIDTH="${CLASS_HASH_EXPERT_WIDTH:-24}"
CLASS_HASH_INITIAL_SCALE="${CLASS_HASH_INITIAL_SCALE:-0.05}"
BACKBONE_LR_MULTIPLIER="${BACKBONE_LR_MULTIPLIER:-1.0}"

# Optional: transfer a pretrained TR-Hash vision tower checkpoint instead of
# starting the visual backbone from random init.
INITIALIZATION=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  INITIALIZATION=(--resume "$RESUME_CHECKPOINT")
elif [[ -n "${VISION_BACKBONE:-}" ]]; then
  INITIALIZATION=(--vision-backbone-checkpoint "$VISION_BACKBONE")
fi

SMOKE=()
if [[ -n "${SMOKE_STEPS:-}" ]]; then
  SMOKE=(--smoke-steps "$SMOKE_STEPS")
fi

exec torchrun \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  -m complexity.generative.sensor_fusion.training \
  --data-root "$DATA_ROOT" \
  --manifest "$MANIFEST" \
  --output "$OUTPUT" \
  "${INITIALIZATION[@]}" \
  "${SMOKE[@]}" \
  --validation-users $VALIDATION_USERS \
  --preprocessing-version "$PREPROCESSING_VERSION" \
  --image-size "$IMAGE_SIZE" \
  --clip-frames "$CLIP_FRAMES" \
  --sensor-steps 64 \
  --hidden-size "$HIDDEN_SIZE" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --num-experts 8 \
  --top-k 2 \
  --shared-width "$SHARED_WIDTH" \
  --expert-width "$EXPERT_WIDTH" \
  --sequence-tokens 32 \
  --class-hash-shared-width "$CLASS_HASH_SHARED_WIDTH" \
  --class-hash-expert-width "$CLASS_HASH_EXPERT_WIDTH" \
  --class-hash-initial-scale "$CLASS_HASH_INITIAL_SCALE" \
  --late-fusion-weight "$LATE_FUSION_WEIGHT" \
  --auxiliary-loss-weight "$AUXILIARY_LOSS_WEIGHT" \
  --gate-calibration-loss-weight "$GATE_CALIBRATION_LOSS_WEIGHT" \
  --gate-quality-temperature "$GATE_QUALITY_TEMPERATURE" \
  --gate-target-smoothing "$GATE_TARGET_SMOOTHING" \
  --contrastive-loss-weight "$CONTRASTIVE_LOSS_WEIGHT" \
  --contrastive-temperature "$CONTRASTIVE_TEMPERATURE" \
  --subject-adversarial-weight "$SUBJECT_ADVERSARIAL_WEIGHT" \
  --subject-adversarial-warmup-epochs "$SUBJECT_ADVERSARIAL_WARMUP_EPOCHS" \
  --precision bf16 \
  --optimizer musgd \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE_PER_GPU" \
  --eval-batch-size 8 \
  --workers 6 \
  --lr 3e-3 \
  --expert-lr-multiplier 1.5 \
  --backbone-lr-multiplier "$BACKBONE_LR_MULTIPLIER" \
  --momentum 0.95 \
  --weight-decay 0.05 \
  --musgd-muon-weight 0.2 \
  --musgd-sgd-weight 1.0 \
  --warmup-steps 200 \
  --min-lr-ratio 0.05 \
  --label-smoothing 0.1 \
  --mixup-alpha "$MIXUP_ALPHA" \
  --visual-jitter 0.1 \
  --temporal-jitter \
  --visual-horizontal-flip 0.5 \
  --visual-crop-jitter 0.15 \
  --sensor-noise 0.01 \
  --modality-dropout 0.15 \
  --class-balance "$CLASS_BALANCE" \
  --class-sampling "$CLASS_SAMPLING" \
  --sample-loss-weighting "$SAMPLE_LOSS_WEIGHTING" \
  --sample-loss-weight-min "$SAMPLE_LOSS_WEIGHT_MIN" \
  --sample-loss-weight-max "$SAMPLE_LOSS_WEIGHT_MAX" \
  --eval-every "$EVAL_EVERY" \
  --save-steps 500 \
  --log-steps 20 \
  --grad-clip 1.0 \
  --no-drop-last \
  --device cuda \
  --require-fused-cuda \
  --seed 42
