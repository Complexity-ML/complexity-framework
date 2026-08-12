#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

# The official ILSVRC repository is gated and its 167 GB original-resolution
# cache is too close to the 200 GB cluster disk. This public 256 px repack keeps
# all 1.43M ImageNet-1K examples in roughly 20 GB and is resized/cropped to 224
# by the same training transforms below.
HF_DATASET="${HF_DATASET:-benjamin-paine/imagenet-1k-256x256}"
OUTPUT="${OUTPUT:-artifacts/tr_hash_vision_v06_imagenet1k}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-64}"
export HF_XET_HIGH_PERFORMANCE=1
INITIALIZATION=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  INITIALIZATION=(--resume "$RESUME_CHECKPOINT")
elif [[ "${AUTO_RESUME:-1}" == "1" ]]; then
  shopt -s nullglob
  checkpoints=("$OUTPUT"/step_*)
  shopt -u nullglob
  if (( ${#checkpoints[@]} > 0 )); then
    latest_checkpoint="${checkpoints[${#checkpoints[@]} - 1]}"
    if [[ -f "$latest_checkpoint/training_state.pt" ]]; then
      echo "Auto-resuming from $latest_checkpoint"
      INITIALIZATION=(--resume "$latest_checkpoint")
    fi
  fi
fi

exec torchrun \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  -m complexity.generative.vision_language.pretraining \
  --hf-dataset "$HF_DATASET" \
  --data-root artifacts/hf-cache \
  --output "$OUTPUT" \
  "${INITIALIZATION[@]}" \
  --architecture-version 6 \
  --image-size 224 \
  --patch-size 8 \
  --hidden-size 128 \
  --layers 4 \
  --stage-depths 1 1 2 \
  --window-size 8 \
  --heads 4 \
  --num-experts 4 \
  --top-k 2 \
  --expert-width 48 \
  --optimizer musgd \
  --epochs 100 \
  --batch-size "$BATCH_SIZE_PER_GPU" \
  --workers 6 \
  --lr 5.4e-3 \
  --expert-lr-multiplier 1.5 \
  --momentum 0.947 \
  --musgd-muon-weight 0.2 \
  --musgd-sgd-weight 1.0 \
  --weight-decay 6.4e-4 \
  --warmup-steps 5000 \
  --log-steps 50 \
  --save-steps 2500 \
  --eval-every 5 \
  --device cuda \
  --require-triton \
  --seed 3
