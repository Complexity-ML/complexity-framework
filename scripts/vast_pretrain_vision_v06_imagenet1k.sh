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

exec torchrun \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  -m complexity.generative.vision_language.pretraining \
  --hf-dataset "$HF_DATASET" \
  --data-root artifacts/hf-cache \
  --output "$OUTPUT" \
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
  --epochs 100 \
  --batch-size "$BATCH_SIZE_PER_GPU" \
  --workers 6 \
  --lr 3e-4 \
  --expert-lr-multiplier 1.5 \
  --weight-decay 0.05 \
  --warmup-steps 5000 \
  --log-steps 50 \
  --device cuda \
  --require-triton \
  --seed 3
