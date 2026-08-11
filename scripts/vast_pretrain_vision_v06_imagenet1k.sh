#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

HF_DATASET="${HF_DATASET:-ILSVRC/imagenet-1k}"
OUTPUT="${OUTPUT:-artifacts/tr_hash_vision_v06_imagenet1k}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-64}"

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
