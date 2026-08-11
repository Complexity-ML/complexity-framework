#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

BASE_CHECKPOINT="${BASE_CHECKPOINT:-artifacts/tr_hash_moe_500m_20b}"
SFT_BIN="${SFT_BIN:-artifacts/complexity_atlas_posttrain/tokenized/32k-v16}"
SAVE_DIR="${SAVE_DIR:-artifacts/tr_hash_500m_32k_sft}"
RUN_NAME="${RUN_NAME:-tr-hash-500m-32k-tr}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-8}"
EPOCHS="${EPOCHS:-1}"
STEPS="${STEPS:-16103}"

INITIALIZATION=(--checkpoint "$BASE_CHECKPOINT")
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  INITIALIZATION+=(--resume "$RESUME_CHECKPOINT")
elif [[ "${AUTO_RESUME:-1}" == "1" && -f "$SAVE_DIR/latest" ]]; then
  INITIALIZATION+=(--resume "$SAVE_DIR")
fi

exec torchrun \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  -m scripts.sft_500m_32k_tr \
  "${INITIALIZATION[@]}" \
  --sft-bin "$SFT_BIN" \
  --epochs "$EPOCHS" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE_PER_GPU" \
  --seq-len 512 \
  --lr 1e-5 \
  --weight-decay 0.0 \
  --warmup-ratio 0.03 \
  --bf16 \
  --freeze-token-io \
  --use-custom-kernels true \
  --grad-ckpt \
  --loss-chunk-tokens 1024 \
  --eval-at-start \
  --eval-steps 2000 \
  --eval-batches 0 \
  --early-stopping-patience 0 \
  --save-steps 1000 \
  --save-total-limit 3 \
  --save-dir "$SAVE_DIR" \
  --run-name "$RUN_NAME" \
  --log-steps 10 \
  --num-workers 0 \
  --empty-cache-every 0 \
  --seed 42
