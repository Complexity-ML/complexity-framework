#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

while [[ ! -s /workspace/.hf_token ]]; do
  echo "Waiting for /workspace/.hf_token before reasoning-SFT checkpoint sync..."
  sleep 60
done

export HF_TOKEN="$(< /workspace/.hf_token)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/tr_hash_moe_200m_reasoning_sft_500m_full_1e}"
MODEL_REPO="${MODEL_REPO:-AETHORIA-AI/TR-HASH-MoE-200M-160B-Reasoning-SFT}"
HF_PATH_PREFIX="${HF_PATH_PREFIX:-training/reasoning-sft-500m/checkpoints}"
exec python scripts/sync_checkpoints_to_hf.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --repo-id "${MODEL_REPO}" \
  --path-prefix "${HF_PATH_PREFIX}" \
  --keep-local 24 \
  --poll-interval 60 \
  --no-private
