#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

while [[ ! -s /workspace/.hf_token ]]; do
  echo "Waiting for /workspace/.hf_token before Hugging Face checkpoint sync..."
  sleep 60
done

export HF_TOKEN="$(< /workspace/.hf_token)"
exec python scripts/sync_checkpoints_to_hf.py \
  --checkpoint-dir artifacts/tr_hash_moe_200m_clean_sft_v2_full_3e \
  --repo-id AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT \
  --path-prefix training/sft-v2-300k/checkpoints \
  --keep-local 3 \
  --poll-interval 60 \
  --no-private
