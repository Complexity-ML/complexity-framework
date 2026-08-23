#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

until [[ -s /workspace/.hf_token ]]; do sleep 10; done
export HF_TOKEN="$(< /workspace/.hf_token)"
export HF_XET_HIGH_PERFORMANCE=1

exec python scripts/sync_checkpoints_to_hf.py \
  --checkpoint-dir artifacts/tr_hash_moe_200m_reasoning_preservation_50m_full_1e \
  --repo-id "${HF_REPO_ID:-AETHORIA-AI/TR-HASH-MoE-200M-160B-Reasoning-Preservation-50M}" \
  --path-prefix training/reasoning-preservation-50m/checkpoints \
  --poll-interval 20 \
  --keep-local 12 \
  --no-private
