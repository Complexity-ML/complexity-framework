#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

exec python -m scripts.run_sft_curriculum \
  --checkpoint artifacts/tr_hash_500m_sft_final_v6 \
  --sft-bin artifacts/complexity_atlas_posttrain/tokenized/32k-v16 \
  --curriculum-config configs/sft_500m_32k_v16_lora_e3.yaml \
  --through-stage generalist-consolidation-e1 \
  --output-root artifacts/tr_hash_500m_reasoning_lora_e3 \
  --world-size 2 \
  --batch-size 24 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-lr-multiplier 20 \
  --no-eval \
  --seed 42
