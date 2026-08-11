#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

exec python -m scripts.run_sft_curriculum \
  --checkpoint artifacts/tr_hash_500m_sft_final_v6 \
  --sft-bin artifacts/complexity_atlas_posttrain/tokenized/32k-v16 \
  --tokenizer artifacts/tr_hash_500m_tokenizer_32k \
  --curriculum-config configs/sft_500m_32k_v16_lora_e3.yaml \
  --through-stage generalist-consolidation-e1 \
  --output-root artifacts/tr_hash_500m_expert_think_lora \
  --world-size 2 \
  --batch-size 24 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-lr-multiplier 20 \
  --lora-targets q_proj,v_proj,o_proj,shared_gate,shared_up,shared_down,expert_gate,expert_up,expert_down \
  --expert-lr-multiplier 1.5 \
  --reasoning-envelope \
  --no-eval \
  --seed 42
