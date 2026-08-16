#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

BASE_CHECKPOINT="${BASE_CHECKPOINT:-artifacts/tr_hash_moe_500m_20b_hf}"
SFT_BIN="${SFT_BIN:-artifacts/complexity_card_corpus_v2_229026/tokenized/32k-v2}"
CURRICULUM_CONFIG="${CURRICULUM_CONFIG:-configs/sft_500m_32k_v2_balanced.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/tr_hash_500m_32k_v2_229026_balanced_lora_r32}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-24}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
EXPERT_LR_MULTIPLIER="${EXPERT_LR_MULTIPLIER:-0.25}"

if [[ -n "${RESUME_FROM:-}" ]]; then
  echo "The balanced V2 run starts from the pretrained base; RESUME_FROM is not allowed." >&2
  exit 2
fi

exec /venv/main/bin/python -m scripts.run_sft_curriculum \
  --checkpoint "$BASE_CHECKPOINT" \
  --sft-bin "$SFT_BIN" \
  --curriculum-config "$CURRICULUM_CONFIG" \
  --through-stage full-shard-weighted \
  --output-root "$OUTPUT_ROOT" \
  --tokenizer tokenizer \
  --world-size "$NPROC_PER_NODE" \
  --batch-size "$BATCH_SIZE_PER_GPU" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout 0.05 \
  --lora-targets q_proj,k_proj,v_proj,o_proj,shared_gate,shared_up,shared_down,expert_gate,expert_up,expert_down \
  --expert-lr-multiplier "$EXPERT_LR_MULTIPLIER" \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 0.0 \
  --seed 42
