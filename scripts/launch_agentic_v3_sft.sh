#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   CHECKPOINT=checkpoints/sft-50m-plus5b-general/step_001000 \
#   bash scripts/launch_agentic_v3_sft.sh
#
# Optional env:
#   DATA=data/sft/agentic_v3_mixed.jsonl
#   TOKENIZER=./tokenizer-o200k
#   RUN_NAME=sft-50m-plus5b-agentic-v3
#   STEPS=1000
#   BATCH_SIZE=128
#   SEQ_LEN=1024
#   LR=3e-5
#   NPROC=1

CHECKPOINT="${CHECKPOINT:-checkpoints/sft-50m-plus5b-general/step_001000}"
DATA="${DATA:-data/sft/agentic_v3_mixed.jsonl}"
TOKENIZER="${TOKENIZER:-./tokenizer-o200k}"
RUN_NAME="${RUN_NAME:-sft-50m-plus5b-agentic-v3}"
STEPS="${STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEQ_LEN="${SEQ_LEN:-1024}"
LR="${LR:-3e-5}"
NPROC="${NPROC:-1}"

if [[ ! -f "$DATA" ]]; then
  python3 scripts/prepare_agentic_v3_dataset.py --out "$DATA" --records "${RECORDS:-80000}"
fi

COMMON_ARGS=(
  -m scripts.sft_100m_o200k_tr_local
  --checkpoint "$CHECKPOINT"
  --tokenizer "$TOKENIZER"
  --jsonl "$DATA"
  --seq-len "$SEQ_LEN"
  --steps "$STEPS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --bf16
  --grad-ckpt
  --loss-chunk-tokens "${LOSS_CHUNK_TOKENS:-512}"
  --save-steps "${SAVE_STEPS:-250}"
  --save-total-limit "${SAVE_TOTAL_LIMIT:-3}"
  --save-dir "checkpoints/$RUN_NAME"
  --run-name "$RUN_NAME"
  --log-steps "${LOG_STEPS:-10}"
  --empty-cache-every "${EMPTY_CACHE_EVERY:-0}"
)

if [[ "$NPROC" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node "$NPROC" "${COMMON_ARGS[@]}"
else
  python3 "${COMMON_ARGS[@]}"
fi
