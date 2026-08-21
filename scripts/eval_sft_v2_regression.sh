#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd "${ROOT:-/workspace/complexity-framework}"

: "${CHECKPOINT:?Set CHECKPOINT to the candidate checkpoint directory or checkpoint.pt}"
TOKENIZER="${TOKENIZER:-/workspace/tr-hash-moe-200m-sft-v2-300k/tokenized/tr-hash-32k-v2-2048/tokenizer}"
PIQA_PROBE="${PIQA_PROBE:-/workspace/physicaliqa-train-dev/physicaliqa-train-dev}"
PANEL="${PANEL:-configs/tr_hash_200m_sft_v2_regression.json}"
OUTPUT="${OUTPUT:-artifacts/evaluations/sft_v2_regression}"

mkdir -p "${OUTPUT}"
python -m scripts.eval_torch_chat_panel \
  --checkpoint "${CHECKPOINT}" \
  --tokenizer "${TOKENIZER}" \
  --panel "${PANEL}" \
  --device cuda \
  --output "${OUTPUT}/chat.json"

python -m scripts.eval_torch_piqa \
  "${CHECKPOINT}" \
  --tokenizer "${TOKENIZER}" \
  --probe "${PIQA_PROBE}" \
  --batch-size "${PIQA_BATCH_SIZE:-64}" \
  --max-length 2048 \
  --dtype "${PIQA_DTYPE:-float16}" \
  --output "${OUTPUT}/piqa.json"

if ! python -m scripts.check_sft_v2_regression \
  --panel "${PANEL}" \
  --chat-report "${OUTPUT}/chat.json" \
  --piqa-report "${OUTPUT}/piqa.json" \
  --output "${OUTPUT}/promotion.json"; then
  if [[ "${PROMOTION_STRICT:-true}" == "true" ]]; then
    exit 1
  fi
  echo "[evaluation] candidate failed promotion gate; report retained for comparison"
fi
