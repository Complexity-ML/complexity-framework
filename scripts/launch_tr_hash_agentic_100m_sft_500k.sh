#!/usr/bin/env bash
set -euo pipefail

# Three-epoch full-parameter SFT from the final 100M Agentic refinement.
# Public Hub artifacts are pinned so a fresh GPU host can launch reproducibly.

WORKSPACE="${WORKSPACE:-/workspace}"
MODEL_REPO="AETHORIA-AI/TR-HASH-MoE-100M-70B-Agentic-Refinement"
MODEL_REVISION="99fee390916154ec9d8c0f049c3a890f81414e20"
MODEL_SUBDIR="token_pack_014_213622"
DATA_REPO="AETHORIA-AI/TR-HASH-Agentic-SFT-32K-500K"
DATA_REVISION="72d2cdc1a4a32a2db17ee7a22d921b07b0026042"
TOKENIZER_REPO="AETHORIA-AI/TR-HASH-Tokenizer-32K-Agentic"
TOKENIZER_REVISION="2fcbc2c5359ded0244ca14531f1b3806eebac55e"

MODEL_DIR="${MODEL_DIR:-$WORKSPACE/models/tr_hash_100m_agentic_refinement}"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data/tr_hash_agentic_sft_500k}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$WORKSPACE/models/tr_hash_tokenizer_32k_agentic}"
SAVE_DIR="${SAVE_DIR:-$WORKSPACE/artifacts/tr_hash_agentic_100m_sft_500k_3epochs}"
RUN_NAME="${RUN_NAME:-tr-hash-100m-agentic-sft-500k-3epochs}"
NPROC="${NPROC:-$(nvidia-smi -L | wc -l | tr -d ' ')}"

if [[ ! -f "$MODEL_DIR/$MODEL_SUBDIR/checkpoint.pt" ]]; then
  hf download "$MODEL_REPO" \
    --revision "$MODEL_REVISION" \
    --include "$MODEL_SUBDIR/checkpoint.pt" \
    --local-dir "$MODEL_DIR"
fi
if [[ ! -f "$DATA_DIR/tokenized/tr-hash-agentic-32k-2048/manifest.json" ]]; then
  hf download "$DATA_REPO" \
    --repo-type dataset \
    --revision "$DATA_REVISION" \
    --include "tokenized/tr-hash-agentic-32k-2048/**" \
    --local-dir "$DATA_DIR"
fi
if [[ ! -f "$TOKENIZER_DIR/tokenizer.json" ]]; then
  hf download "$TOKENIZER_REPO" \
    --revision "$TOKENIZER_REVISION" \
    --include "tokenizer.json" \
    --local-dir "$TOKENIZER_DIR"
fi

if [[ "$NPROC" -lt 1 ]]; then
  echo "No NVIDIA GPU detected" >&2
  exit 1
fi

COMMON_ARGS=(
  -m scripts.sft_500m_32k_tr
  --checkpoint "$MODEL_DIR/$MODEL_SUBDIR"
  --source-stage refinement
  --tokenizer "$TOKENIZER_DIR/tokenizer.json"
  --sft-bin "$DATA_DIR/tokenized/tr-hash-agentic-32k-2048"
  --require-release-ready
  --seq-len 2048
  --steps 0
  --epochs 3
  --batch-size "${BATCH_SIZE:-16}"
  --pack-sequences
  --lr "${LR:-1e-5}"
  --weight-decay 0.0
  --beta1 0.9
  --beta2 0.95
  --warmup-ratio 0.03
  --full-parameter
  --bf16
  --use-custom-kernels auto
  --loss-chunk-tokens 1024
  --sft-fp32-loss
  --log-steps 10
  --eval-batches 0
  --min-eval-fraction 0.05
  --eval-at-start
  --save-best
  --early-stopping-patience 0
  --save-every-epoch
  --eval-every-epoch
  --no-reset-lr-each-epoch
  --save-dir "$SAVE_DIR"
  --save-total-limit 4
  --run-name "$RUN_NAME"
  --tensorboard-dir "$WORKSPACE/runs/$RUN_NAME/tensorboard"
  --seed 42
  --num-workers 0
  --empty-cache-every 0
)

if [[ "$NPROC" -gt 1 ]]; then
  exec torchrun --standalone --nproc_per_node "$NPROC" "${COMMON_ARGS[@]}"
fi
exec python3 "${COMMON_ARGS[@]}"
