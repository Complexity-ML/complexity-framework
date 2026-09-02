#!/usr/bin/env bash
set -euo pipefail

# Production lineage for the 100.4M-parameter TR-HASH Agentic model:
#   prepare     -> build audited 70B-unique / 125B-trained plans
#   pretraining -> 70B unique core + 55B proportional replay
#   refinement  -> fresh optimizer, one exact pass over the same 70B core

STAGE="${1:-}"
if [[ "$STAGE" != "prepare" && "$STAGE" != "pretraining" && "$STAGE" != "refinement" ]]; then
  echo "usage: $0 {prepare|pretraining|refinement} [trainer arguments...]" >&2
  exit 2
fi
shift

REPO_ROOT="${REPO_ROOT:-/workspace/complexity-framework}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/venv/main/bin/activate}"
TOKENIZED_DATA="${TOKENIZED_DATA:-hf://datasets/AETHORIA-AI/TR-HASH-Pretraining-125B-Agentic-32K}"
TOKENIZED_REVISION="${TOKENIZED_REVISION:-fc738b3a10c5c093e3b34b48bcf1cb7066184706}"
TOKENIZED_CACHE_DIR="${TOKENIZED_CACHE_DIR:-/workspace/tr_hash_agentic_100m_token_cache}"
PRETRAIN_PLAN="${PRETRAIN_PLAN:-configs/replay_plans/tr_hash_agentic_100m_70b_unique_125b_pretrain.json}"
REFINEMENT_PLAN="${REFINEMENT_PLAN:-configs/replay_plans/tr_hash_agentic_100m_70b_refinement.json}"

cd "$REPO_ROOT"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

if [[ "$STAGE" == "prepare" ]]; then
  python -m scripts.build_tr_hash_agentic_100m_plans \
    --tokenized-data "$TOKENIZED_DATA" \
    --revision "$TOKENIZED_REVISION" \
    --cache-dir "$TOKENIZED_CACHE_DIR" \
    --unique-tokens "${UNIQUE_TOKENS:-70B}" \
    --pretrain-tokens "${PRETRAIN_TOKENS:-125B}" \
    --pretrain-output "$PRETRAIN_PLAN" \
    --refinement-output "$REFINEMENT_PLAN"
  exit 0
fi

TOKENIZER="${TOKENIZER:?Set TOKENIZER to the pinned local TR-HASH-Tokenizer-32K-Agentic snapshot}"
if [[ ! -f "$TOKENIZER/tokenizer.json" || ! -f "$TOKENIZER/chat_template.jinja" ]]; then
  echo "[error] TOKENIZER must contain tokenizer.json and chat_template.jinja: $TOKENIZER" >&2
  exit 2
fi
if [[ ! -f "$PRETRAIN_PLAN" || ! -f "$REFINEMENT_PLAN" ]]; then
  echo "[error] missing audited plans; run '$0 prepare' first" >&2
  exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"
SEQ_LEN="${SEQ_LEN:-2048}"
TOKENIZED_CACHE_GB="${TOKENIZED_CACHE_GB:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/tr_hash_agentic_100m_${STAGE}}"
RESUME="${RESUME:-auto}"

if [[ "${GRADIENT_CHECKPOINTING:-1}" == "1" ]]; then
  checkpointing_args=(--gradient-checkpointing)
else
  checkpointing_args=(--no-gradient-checkpointing)
fi

if [[ "$STAGE" == "pretraining" ]]; then
  TOKENIZED_PLAN="$PRETRAIN_PLAN"
  LR="${LR:-3e-4}"
  WARMUP_TOKENS="${WARMUP_TOKENS:-1000000000}"
  TOKEN_PACKS="${TOKEN_PACKS:-40}"
  INIT_ARGS=()
else
  TOKENIZED_PLAN="$REFINEMENT_PLAN"
  LR="${LR:-3e-5}"
  WARMUP_TOKENS="${WARMUP_TOKENS:-500000000}"
  TOKEN_PACKS="${TOKEN_PACKS:-20}"
  INIT_CHECKPOINT="${INIT_CHECKPOINT:-artifacts/tr_hash_agentic_100m_pretraining/final}"
  if [[ ! -d "$INIT_CHECKPOINT" && ! -f "$INIT_CHECKPOINT" ]]; then
    echo "[error] completed pretraining checkpoint not found: $INIT_CHECKPOINT" >&2
    exit 2
  fi
  INIT_ARGS=(--pretrain-plan "$PRETRAIN_PLAN" --init-checkpoint "$INIT_CHECKPOINT")
fi

TARGET_TOKENS="$(
  python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["trained_tokens"]))' \
    "$TOKENIZED_PLAN"
)"
TOKENS_PER_STEP=$((NPROC_PER_NODE * BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION * SEQ_LEN))
schedule_args=()
has_max_steps=0
for argument in "$@"; do
  if [[ "$argument" == "--max-steps" || "$argument" == --max-steps=* ]]; then
    has_max_steps=1
    break
  fi
done
if [[ "$has_max_steps" == "0" ]]; then
  MAX_STEPS=$((TARGET_TOKENS / TOKENS_PER_STEP))
  if [[ "$MAX_STEPS" -lt 1 ]]; then
    echo "[error] plan contains fewer tokens than one global optimizer step" >&2
    exit 2
  fi
  SCHEDULED_TOKENS=$((MAX_STEPS * TOKENS_PER_STEP))
  UNUSED_TOKENS=$((TARGET_TOKENS - SCHEDULED_TOKENS))
  schedule_args=(--max-steps "$MAX_STEPS")
  echo "[agentic-100m] exact bounded schedule=$MAX_STEPS steps trained=$SCHEDULED_TOKENS unused_tail=$UNUSED_TOKENS"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[agentic-100m] stage=$STAGE preset=complexity-100m"
echo "[agentic-100m] dataset=$TOKENIZED_DATA revision=$TOKENIZED_REVISION"
echo "[agentic-100m] plan=$TOKENIZED_PLAN target_tokens=$TARGET_TOKENS"
echo "[agentic-100m] nproc=$NPROC_PER_NODE batch/gpu=$BATCH_SIZE_PER_GPU grad_accum=$GRADIENT_ACCUMULATION seq=$SEQ_LEN"
echo "[agentic-100m] output=$OUTPUT_DIR resume=$RESUME"

trainer_command=(
  torchrun --standalone --nproc_per_node "$NPROC_PER_NODE"
  -m scripts.train_tr_hash_text_lineage
  --model-preset complexity-100m
  --stage "$STAGE"
  --tokenizer "$TOKENIZER"
  --tokenized-data "$TOKENIZED_DATA"
  --tokenized-revision "$TOKENIZED_REVISION"
  --tokenized-cache-dir "$TOKENIZED_CACHE_DIR"
  --tokenized-cache-gb "$TOKENIZED_CACHE_GB"
  --tokenized-prefetch-shards "${TOKENIZED_PREFETCH_SHARDS:-1}"
  --tokenized-plan "$TOKENIZED_PLAN"
  --target-tokens "$TARGET_TOKENS"
  --batch-size "$BATCH_SIZE_PER_GPU"
  --seq-len "$SEQ_LEN"
  --gradient-accumulation "$GRADIENT_ACCUMULATION"
  --precision bf16
  --lr "$LR"
  --warmup-tokens "$WARMUP_TOKENS"
  --lr-scheduler "${LR_SCHEDULER:-wsd}"
  --token-packs "$TOKEN_PACKS"
  --save-steps 0
  --log-steps "${LOG_STEPS:-10}"
  --num-workers 0
  --distributed-mode ddp
  --optimizer "${OPTIMIZER:-adamw}"
  --use-custom-kernels auto
  --require-cuda
  --top-k 2
  --checkpoint-dir "$OUTPUT_DIR"
  --resume "$RESUME"
)
if [[ "$has_max_steps" == "0" ]]; then
  trainer_command+=("${schedule_args[@]}")
fi
trainer_command+=("${checkpointing_args[@]}")
if [[ "$STAGE" == "refinement" ]]; then
  trainer_command+=("${INIT_ARGS[@]}")
fi
trainer_command+=("$@")
exec "${trainer_command[@]}"
