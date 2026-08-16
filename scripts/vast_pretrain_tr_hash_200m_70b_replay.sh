#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/complexity-framework}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/venv/main/bin/activate}"
TOKENIZED_DATA="${TOKENIZED_DATA:-hf://datasets/Pacific-i64/data-32k-200b-tokens}"
TOKENIZED_REVISION="${TOKENIZED_REVISION:-main}"
TOKENIZED_CACHE_DIR="${TOKENIZED_CACHE_DIR:-/workspace/tr_hash_token_cache}"
TOKENIZED_PLAN="${TOKENIZED_PLAN:-artifacts/tr_hash_70b_quality_replay.json}"
QUALITY_SCORES="${QUALITY_SCORES:-}"

cd "$REPO_ROOT"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

mkdir -p "$(dirname "$TOKENIZED_PLAN")" "$TOKENIZED_CACHE_DIR"
if [[ ! -f "$TOKENIZED_PLAN" || "${REBUILD_TOKENIZED_PLAN:-0}" == "1" ]]; then
  plan_args=(
    --tokenized-data "$TOKENIZED_DATA"
    --revision "$TOKENIZED_REVISION"
    --cache-dir "$TOKENIZED_CACHE_DIR"
    --output "$TOKENIZED_PLAN"
  )
  if [[ -n "$QUALITY_SCORES" ]]; then
    plan_args+=(--quality-scores "$QUALITY_SCORES")
  fi
  python -m scripts.build_tr_hash_70b_replay_plan "${plan_args[@]}"
fi

planned_tokens="$(
  python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["trained_tokens"]))' \
    "$TOKENIZED_PLAN"
)"
if [[ -n "${TARGET_TOKENS:-}" && "$TARGET_TOKENS" != "$planned_tokens" ]]; then
  echo "TARGET_TOKENS=$TARGET_TOKENS conflicts with replay plan trained_tokens=$planned_tokens" >&2
  exit 2
fi

export TOKENIZED_DATA TOKENIZED_REVISION TOKENIZED_CACHE_DIR TOKENIZED_PLAN
export TARGET_TOKENS="$planned_tokens"
export OUTPUT_DIR="${OUTPUT_DIR:-artifacts/tr_hash_200m_70b_replay}"
export PYTHONUNBUFFERED=1

echo "[200m-70b] dataset=$TOKENIZED_DATA revision=$TOKENIZED_REVISION"
echo "[200m-70b] plan=$TOKENIZED_PLAN trained_tokens=$TARGET_TOKENS"
echo "[200m-70b] cache=$TOKENIZED_CACHE_DIR limit=${TOKENIZED_CACHE_GB:-24}GiB"

exec torchrun --standalone --nproc_per_node "${NPROC_PER_NODE:-8}" \
  -m scripts.train_tr_hash_200m_200b \
  --tokenizer "${TOKENIZER:-tokenizer}" \
  --tokenized-data "$TOKENIZED_DATA" \
  --tokenized-cache-dir "$TOKENIZED_CACHE_DIR" \
  --tokenized-cache-gb "${TOKENIZED_CACHE_GB:-24}" \
  --tokenized-revision "$TOKENIZED_REVISION" \
  --tokenized-prefetch-shards "${TOKENIZED_PREFETCH_SHARDS:-1}" \
  --tokenized-plan "$TOKENIZED_PLAN" \
  --target-tokens "$TARGET_TOKENS" \
  --token-packs "${TOKEN_PACKS:-40}" \
  --batch-size "${BATCH_SIZE_PER_GPU:-8}" \
  --seq-len "${SEQ_LEN:-1024}" \
  --gradient-accumulation "${GRADIENT_ACCUMULATION:-8}" \
  --lr "${LR:-3e-4}" \
  --warmup-tokens "${WARMUP_TOKENS:-1000000000}" \
  --lr-scheduler "${LR_SCHEDULER:-wsd}" \
  --precision bf16 \
  --no-gradient-checkpointing \
  --distributed-mode ddp \
  --optimizer "${OPTIMIZER:-adamw}" \
  --use-custom-kernels auto \
  --save-steps 0 \
  --log-steps "${LOG_STEPS:-10}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --checkpoint-dir "${OUTPUT_DIR:-artifacts/tr_hash_200m_70b_replay}" \
  "$@"
