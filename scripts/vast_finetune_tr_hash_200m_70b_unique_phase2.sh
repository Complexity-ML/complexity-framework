#!/usr/bin/env bash
set -euo pipefail

# Phase 2: full-parameter text refinement of a completed TR-Hash 200M pretrain.
# Text analogue of scripts/vast_finetune_detector_coco_v08_nano.sh's vision
# SFT stage: reruns the SAME corpus the model already saw in phase 1 --
# configs/replay_plans/tr_hash_70b_unique_only_phase2.json selects the exact
# same 70B unique shards as the pretrain's unique_core phase, replay_passes=1
# for every source (no repetition) -- with a fresh optimizer/scheduler and a
# low re-warmed LR, instead of resuming. --init-checkpoint loads only the
# phase-1 final weights (see CheckpointManager.load_weights_only); step,
# optimizer, and scheduler all start clean.
#
# finetuning.validate_full_parameter_finetuning enforces that the plan's
# unique_tokens exactly matches the pretrain's (69,997,690,880) before this
# is allowed to run full-parameter at all -- see scripts/train_tr_hash_200m_200b.py.

REPO_ROOT="${REPO_ROOT:-/workspace/complexity-framework}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/venv/main/bin/activate}"
TOKENIZED_DATA="${TOKENIZED_DATA:-hf://datasets/Pacific-i64/data-32k-200b-tokens}"
TOKENIZED_REVISION="${TOKENIZED_REVISION:-main}"
TOKENIZED_CACHE_DIR="${TOKENIZED_CACHE_DIR:-/workspace/tr_hash_token_cache}"
TOKENIZED_PLAN="${TOKENIZED_PLAN:-configs/replay_plans/tr_hash_70b_unique_only_phase2.json}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-artifacts/tr_hash_200m_70b_replay/final}"

cd "$REPO_ROOT"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

if [[ ! -f "$TOKENIZED_PLAN" ]]; then
  echo "[error] missing $TOKENIZED_PLAN -- this is a committed, pre-built plan, not rebuilt here" >&2
  exit 2
fi
if [[ "${RESUME:-auto}" == "auto" && ! -f "$INIT_CHECKPOINT/model.safetensors" ]]; then
  echo "[error] missing $INIT_CHECKPOINT/model.safetensors -- phase 1 must finish and export 'final' first" >&2
  exit 2
fi

planned_tokens="$(
  python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["trained_tokens"]))' \
    "$TOKENIZED_PLAN"
)"
if [[ -n "${TARGET_TOKENS:-}" && "$TARGET_TOKENS" != "$planned_tokens" ]]; then
  echo "TARGET_TOKENS=$TARGET_TOKENS conflicts with plan trained_tokens=$planned_tokens" >&2
  exit 2
fi

export TOKENIZED_DATA TOKENIZED_REVISION TOKENIZED_CACHE_DIR TOKENIZED_PLAN
export TARGET_TOKENS="$planned_tokens"
export OUTPUT_DIR="${OUTPUT_DIR:-artifacts/tr_hash_200m_70b_unique_phase2}"
export PYTHONUNBUFFERED=1

echo "[200m-phase2] plan=$TOKENIZED_PLAN trained_tokens=$TARGET_TOKENS (single pass, no replay)"
echo "[200m-phase2] init_checkpoint=$INIT_CHECKPOINT"
echo "[200m-phase2] batch_size=${BATCH_SIZE_PER_GPU:-32} gradient_accumulation=${GRADIENT_ACCUMULATION:-1}"
echo "[200m-phase2] lr=${LR:-1e-4} scheduler=${LR_SCHEDULER:-cosine}"

set +e
torchrun --standalone --nproc_per_node "${NPROC_PER_NODE:-8}" \
  -m scripts.train_tr_hash_200m_200b \
  --tokenizer "${TOKENIZER:-tokenizer}" \
  --tokenized-data "$TOKENIZED_DATA" \
  --tokenized-cache-dir "$TOKENIZED_CACHE_DIR" \
  --tokenized-cache-gb "${TOKENIZED_CACHE_GB:-24}" \
  --tokenized-revision "$TOKENIZED_REVISION" \
  --tokenized-prefetch-shards "${TOKENIZED_PREFETCH_SHARDS:-1}" \
  --tokenized-plan "$TOKENIZED_PLAN" \
  --target-tokens "$TARGET_TOKENS" \
  --token-packs "${TOKEN_PACKS:-10}" \
  --batch-size "${BATCH_SIZE_PER_GPU:-32}" \
  --seq-len "${SEQ_LEN:-1024}" \
  --gradient-accumulation "${GRADIENT_ACCUMULATION:-1}" \
  --lr "${LR:-1e-4}" \
  --warmup-tokens "${WARMUP_TOKENS:-500000000}" \
  --lr-scheduler "${LR_SCHEDULER:-cosine}" \
  --precision bf16 \
  --no-gradient-checkpointing \
  --distributed-mode ddp \
  --optimizer "${OPTIMIZER:-adamw}" \
  --use-custom-kernels auto \
  --save-steps 0 \
  --log-steps "${LOG_STEPS:-10}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --checkpoint-dir "$OUTPUT_DIR" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --resume "${RESUME:-auto}" \
  "$@"
training_status=$?
set -e

exit "$training_status"
