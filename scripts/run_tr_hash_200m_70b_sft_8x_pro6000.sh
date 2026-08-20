#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT=/workspace/complexity-framework
export VENV_ACTIVATE=/venv/main/bin/activate
export INIT_CHECKPOINT=/workspace/complexity-framework/artifacts/tr_hash_200m_70b_replay/final
export OUTPUT_DIR=/workspace/complexity-framework/artifacts/tr_hash_200m_70b_unique_phase2
export NPROC_PER_NODE=8
export BATCH_SIZE_PER_GPU=96
export GRADIENT_ACCUMULATION=5
export SAVE_STEPS=17803
export HF_ENDPOINT=https://huggingface.co
unset HF_HUB_DISABLE_XET
export TOKENIZED_DATA=hf://datasets/Pacific-i64/data-32k-200b-tokens
export TOKENIZED_CACHE_DIR=/workspace/tr_hash_token_cache
export TOKENIZED_CACHE_GB=40
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COMPLEXITY_REQUIRE_LIGER=1
export PYTHONUNBUFFERED=1

if [[ -s /workspace/.hf_token ]]; then
  export HF_TOKEN
  HF_TOKEN="$(< /workspace/.hf_token)"
fi

# Start from the final pretrain weights with a fresh optimizer and step zero.
exec /bin/bash "$REPO_ROOT/scripts/vast_finetune_tr_hash_200m_70b_unique_phase2.sh" --resume ""
