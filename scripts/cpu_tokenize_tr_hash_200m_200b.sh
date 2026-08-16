#!/usr/bin/env bash
set -euo pipefail

# 360-vCPU / 600-GiB materialization job. Authentication is read by
# huggingface_hub from HF_TOKEN or the local `hf auth login` cache.
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-320}"
export HF_DATASET_REPO="${HF_DATASET_REPO:-Pacific-i64/data-32k-200b-tokens}"

exec python -m scripts.tokenize_tr_hash_200m_200b \
  --tokenizer "${TOKENIZER:-tokenizer}" \
  --output "${OUTPUT_DIR:-artifacts/tr_hash_200m_200b_tokens}" \
  --target-tokens "${TARGET_TOKENS:-200000000000}" \
  --seq-len "${SEQ_LEN:-1024}" \
  --global-batch-sequences "${GLOBAL_BATCH_SEQUENCES:-512}" \
  --shard-trained-tokens "${SHARD_TRAINED_TOKENS:-1000000000}" \
  --document-batch-size "${DOCUMENT_BATCH_SIZE:-4096}" \
  --parallel-corpora "${PARALLEL_CORPORA:-3}" \
  --stack-download-workers "${STACK_DOWNLOAD_WORKERS:-256}" \
  --hf-repo "$HF_DATASET_REPO" \
  --hf-upload-workers "${HF_UPLOAD_WORKERS:-64}" \
  "$@"
