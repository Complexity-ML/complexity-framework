#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${TR_HASH_125B_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
TOKENIZER_DIR="${TR_HASH_125B_TOKENIZER:?Set TR_HASH_125B_TOKENIZER to the pinned tokenizer snapshot}"
WORK_DIR="${TR_HASH_125B_WORK_DIR:-/workspace/builds/tr-hash-pretraining-125b-direct}"
HF_REPO="${TR_HASH_125B_HF_REPO:-AETHORIA-AI/TR-HASH-Pretraining-125B-Agentic-32K}"

CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')"
DEFAULT_RAYON_THREADS="$((CPU_COUNT - 8))"
if (( DEFAULT_RAYON_THREADS < 1 )); then DEFAULT_RAYON_THREADS=1; fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${DEFAULT_RAYON_THREADS}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TR_HASH_SOURCE_PARALLELISM="${TR_HASH_SOURCE_PARALLELISM:-6}"
export TR_HASH_STACK_DOWNLOAD_WORKERS="${TR_HASH_STACK_DOWNLOAD_WORKERS:-64}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}"

NOFILE_LIMIT="${TR_HASH_NOFILE_LIMIT:-65536}"
if ! ulimit -Sn "${NOFILE_LIMIT}"; then
  echo "Unable to raise the soft open-file limit to ${NOFILE_LIMIT}" >&2
  exit 2
fi

HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HF_HOME:-${HOME}/.cache/huggingface}/token}"
if [[ -z "${HF_TOKEN:-}" && ! -s "${HF_TOKEN_FILE}" ]]; then
  echo "HF authentication missing: set HF_TOKEN or provide ${HF_TOKEN_FILE}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m scripts.build_tr_hash_pretraining_125b \
  --config configs/agentic_pretraining/tr_hash_pretraining_125b.json \
  --curriculum configs/agentic_pretraining/tr_hash_pretraining_125b_curriculum.json \
  --tokenizer "${TOKENIZER_DIR}" \
  --work-dir "${WORK_DIR}" \
  --hf-repo "${HF_REPO}" \
  --repo-prefix "" \
  --dataset-card docs/datasets/tr-hash-pretraining-125b-source-curated.md \
  --direct-source-curated \
  --create-private-repo \
  "$@"
