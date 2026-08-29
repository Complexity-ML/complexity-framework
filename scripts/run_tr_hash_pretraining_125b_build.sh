#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${TR_HASH_125B_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
TOKENIZER_DIR="${TR_HASH_125B_TOKENIZER:?Set TR_HASH_125B_TOKENIZER to the pinned tokenizer snapshot}"
WORK_DIR="${TR_HASH_125B_WORK_DIR:-/workspace/builds/tr-hash-pretraining-125b}"
HF_REPO="${TR_HASH_125B_HF_REPO:-AETHORIA-AI/TR-HASH-Pretraining-125B-Agentic-32K}"
CANDIDATE_DIR="${TR_HASH_125B_CANDIDATE_DIR:-${WORK_DIR}/candidates}"
PACK_DIR="${TR_HASH_125B_PACK_DIR:-${WORK_DIR}/final}"
CANDIDATE_CACHE_DIR="${TR_HASH_125B_CANDIDATE_CACHE_DIR:-${WORK_DIR}/candidate-cache}"
SOURCE_WORKERS="${TR_HASH_125B_SOURCE_WORKERS:-12}"

CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')"
DEFAULT_RAYON_THREADS="$((CPU_COUNT / SOURCE_WORKERS))"
if (( DEFAULT_RAYON_THREADS < 1 )); then DEFAULT_RAYON_THREADS=1; fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${DEFAULT_RAYON_THREADS}}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}"

HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HF_HOME:-${HOME}/.cache/huggingface}/token}"
if [[ -z "${HF_TOKEN:-}" && ! -s "${HF_TOKEN_FILE}" ]]; then
  echo "HF authentication missing: set HF_TOKEN or provide ${HF_TOKEN_FILE}" >&2
  exit 2
fi

config="${REPO_ROOT}/configs/agentic_pretraining/tr_hash_pretraining_125b.json"
curriculum="${REPO_ROOT}/configs/agentic_pretraining/tr_hash_pretraining_125b_curriculum.json"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m scripts.stage_tr_hash_pretraining_125b_candidates \
  --config "${config}" \
  --tokenizer "${TOKENIZER_DIR}" \
  --work-dir "${CANDIDATE_DIR}" \
  --hf-repo "${HF_REPO}" \
  --source-workers "${SOURCE_WORKERS}" \
  --rayon-threads-per-source "${RAYON_NUM_THREADS}"

if [[ "${TR_HASH_125B_STAGE_ONLY:-0}" == "1" ]]; then
  exit 0
fi

exec "${PYTHON_BIN}" -m scripts.pack_tr_hash_pretraining_125b_candidates \
  --config "${config}" \
  --curriculum "${curriculum}" \
  --candidate-manifest "${CANDIDATE_DIR}/.metadata/_candidates/manifest.json" \
  --tokenizer "${TOKENIZER_DIR}" \
  --work-dir "${PACK_DIR}" \
  --candidate-cache-dir "${CANDIDATE_CACHE_DIR}" \
  --hf-repo "${HF_REPO}" \
  --repo-prefix production \
  "$@"
