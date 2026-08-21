#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

if [[ ! -s /workspace/.hf_token ]]; then
  echo "Missing /workspace/.hf_token" >&2
  exit 2
fi
export HF_TOKEN="$(< /workspace/.hf_token)"

exec python -m scripts.tokenize_luciole_16way_sft \
  --source /workspace/luciole-16way-sft \
  --tokenizer /workspace/tr-hash-refinement \
  --output /workspace/luciole-16way-tokenized/tr-hash-32k-v1 \
  --seq-len 512 \
  --min-completion-tokens 32 \
  --source-revision e685fc09503a0f45c476ab3102481e85a5f2b00d \
  --upload-repo AETHORIA-AI/luciole-16way-sft-209k
