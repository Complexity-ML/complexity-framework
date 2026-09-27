#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
if [[ -e runtime/request-stop ]]; then
  echo "Graceful stop requested; remove runtime/request-stop before resuming."
  exit 0
fi
unset CUDA_LAUNCH_BLOCKING
export OMP_NUM_THREADS=4
export COMPLEXITY_SDPA_BACKENDS=FLASH_ATTENTION
export COMPLEXITY_REQUIRE_LIGER=1
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
exec "${PYTHON_BIN:-python}" -u -m torch.distributed.run --standalone --nproc_per_node=4 train.py --require-resume "$@"
