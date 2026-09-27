#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export COMPLEXITY_SDPA_BACKENDS=FLASH_ATTENTION
export COMPLEXITY_REQUIRE_LIGER=1
export PYTHONUNBUFFERED=1
exec python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m scripts.pretrain_1b.train "$@"
