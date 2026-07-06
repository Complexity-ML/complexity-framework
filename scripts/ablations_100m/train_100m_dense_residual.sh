#!/usr/bin/env bash
set -euo pipefail

# 100M ablation: 100m_dense_residual
# Token budget: 954 steps x 8 GPUs x batch 256 x seq 2048 = 4.001B tokens.
# Override with extra CLI args after the script, e.g. --steps 10 --dataset random.

python3 scripts/train_100m_o200k_tr_local.py \
  --config configs/run_configs/ablations_100m/100m_dense_residual.yaml \
  "$@"
