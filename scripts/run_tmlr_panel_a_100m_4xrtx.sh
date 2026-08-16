#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-/venv/main/bin/python}"
NPROC="${NPROC_PER_NODE:-4}"
MODE="${1:-tr}"

case "${MODE}" in
  tr)
    CONFIG="${REPO_ROOT}/configs/run_configs/100m_o200k_chinchilla/tr_gqa_tmlr_panel_a_seed42_1b_4xrtx.yaml"
    RUN_NAME="tr-gqa-tmlr-panel-a-100m-o200k-1b-s42"
    ;;
  dense)
    CONFIG="${REPO_ROOT}/configs/run_configs/100m_o200k_chinchilla/dense_gqa_tmlr_panel_a_seed42_1b_4xrtx.yaml"
    RUN_NAME="dense-gqa-tmlr-panel-a-100m-o200k-1b-s42"
    ;;
  *)
    echo "usage: $0 [tr|dense]" >&2
    exit 2
    ;;
esac

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/artifacts/complexity-100m-o200k}"
LOG_DIR="${ARTIFACT_ROOT}/logs"
CHECKPOINT_DIR="${ARTIFACT_ROOT}/checkpoints/${RUN_NAME}"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT:-29603}" \
  -m complexity.training.o200k_pretrain \
  --config "${CONFIG}" \
  --run-name "${RUN_NAME}" \
  --save-dir "${CHECKPOINT_DIR}" \
  2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"
