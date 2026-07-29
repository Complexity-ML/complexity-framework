#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/Users/boris/Dev/fineweb_edu_o200k_4b}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/mps-100m-o200k-diagnostic}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-600}"
EVAL_STEPS="${EVAL_STEPS:-100}"
EVAL_BATCHES="${EVAL_BATCHES:-4}"

DENSE_CONFIG="$REPO_ROOT/configs/run_configs/diagnostics_100m/dense_gqa_o200k_mps_10min.yaml"
CYCLIC_CONFIG="$REPO_ROOT/configs/run_configs/diagnostics_100m/tr_gqa_modulo_cyclic_o200k_mps_10min.yaml"

mkdir -p "$ARTIFACT_ROOT/logs"
cd "$REPO_ROOT"

run_one() {
  local config="$1"
  local run_name="$2"
  PYTORCH_ENABLE_MPS_FALLBACK=1 "$PYTHON_BIN" \
    -m complexity.training.o200k_pretrain \
    --config "$config" \
    --tokens-path "$DATA_ROOT/train" \
    --eval-tokens-path "$DATA_ROOT/eval" \
    --steps "$STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --eval-batches "$EVAL_BATCHES" \
    --save-steps 0 \
    --run-name "$run_name" \
    2>&1 | tee "$ARTIFACT_ROOT/logs/$run_name.log"
}

case "${1:-all}" in
  dense)
    run_one "$DENSE_CONFIG" "diagnostic-mps-dense-gqa-o200k-s42"
    ;;
  cyclic)
    run_one "$CYCLIC_CONFIG" "diagnostic-mps-tr-gqa-modulo-cyclic-o200k-s42"
    ;;
  all)
    run_one "$DENSE_CONFIG" "diagnostic-mps-dense-gqa-o200k-s42"
    run_one "$CYCLIC_CONFIG" "diagnostic-mps-tr-gqa-modulo-cyclic-o200k-s42"
    ;;
  *)
    echo "usage: $0 {dense|cyclic|all}" >&2
    exit 2
    ;;
esac
