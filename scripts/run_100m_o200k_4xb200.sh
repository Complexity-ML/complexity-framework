#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/data/fineweb_edu_o200k_4b}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/artifacts/complexity-100m-o200k}"
NPROC="${NPROC:-4}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /venv/main/bin/python ]]; then
    PYTHON_BIN=/venv/main/bin/python
  else
    PYTHON_BIN=python3
  fi
fi

DENSE_CONFIG="$REPO_ROOT/configs/run_configs/100m_o200k_chinchilla/dense_gqa_seed42_2b_b200.yaml"
TR_CONFIG="$REPO_ROOT/configs/run_configs/100m_o200k_chinchilla/tr_gqa_fixed_id_seed42_2b_b200.yaml"

mkdir -p "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/checkpoints"
cd "$REPO_ROOT"

verify_dataset() {
  "$PYTHON_BIN" scripts/verify_token_shards.py "$DATA_ROOT"
}

launch() {
  local config="$1"
  local run_name="$2"
  local save_name="$3"
  shift 3
  verify_dataset
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    -m complexity.training.o200k_pretrain \
    --config "$config" \
    --tokens-path "$DATA_ROOT/train" \
    --eval-tokens-path "$DATA_ROOT/eval" \
    --run-name "$run_name" \
    --save-dir "$ARTIFACT_ROOT/checkpoints/$save_name" \
    "$@" \
    2>&1 | tee "$ARTIFACT_ROOT/logs/$save_name.log"
}

case "${1:-}" in
  smoke-dense)
    launch \
      "$DENSE_CONFIG" \
      "smoke-dense-gqa-100m-o200k-2b-s42" \
      "smoke-dense-gqa-100m-o200k-2b-s42" \
      --steps 10 --eval-steps 5 --eval-batches 2 --save-steps 0
    ;;
  dense)
    launch \
      "$DENSE_CONFIG" \
      "dense-gqa-100m-o200k-2b-s42" \
      "dense-gqa-100m-o200k-2b-s42"
    ;;
  tr)
    launch \
      "$TR_CONFIG" \
      "tr-gqa-fixed-id-100m-o200k-2b-s42" \
      "tr-gqa-fixed-id-100m-o200k-2b-s42"
    ;;
  *)
    echo "usage: $0 {smoke-dense|dense|tr}" >&2
    exit 2
    ;;
esac
