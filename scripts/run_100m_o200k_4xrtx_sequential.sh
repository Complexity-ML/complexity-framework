#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/data/fineweb_edu_o200k_4b}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/artifacts/complexity-100m-o200k}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
MODE="${1:-pair}"

DENSE_CONFIG="$REPO_ROOT/configs/run_configs/100m_o200k_chinchilla/dense_gqa_seed42_2b_b200.yaml"
TR_CONFIG="$REPO_ROOT/configs/run_configs/100m_o200k_chinchilla/tr_gqa_expert_lr2_seed42_2b_b200.yaml"

FULL_ARGS=(
  --batch-size 24
  --steps 10172
  --eval-steps 500
  --eval-batches 64
  --save-steps 2000
)
SMOKE_ARGS=(
  --batch-size 24
  --steps 10
  --eval-steps 5
  --eval-batches 2
  --save-steps 0
)

mkdir -p "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/checkpoints"
cd "$REPO_ROOT"
"$PYTHON_BIN" scripts/verify_token_shards.py "$DATA_ROOT"

run_model() {
  local architecture="$1"
  local run_mode="$2"
  local config run_name master_port
  local -a run_args

  case "$architecture" in
    dense)
      config="$DENSE_CONFIG"
      run_name="dense-gqa-100m-o200k-2b-s42-b24"
      master_port=29601
      ;;
    tr)
      config="$TR_CONFIG"
      run_name="tr-gqa-expert-lr2-100m-o200k-2b-s42-b24"
      master_port=29602
      ;;
    *)
      echo "unknown architecture: $architecture" >&2
      exit 2
      ;;
  esac

  case "$run_mode" in
    full)
      run_args=("${FULL_ARGS[@]}")
      ;;
    smoke)
      run_args=("${SMOKE_ARGS[@]}")
      run_name="smoke-${run_name}"
      ;;
    *)
      echo "unknown run mode: $run_mode" >&2
      exit 2
      ;;
  esac

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port="$master_port" \
    -m complexity.training.o200k_pretrain \
    --config "$config" \
    --tokens-path "$DATA_ROOT/train" \
    --eval-tokens-path "$DATA_ROOT/eval" \
    --run-name "$run_name" \
    --save-dir "$ARTIFACT_ROOT/checkpoints/$run_name" \
    "${run_args[@]}" \
    2>&1 | tee "$ARTIFACT_ROOT/logs/$run_name.log"
}

case "$MODE" in
  dense)
    run_model dense full
    ;;
  tr)
    run_model tr full
    ;;
  smoke-dense)
    run_model dense smoke
    ;;
  smoke-tr)
    run_model tr smoke
    ;;
  pair)
    run_model dense full
    run_model tr full
    ;;
  *)
    echo "usage: $0 {dense|tr|smoke-dense|smoke-tr|pair}" >&2
    exit 2
    ;;
esac
