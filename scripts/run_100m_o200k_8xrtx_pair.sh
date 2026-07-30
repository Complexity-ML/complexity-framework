#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/data/fineweb_edu_o200k_4b}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/artifacts/complexity-100m-o200k}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
MODE="${1:-full}"

DENSE_CONFIG="$REPO_ROOT/configs/run_configs/100m_o200k_chinchilla/dense_gqa_seed42_2b_b200.yaml"
TR_CONFIG="$REPO_ROOT/configs/run_configs/100m_o200k_chinchilla/tr_gqa_fixed_id_seed42_2b_b200.yaml"

case "$MODE" in
  full)
    EXTRA_ARGS=()
    NAME_PREFIX=""
    ;;
  smoke)
    EXTRA_ARGS=(
      --steps 10
      --eval-steps 5
      --eval-batches 2
      --save-steps 0
    )
    NAME_PREFIX="smoke-"
    ;;
  *)
    echo "usage: $0 {smoke|full}" >&2
    exit 2
    ;;
esac

mkdir -p "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/checkpoints"
cd "$REPO_ROOT"
"$PYTHON_BIN" scripts/verify_token_shards.py "$DATA_ROOT"

run_dense() {
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29601 \
    -m complexity.training.o200k_pretrain \
    --config "$DENSE_CONFIG" \
    --tokens-path "$DATA_ROOT/train" \
    --eval-tokens-path "$DATA_ROOT/eval" \
    --expert-initialization gpt_normal \
    --run-name "${NAME_PREFIX}dense-gqa-100m-o200k-2b-s42" \
    --save-dir "$ARTIFACT_ROOT/checkpoints/${NAME_PREFIX}dense-gqa-100m-o200k-2b-s42" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$ARTIFACT_ROOT/logs/${NAME_PREFIX}dense-gqa-100m-o200k-2b-s42.log"
}

run_tr() {
  CUDA_VISIBLE_DEVICES=4,5,6,7 "$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29602 \
    -m complexity.training.o200k_pretrain \
    --config "$TR_CONFIG" \
    --tokens-path "$DATA_ROOT/train" \
    --eval-tokens-path "$DATA_ROOT/eval" \
    --expert-initialization gpt_normal \
    --run-name "${NAME_PREFIX}tr-gqa-fixed-id-100m-o200k-2b-s42" \
    --save-dir "$ARTIFACT_ROOT/checkpoints/${NAME_PREFIX}tr-gqa-fixed-id-100m-o200k-2b-s42" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$ARTIFACT_ROOT/logs/${NAME_PREFIX}tr-gqa-fixed-id-100m-o200k-2b-s42.log"
}

run_dense &
DENSE_PID=$!
run_tr &
TR_PID=$!

status=0
wait "$DENSE_PID" || status=$?
wait "$TR_PID" || status=$?
exit "$status"
