#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/data/fineweb_edu_32k_4b}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/artifacts/complexity-200m-32k}"
NPROC="${NPROC:-4}"
REVISION="${FINEWEB_REVISION:-87f09149ef4734204d70ed1d046ddc9ca3f2b8f9}"

TRAIN_TOKENS=3999793153
EVAL_TOKENS=16777217
DENSE_CONFIG="$REPO_ROOT/configs/run_configs/200m_32k_chinchilla/dense_gqa_seed42_4b_b200.yaml"
TR_CONFIG="$REPO_ROOT/configs/run_configs/200m_32k_chinchilla/tr_gqa_fixed_id_seed42_4b_b200.yaml"

mkdir -p "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/checkpoints"
cd "$REPO_ROOT"

prepare_dataset() {
  python3 scripts/prepare_fineweb_32k_shards.py \
    --output-root "$DATA_ROOT" \
    --tokenizer "$REPO_ROOT/tokenizer" \
    --revision "$REVISION" \
    --train-tokens "$TRAIN_TOKENS" \
    --eval-tokens "$EVAL_TOKENS"
}

verify_dataset() {
  python3 scripts/verify_token_shards.py "$DATA_ROOT"
}

run_dense() {
  verify_dataset
  torchrun --standalone --nproc_per_node="$NPROC" \
    -m complexity.training.o200k_pretrain \
    --config "$DENSE_CONFIG" \
    --tokens-path "$DATA_ROOT/train" \
    --eval-tokens-path "$DATA_ROOT/eval" \
    --save-dir "$ARTIFACT_ROOT/checkpoints/dense-gqa-200m-32k-4b-s42" \
    2>&1 | tee "$ARTIFACT_ROOT/logs/dense-gqa-200m-32k-4b-s42.log"
}

run_tr() {
  verify_dataset
  torchrun --standalone --nproc_per_node="$NPROC" \
    -m complexity.training.o200k_pretrain \
    --config "$TR_CONFIG" \
    --tokens-path "$DATA_ROOT/train" \
    --eval-tokens-path "$DATA_ROOT/eval" \
    --save-dir "$ARTIFACT_ROOT/checkpoints/tr-gqa-fixed-id-200m-32k-4b-s42" \
    2>&1 | tee "$ARTIFACT_ROOT/logs/tr-gqa-fixed-id-200m-32k-4b-s42.log"
}

smoke_dense() {
  verify_dataset
  torchrun --standalone --nproc_per_node="$NPROC" \
    -m complexity.training.o200k_pretrain \
    --config "$DENSE_CONFIG" \
    --tokens-path "$DATA_ROOT/train" \
    --eval-tokens-path "$DATA_ROOT/eval" \
    --steps 10 \
    --eval-steps 5 \
    --eval-batches 2 \
    --save-steps 0 \
    --run-name smoke-dense-gqa-200m-32k-4b-s42 \
    --save-dir "$ARTIFACT_ROOT/checkpoints/smoke-dense-gqa-200m-32k-4b-s42" \
    2>&1 | tee "$ARTIFACT_ROOT/logs/smoke-dense-gqa-200m-32k-4b-s42.log"
}

collect_run() {
  local run_name="$1"
  local checkpoint_name="$2"
  local log_name="$3"
  python3 scripts/collect_200m_run_artifacts.py \
    --run-name "$run_name" \
    --checkpoint-root "$ARTIFACT_ROOT/checkpoints/$checkpoint_name/latest" \
    --run-dir "$REPO_ROOT/runs/$run_name" \
    --data-root "$DATA_ROOT" \
    --tokenizer "$REPO_ROOT/tokenizer" \
    --log "$ARTIFACT_ROOT/logs/$log_name.log" \
    --output-dir "$ARTIFACT_ROOT/releases/$run_name"
}

case "${1:-}" in
  prepare)
    prepare_dataset
    ;;
  verify)
    verify_dataset
    ;;
  smoke-dense)
    smoke_dense
    ;;
  dense)
    run_dense
    ;;
  tr)
    run_tr
    ;;
  collect-dense)
    collect_run \
      "dense-gqa-200m-32k-4b-s42" \
      "dense-gqa-200m-32k-4b-s42" \
      "dense-gqa-200m-32k-4b-s42"
    ;;
  collect-tr)
    collect_run \
      "tr-gqa-fixed-id-200m-32k-4b-s42" \
      "tr-gqa-fixed-id-200m-32k-4b-s42" \
      "tr-gqa-fixed-id-200m-32k-4b-s42"
    ;;
  *)
    echo "usage: $0 {prepare|verify|smoke-dense|dense|tr|collect-dense|collect-tr}" >&2
    exit 2
    ;;
esac
