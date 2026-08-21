#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/complexity-framework}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/artifacts/tr_hash_moe_200m_160b_luciole_16way_full_sft_3e}"
TOKENIZER="${TOKENIZER:-/workspace/tr-hash-refinement}"
PROBE="${PROBE:-/workspace/physicaliqa-train-dev/physicaliqa-train-dev}"
OUTPUT="${OUTPUT:-${ROOT}/artifacts/evaluations/full_sft_piqa}"
STEPS=(463 926 1389)

mkdir -p "${OUTPUT}"
pids=()
for index in "${!STEPS[@]}"; do
  step="${STEPS[$index]}"
  checkpoint="${CHECKPOINT_ROOT}/step_$(printf '%06d' "${step}")/checkpoint.pt"
  report="${OUTPUT}/epoch_$(printf '%02d' "$((index + 1))")_step_$(printf '%06d' "${step}").json"
  CUDA_VISIBLE_DEVICES="${index}" /venv/main/bin/python -m scripts.eval_torch_piqa \
    "${checkpoint}" \
    --tokenizer "${TOKENIZER}" \
    --probe "${PROBE}" \
    --batch-size 64 \
    --max-length 2048 \
    --dtype float16 \
    --output "${report}" \
    > "${report%.json}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
