#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-artifacts/tr_hash_moe_200m_clean_sft_v2_full_3e}"
METRICS="${METRICS:-runs/tr-hash-moe-200m-clean-sft-v2-full-3e/metrics.csv}"
EVALUATION_ROOT="${EVALUATION_ROOT:-artifacts/evaluations/tr_hash_moe_200m_clean_sft_v2_full_3e}"
RELEASE_ROOT="${RELEASE_ROOT:-artifacts/releases/tr_hash_moe_200m_clean_sft_v2}"
TOKENIZER="${TOKENIZER:-/workspace/tr-hash-moe-200m-sft-v2-300k/tokenized/tr-hash-32k-v2-2048/tokenizer}"
PIQA_PROBE="${PIQA_PROBE:-/workspace/physicaliqa-train-dev/physicaliqa-train-dev}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-43200}"

started="$(date +%s)"
while [[ ! -f "${CHECKPOINT_ROOT}/.training_complete" ]]; do
  now="$(date +%s)"
  if (( now - started > WAIT_TIMEOUT_SECONDS )); then
    echo "Timed out waiting for clean SFT v2 training completion." >&2
    exit 2
  fi
  sleep 30
done

mapfile -t checkpoints < <(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step_*' | sort)
if (( ${#checkpoints[@]} != 3 )); then
  echo "Expected exactly 3 epoch checkpoints, found ${#checkpoints[@]}: ${checkpoints[*]-}" >&2
  exit 2
fi
if [[ ! -s "${METRICS}" ]]; then
  echo "Missing completed training metrics: ${METRICS}" >&2
  exit 2
fi

mkdir -p "${EVALUATION_ROOT}"
pids=()
for index in "${!checkpoints[@]}"; do
  checkpoint="${checkpoints[$index]}"
  step="${checkpoint##*_}"
  epoch="$((index + 1))"
  output="${EVALUATION_ROOT}/epoch_$(printf '%02d' "${epoch}")_step_${step}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${index}" \
    CHECKPOINT="${checkpoint}" \
    TOKENIZER="${TOKENIZER}" \
    PIQA_PROBE="${PIQA_PROBE}" \
    OUTPUT="${output}" \
    PROMOTION_STRICT=false \
    bash scripts/eval_sft_v2_regression.sh \
    > "${output}/evaluation.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "At least one checkpoint evaluation failed to execute; see evaluation.log files." >&2
  exit "${status}"
fi

selection_status=0
python -m scripts.select_sft_v2_checkpoint \
  --evaluation-root "${EVALUATION_ROOT}" \
  --metrics "${METRICS}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --output "${EVALUATION_ROOT}/summary.json" \
  --selected-checkpoint "${EVALUATION_ROOT}/selected_checkpoint.txt" \
  || selection_status="$?"

HF_TOKEN="$(< /workspace/.hf_token)" python -m scripts.upload_sft_v2_evaluations \
  --evaluation-root "${EVALUATION_ROOT}" \
  --metrics "${METRICS}" \
  --panel configs/tr_hash_200m_sft_v2_regression.json \
  --repo-id AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT

if (( selection_status != 0 )); then
  exit "${selection_status}"
fi

python -m scripts.export_sft_v2_release \
  --summary "${EVALUATION_ROOT}/summary.json" \
  --metrics "${METRICS}" \
  --evaluation-root "${EVALUATION_ROOT}" \
  --tokenizer "${TOKENIZER}" \
  --output "${RELEASE_ROOT}"
HF_TOKEN="$(< /workspace/.hf_token)" python -m scripts.publish_sft_v2_release \
  --bundle "${RELEASE_ROOT}" \
  --repo-id AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT
HF_TOKEN="$(< /workspace/.hf_token)" python -m scripts.cleanup_sft_v2_legacy \
  --repo-id AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT \
  --summary "${EVALUATION_ROOT}/summary.json" \
  --execute
touch "${EVALUATION_ROOT}/.evaluation_complete"
