#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-artifacts/tr_hash_moe_200m_reasoning_sft_500m_full_1e}"
METRICS="${METRICS:-runs/tr-hash-moe-200m-reasoning-sft-500m-full-1e/metrics.csv}"
EVALUATION_ROOT="${EVALUATION_ROOT:-artifacts/evaluations/tr_hash_moe_200m_reasoning_sft_500m_full_1e}"
TOKENIZER="${TOKENIZER:-/workspace/tr-hash-moe-200m-reasoning-sft-500m/tokenized/tr-hash-32k-v2-2048/tokenizer}"
PIQA_PROBE="${PIQA_PROBE:-/workspace/physicaliqa-train-dev/physicaliqa-train-dev}"
ARC_PROBE="${ARC_PROBE:-/workspace/arc-evaluation-samples}"
MODEL_REPO="${MODEL_REPO:-AETHORIA-AI/TR-HASH-MoE-200M-160B-Reasoning-SFT}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/tr-hash-moe-200m-reasoning-sft-500m}"
RELEASE_ROOT="${RELEASE_ROOT:-artifacts/releases/tr_hash_moe_200m_reasoning_sft_500m}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-43200}"

if [[ -f "${EVALUATION_ROOT}/.evaluation_complete" ]]; then
  echo "[evaluation] reasoning SFT evaluation is already complete"
  exit 0
fi

started="$(date +%s)"
while [[ ! -f "${CHECKPOINT_ROOT}/.training_complete" ]]; do
  now="$(date +%s)"
  if (( now - started > WAIT_TIMEOUT_SECONDS )); then
    echo "Timed out waiting for reasoning SFT completion." >&2
    exit 2
  fi
  sleep 30
done

if [[ ! -s "${METRICS}" ]]; then
  echo "Missing completed training metrics: ${METRICS}" >&2
  exit 2
fi
if [[ ! -s "${PIQA_PROBE}/dev.jsonl" || ! -s "${PIQA_PROBE}/dev-labels.lst" ]]; then
  echo "Missing PIQA validation probe: ${PIQA_PROBE}" >&2
  exit 2
fi
if [[ ! -s "${ARC_PROBE}/samples_arc_easy.jsonl" || ! -s "${ARC_PROBE}/samples_arc_challenge.jsonl" ]]; then
  echo "Missing pinned ARC samples: ${ARC_PROBE}" >&2
  exit 2
fi

mapfile -t checkpoints < <(
  find "${CHECKPOINT_ROOT}" -mindepth 2 -maxdepth 2 -type f \
    -name checkpoint.pt -path "${CHECKPOINT_ROOT}/step_*/*" -size +0c \
    -print | sed 's#/checkpoint.pt$##' | sort -V
)
if (( ${#checkpoints[@]} < 2 )); then
  echo "Expected at least two complete step checkpoints, found ${#checkpoints[@]}." >&2
  exit 2
fi

GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( GPU_COUNT < 1 )); then
  echo "No CUDA GPU is visible for checkpoint evaluation." >&2
  exit 2
fi

mkdir -p "${EVALUATION_ROOT}"
for ((start=0; start<${#checkpoints[@]}; start+=GPU_COUNT)); do
  pids=()
  names=()
  for ((slot=0; slot<GPU_COUNT && start+slot<${#checkpoints[@]}; slot++)); do
    checkpoint="${checkpoints[$((start + slot))]}"
    name="$(basename "${checkpoint}")"
    output="${EVALUATION_ROOT}/${name}"
    mkdir -p "${output}"
    if [[ -s "${output}/piqa.json" ]]; then
      echo "[evaluation] reuse ${output}/piqa.json"
      continue
    fi
    CUDA_VISIBLE_DEVICES="${slot}" python -m scripts.eval_torch_piqa \
      "${checkpoint}" \
      --tokenizer "${TOKENIZER}" \
      --probe "${PIQA_PROBE}" \
      --batch-size "${PIQA_BATCH_SIZE:-64}" \
      --max-length 2048 \
      --dtype "${PIQA_DTYPE:-float16}" \
      --output "${output}/piqa.json" \
      > "${output}/piqa.log" 2>&1 &
    pids+=("$!")
    names+=("${name}")
  done
  status=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "PIQA evaluation failed for ${names[$index]}." >&2
      status=1
    fi
  done
  if (( status != 0 )); then
    exit "${status}"
  fi
done

python -m scripts.select_reasoning_sft_checkpoint \
  --evaluation-root "${EVALUATION_ROOT}" \
  --metrics "${METRICS}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --output "${EVALUATION_ROOT}/summary.json" \
  --selected-checkpoint "${EVALUATION_ROOT}/selected_checkpoint.txt"

selected="$(< "${EVALUATION_ROOT}/selected_checkpoint.txt")"
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_arc_generative \
  tr_hash_torch "${selected}" \
  --tokenizer "${TOKENIZER}" \
  --arc-easy-samples "${ARC_PROBE}/samples_arc_easy.jsonl" \
  --arc-challenge-samples "${ARC_PROBE}/samples_arc_challenge.jsonl" \
  --max-samples-per-task "${ARC_SAMPLES_PER_TASK:-64}" \
  --device cuda \
  --output "${EVALUATION_ROOT}/selected_arc_reasoning_64.json" \
  > "${EVALUATION_ROOT}/selected_arc_reasoning_64.log" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_torch_chat_panel \
  --checkpoint "${selected}" \
  --tokenizer "${TOKENIZER}" \
  --panel configs/tr_hash_200m_sft_v2_regression.json \
  --device cuda \
  --output "${EVALUATION_ROOT}/selected_chat_panel.json" \
  > "${EVALUATION_ROOT}/selected_chat_panel.log" 2>&1

while [[ ! -s /workspace/.hf_token ]]; do
  echo "Waiting for /workspace/.hf_token before evaluation upload..."
  sleep 60
done
export HF_TOKEN="$(< /workspace/.hf_token)"
export HF_XET_HIGH_PERFORMANCE=1
python -m scripts.upload_reasoning_sft_evaluations \
  --evaluation-root "${EVALUATION_ROOT}" \
  --repo-id "${MODEL_REPO}"

if [[ ! -s "${RELEASE_ROOT}/release_manifest.json" ]]; then
  python -m scripts.export_reasoning_sft_release \
    --summary "${EVALUATION_ROOT}/summary.json" \
    --metrics "${METRICS}" \
    --evaluation-root "${EVALUATION_ROOT}" \
    --tokenizer "${TOKENIZER}" \
    --dataset-audit "${DATASET_ROOT}/metadata/release-audit.json" \
    --output "${RELEASE_ROOT}"
fi
python -m scripts.publish_reasoning_sft_release \
  --bundle "${RELEASE_ROOT}" \
  --repo-id "${MODEL_REPO}"
touch "${EVALUATION_ROOT}/.evaluation_complete"
