#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-artifacts/tr_hash_moe_200m_reasoning_preservation_50m_full_1e}"
METRICS="${METRICS:-runs/tr-hash-moe-200m-reasoning-preservation-50m-full-1e/metrics.csv}"
EVALUATION_ROOT="${EVALUATION_ROOT:-artifacts/evaluations/tr_hash_moe_200m_reasoning_preservation_50m_full_1e}"
SOURCE_MODEL="${SOURCE_MODEL:-/workspace/tr-hash-sft-v2}"
TOKENIZER="${TOKENIZER:-/workspace/tr-hash-sft-v2}"
PIQA_PROBE="${PIQA_PROBE:-/workspace/physicaliqa-train-dev/physicaliqa-train-dev}"
ARC_PROBE="${ARC_PROBE:-/workspace/arc-evaluation-samples}"
PANEL="${PANEL:-configs/tr_hash_200m_reasoning_preservation_regression.json}"
MODEL_REPO="${MODEL_REPO:-AETHORIA-AI/TR-HASH-MoE-200M-160B-Reasoning-Preservation-50M}"
RELEASE_ROOT="${RELEASE_ROOT:-artifacts/releases/tr_hash_moe_200m_reasoning_preservation_50m}"

[[ -f "${CHECKPOINT_ROOT}/.training_complete" ]] || { echo "Training is not complete." >&2; exit 2; }
[[ -s "${METRICS}" ]] || { echo "Missing metrics: ${METRICS}" >&2; exit 2; }
[[ -s "${SOURCE_MODEL}/model.safetensors" ]] || { echo "Missing source model." >&2; exit 2; }
[[ -s "${PIQA_PROBE}/dev.jsonl" && -s "${PIQA_PROBE}/dev-labels.lst" ]] || {
  echo "Missing PIQA probe." >&2; exit 2;
}
[[ -s "${ARC_PROBE}/samples_arc_easy.jsonl" && -s "${ARC_PROBE}/samples_arc_challenge.jsonl" ]] || {
  echo "Missing ARC probe." >&2; exit 2;
}

mapfile -t checkpoints < <(
  find "${CHECKPOINT_ROOT}" -mindepth 2 -maxdepth 2 -type f \
    -name checkpoint.pt -path "${CHECKPOINT_ROOT}/step_*/*" -size +0c \
    -print | sed 's#/checkpoint.pt$##' | sort -V
)
(( ${#checkpoints[@]} >= 2 )) || { echo "Not enough checkpoints." >&2; exit 2; }
GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
(( GPU_COUNT >= 1 )) || { echo "No CUDA GPU visible." >&2; exit 2; }
mkdir -p "${EVALUATION_ROOT}"

names=(source_sft_v2)
paths=("${SOURCE_MODEL}")
for checkpoint in "${checkpoints[@]}"; do
  names+=("$(basename "${checkpoint}")")
  paths+=("${checkpoint}")
done

evaluate_candidate() {
  local slot="$1" name="$2" checkpoint="$3" output="${EVALUATION_ROOT}/${name}"
  mkdir -p "${output}"
  if [[ ! -s "${output}/piqa.json" ]]; then
    CUDA_VISIBLE_DEVICES="${slot}" python -m scripts.eval_torch_piqa "${checkpoint}" \
      --tokenizer "${TOKENIZER}" --probe "${PIQA_PROBE}" --batch-size 64 \
      --max-length 2048 --dtype float16 --output "${output}/piqa.json" \
      > "${output}/piqa.log" 2>&1
  fi
  if [[ ! -s "${output}/chat.json" ]]; then
    CUDA_VISIBLE_DEVICES="${slot}" python -m scripts.eval_torch_chat_panel \
      --checkpoint "${checkpoint}" --tokenizer "${TOKENIZER}" --panel "${PANEL}" \
      --device cuda --output "${output}/chat.json" > "${output}/chat.log" 2>&1
  fi
  if [[ ! -s "${output}/promotion.json" ]]; then
    python -m scripts.check_sft_v2_regression --panel "${PANEL}" \
      --chat-report "${output}/chat.json" --piqa-report "${output}/piqa.json" \
      --output "${output}/promotion.json" > "${output}/promotion.log" 2>&1 || true
  fi
  if [[ ! -s "${output}/arc_zero_shot.json" ]]; then
    CUDA_VISIBLE_DEVICES="${slot}" python -m scripts.eval_torch_arc_zero_shot "${checkpoint}" \
      --tokenizer "${TOKENIZER}" \
      --arc-easy-samples "${ARC_PROBE}/samples_arc_easy.jsonl" \
      --arc-challenge-samples "${ARC_PROBE}/samples_arc_challenge.jsonl" \
      --batch-size 64 --dtype float16 --output "${output}/arc_zero_shot.json" \
      > "${output}/arc_zero_shot.log" 2>&1
  fi
  if [[ ! -s "${output}/arc_reasoning_64.json" ]]; then
    CUDA_VISIBLE_DEVICES="${slot}" python -m scripts.eval_arc_generative \
      tr_hash_torch "${checkpoint}" --tokenizer "${TOKENIZER}" \
      --arc-easy-samples "${ARC_PROBE}/samples_arc_easy.jsonl" \
      --arc-challenge-samples "${ARC_PROBE}/samples_arc_challenge.jsonl" \
      --max-samples-per-task 32 --prompt-style bare --device cuda \
      --output "${output}/arc_reasoning_64.json" \
      > "${output}/arc_reasoning_64.log" 2>&1
  fi
  touch "${output}/.complete"
}

for ((start=0; start<${#paths[@]}; start+=GPU_COUNT)); do
  pids=(); batch_names=()
  for ((slot=0; slot<GPU_COUNT && start+slot<${#paths[@]}; slot++)); do
    index="$((start + slot))"
    evaluate_candidate "${slot}" "${names[$index]}" "${paths[$index]}" &
    pids+=("$!"); batch_names+=("${names[$index]}")
  done
  status=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "Evaluation failed for ${batch_names[$index]}." >&2
      status=1
    fi
  done
  (( status == 0 )) || exit "${status}"
done

selection_status=0
python -m scripts.select_reasoning_preservation_checkpoint \
  --evaluation-root "${EVALUATION_ROOT}" --metrics "${METRICS}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" --panel "${PANEL}" \
  --output "${EVALUATION_ROOT}/summary.json" \
  --selected-checkpoint "${EVALUATION_ROOT}/selected_checkpoint.txt" \
  > "${EVALUATION_ROOT}/selection.log" 2>&1 || selection_status="$?"

export HF_TOKEN="$(< /workspace/.hf_token)"
export HF_XET_HIGH_PERFORMANCE=1
python -m scripts.upload_reasoning_sft_evaluations \
  --evaluation-root "${EVALUATION_ROOT}" --repo-id "${MODEL_REPO}" \
  --path-in-repo evaluation/reasoning-preservation-50m

if (( selection_status != 0 )); then
  echo "No checkpoint passed all preservation guards; reports uploaded, root not promoted." >&2
  exit "${selection_status}"
fi

rm -rf "${RELEASE_ROOT}"
python -m scripts.export_reasoning_preservation_release \
  --summary "${EVALUATION_ROOT}/summary.json" --metrics "${METRICS}" \
  --evaluation-root "${EVALUATION_ROOT}" --tokenizer "${TOKENIZER}" \
  --dataset-manifest /workspace/tr-hash-reasoning-preservation-50m-mix/manifest.json \
  --output "${RELEASE_ROOT}"
python -m scripts.publish_reasoning_preservation_release \
  --bundle "${RELEASE_ROOT}" --repo-id "${MODEL_REPO}"
touch "${EVALUATION_ROOT}/.evaluation_complete"
