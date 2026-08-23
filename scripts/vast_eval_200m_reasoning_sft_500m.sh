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
REFINEMENT="${REFINEMENT:-/workspace/tr-hash-refinement}"
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

piqa_selected="$(< "${EVALUATION_ROOT}/selected_checkpoint.txt")"
final_selected="${checkpoints[$((${#checkpoints[@]} - 1))]}"
# Screen the early checkpoints where the original, higher-LR experiment peaked,
# then include the PIQA-selected and final checkpoints.  Deduplication keeps the
# candidate set within the visible GPU count on the intended 4x/8x hosts.
candidate_names=()
candidate_paths=()
declare -A seen_candidates=()
add_candidate() {
  local name="$1" path="$2"
  [[ -s "${path}/checkpoint.pt" ]] || return 0
  [[ -z "${seen_candidates[${path}]:-}" ]] || return 0
  seen_candidates["${path}"]=1
  candidate_names+=("${name}")
  candidate_paths+=("${path}")
}
add_candidate step250 "${CHECKPOINT_ROOT}/step_000250"
add_candidate step500 "${CHECKPOINT_ROOT}/step_000500"
add_candidate piqa "${piqa_selected}"
add_candidate final "${final_selected}"
if (( ${#candidate_paths[@]} > GPU_COUNT )); then
  echo "Reasoning candidate count exceeds visible GPUs." >&2
  exit 2
fi
SHARDS_PER_CANDIDATE="$((GPU_COUNT / ${#candidate_paths[@]}))"
(( SHARDS_PER_CANDIDATE >= 1 )) || exit 2
pids=()
names=()
for index in "${!candidate_paths[@]}"; do
  for ((shard=0; shard<SHARDS_PER_CANDIDATE; shard++)); do
    slot="$((index * SHARDS_PER_CANDIDATE + shard))"
    prefix="candidate_${candidate_names[$index]}_arc_reasoning_64"
    output="${EVALUATION_ROOT}/${prefix}.shard${shard}.json"
    CUDA_VISIBLE_DEVICES="${slot}" python -m scripts.eval_arc_generative \
      tr_hash_torch "${candidate_paths[$index]}" \
      --tokenizer "${TOKENIZER}" \
      --arc-easy-samples "${ARC_PROBE}/samples_arc_easy.jsonl" \
      --arc-challenge-samples "${ARC_PROBE}/samples_arc_challenge.jsonl" \
      --max-samples-per-task "${ARC_SAMPLES_PER_TASK:-32}" \
      --prompt-style bare \
      --num-shards "${SHARDS_PER_CANDIDATE}" --shard-index "${shard}" \
      --device cuda --output "${output}" > "${output%.json}.log" 2>&1 &
    pids+=("$!"); names+=("${candidate_names[$index]} reasoning shard ${shard}")
  done
done
status=0
for index in "${!pids[@]}"; do wait "${pids[$index]}" || status=1; done
(( status == 0 )) || exit "${status}"
reasoning_args=()
for index in "${!candidate_paths[@]}"; do
  prefix="candidate_${candidate_names[$index]}_arc_reasoning_64"
  merge_args=()
  for ((shard=0; shard<SHARDS_PER_CANDIDATE; shard++)); do
    merge_args+=(--shard "${EVALUATION_ROOT}/${prefix}.shard${shard}.json")
  done
  python -m scripts.merge_arc_generative_shards "${merge_args[@]}" \
    --expected-examples "$((2 * ${ARC_SAMPLES_PER_TASK:-32}))" \
    --output "${EVALUATION_ROOT}/${prefix}.json" > "${EVALUATION_ROOT}/${prefix}.log" 2>&1
  reasoning_args+=(--reasoning-report "${EVALUATION_ROOT}/${prefix}.json")
done

# Establish the source retention reference first, then use every GPU for the
# candidate batch.  This also works when four distinct candidates are present.
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_torch_arc_zero_shot "${REFINEMENT}" \
  --tokenizer "${TOKENIZER}" --arc-easy-samples "${ARC_PROBE}/samples_arc_easy.jsonl" \
  --arc-challenge-samples "${ARC_PROBE}/samples_arc_challenge.jsonl" --batch-size 64 \
  --output "${EVALUATION_ROOT}/source_arc_zero_shot_full.json" \
  > "${EVALUATION_ROOT}/source_arc_zero_shot_full.log" 2>&1
pids=(); names=()
zero_args=()
for index in "${!candidate_paths[@]}"; do
  output="${EVALUATION_ROOT}/candidate_${candidate_names[$index]}_arc_zero_shot_full.json"
  CUDA_VISIBLE_DEVICES="${index}" python -m scripts.eval_torch_arc_zero_shot \
    "${candidate_paths[$index]}" --tokenizer "${TOKENIZER}" \
    --arc-easy-samples "${ARC_PROBE}/samples_arc_easy.jsonl" \
    --arc-challenge-samples "${ARC_PROBE}/samples_arc_challenge.jsonl" --batch-size 64 \
    --output "${output}" > "${output%.json}.log" 2>&1 &
  pids+=("$!"); names+=("${candidate_names[$index]} zero-shot")
  zero_args+=(--zero-shot-report "${output}")
done
status=0
for index in "${!pids[@]}"; do wait "${pids[$index]}" || status=1; done
(( status == 0 )) || exit "${status}"

python -m scripts.promote_reasoning_sft_checkpoint \
  --summary "${EVALUATION_ROOT}/summary.json" "${reasoning_args[@]}" "${zero_args[@]}" \
  --source-zero-shot "${EVALUATION_ROOT}/source_arc_zero_shot_full.json" \
  --selected-checkpoint "${EVALUATION_ROOT}/selected_checkpoint.txt" \
  --selected-reasoning-report "${EVALUATION_ROOT}/selected_arc_reasoning_64.json" \
  --selected-zero-shot-report "${EVALUATION_ROOT}/selected_arc_zero_shot_full.json"
selected="$(< "${EVALUATION_ROOT}/selected_checkpoint.txt")"

# Diagnostic: repeat the same balanced 64-question probe with the short format
# instruction.  Promotion uses the native bare prompt above because that is
# the format represented in the reasoning-SFT targets.
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_arc_generative \
  tr_hash_torch "${selected}" \
  --tokenizer "${TOKENIZER}" \
  --arc-easy-samples "${ARC_PROBE}/samples_arc_easy.jsonl" \
  --arc-challenge-samples "${ARC_PROBE}/samples_arc_challenge.jsonl" \
  --max-samples-per-task "${ARC_SAMPLES_PER_TASK:-32}" \
  --prompt-style minimal --device cuda \
  --output "${EVALUATION_ROOT}/selected_arc_reasoning_64_minimal.json" \
  > "${EVALUATION_ROOT}/selected_arc_reasoning_64_minimal.log" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_torch_chat_panel \
  --checkpoint "${selected}" --tokenizer "${TOKENIZER}" \
  --panel configs/tr_hash_200m_sft_v2_regression.json --device cuda \
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
