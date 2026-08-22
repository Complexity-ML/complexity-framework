#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

BASE_CHECKPOINT="${BASE_CHECKPOINT:-/workspace/tr-hash-refinement}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/tr-hash-moe-200m-sft-v2-300k}"
SFT_BIN="${SFT_BIN:-${DATASET_ROOT}/tokenized/tr-hash-32k-v2-2048}"
TOKENIZER="${TOKENIZER:-${SFT_BIN}/tokenizer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/tr_hash_moe_200m_clean_sft_v2_full_3e}"
DETECTED_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
NPROC_PER_NODE="${NPROC_PER_NODE:-${DETECTED_GPU_COUNT}}"
# Packed 2048-token SFT needs activation headroom on 32 GiB consumer RTX 5090
# cards. Batch 24 and 20 both exhaust memory before the first optimizer step;
# 16 is the validated anti-OOM production setting.
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export COMPLEXITY_REQUIRE_LIGER=1

if [[ "${NPROC_PER_NODE}" != "4" && "${NPROC_PER_NODE}" != "8" ]]; then
  echo "Clean SFT v2 supports exactly 4 or 8 RTX 5090 ranks; got ${NPROC_PER_NODE}." >&2
  exit 2
fi
if (( NPROC_PER_NODE > DETECTED_GPU_COUNT )); then
  echo "Requested ${NPROC_PER_NODE} ranks but only ${DETECTED_GPU_COUNT} CUDA GPUs are visible." >&2
  exit 2
fi
echo "[preflight] ddp_world_size=${NPROC_PER_NODE} batch_per_gpu=${BATCH_SIZE_PER_GPU} global_batch=$((NPROC_PER_NODE * BATCH_SIZE_PER_GPU))"

if [[ -n "${RESUME_FROM:-}" ]]; then
  echo "Clean SFT v2 must start from Refinement step 8156; RESUME_FROM is not allowed." >&2
  exit 2
fi

# The HF sync supervisor is allowed to create OUTPUT_ROOT before training, but
# no training artifact may already exist there. This prevents a rental reused
# for another experiment from silently mixing or resuming old weights.
shopt -s nullglob
stale_artifacts=(
  "${OUTPUT_ROOT}"/step_*
  "${OUTPUT_ROOT}"/final_*
  "${OUTPUT_ROOT}"/interrupted_*
  "${OUTPUT_ROOT}"/token_pack_*
)
for exact_artifact in "${OUTPUT_ROOT}/best" "${OUTPUT_ROOT}/final"; do
  if [[ -e "${exact_artifact}" ]]; then
    stale_artifacts+=("${exact_artifact}")
  fi
done
if [[ -e "${OUTPUT_ROOT}/.training_complete" ]]; then
  stale_artifacts+=("${OUTPUT_ROOT}/.training_complete")
fi
if (( ${#stale_artifacts[@]} > 0 )); then
  printf 'Clean SFT v2 refuses stale training artifact: %s\n' "${stale_artifacts[@]}" >&2
  exit 2
fi
shopt -u nullglob

python -c 'from complexity.core.losses import has_liger_fused_linear_ce; assert has_liger_fused_linear_ce(); print("[preflight] liger_fused_linear_ce=required+available")'
python - "${SFT_BIN}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
train = manifest.get("partitions", {}).get("train", {})
checks = {
    "quality_status": manifest.get("quality_status") == "passed",
    "release_ready": manifest.get("release_quality", {}).get("ready") is True,
    "no_truncation": manifest.get("release_quality", {}).get("token_truncation") is False,
    "train_examples": train.get("examples") == 300_000,
    "vocab_size": manifest.get("tokenizer_vocab_size") == 32_000,
    "sequence_length": manifest.get("sequence_length_cap") == 2_048,
    "chat_eos": manifest.get("chat_template_eos_token") == "</s>",
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"SFT v2 preflight failed: {failed}")
print("[preflight] clean-sft-v2 train=300000 vocab=32000 seq=2048 eos=</s> truncation=false")
PY

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node "${NPROC_PER_NODE}" \
  -m scripts.sft_tr \
  --checkpoint "${BASE_CHECKPOINT}" \
  --source-stage refinement \
  --sft-bin "${SFT_BIN}" \
  --require-release-ready \
  --tokenizer "${TOKENIZER}" \
  --pack-sequences \
  --steps 0 \
  --epochs 3 \
  --batch-size "${BATCH_SIZE_PER_GPU}" \
  --seq-len 2048 \
  --lr 2e-5 \
  --weight-decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --warmup-ratio 0.03 \
  --no-reset-lr-each-epoch \
  --bf16 \
  --no-sft-fp32-loss \
  --sft-liger-loss \
  --loss-chunk-tokens 1024 \
  --save-steps 0 \
  --save-every-epoch \
  --save-total-limit 3 \
  --save-best \
  --save-dir "${OUTPUT_ROOT}" \
  --run-name tr-hash-moe-200m-clean-sft-v2-full-3e \
  --seed 42 \
  --use-custom-kernels true \
  --full-parameter \
  --expert-lr-multiplier 1.0 \
  --eval-steps 0 \
  --eval-every-epoch \
  --eval-batches 0 \
  --min-eval-fraction 0.01 \
  --eval-at-start \
  --early-stopping-min-epochs 1 \
  --early-stopping-patience 0

# Written only after torchrun exits successfully. Evaluation and publication
# supervisors use this as the authoritative clean-run completion signal.
touch "${OUTPUT_ROOT}/.training_complete"
