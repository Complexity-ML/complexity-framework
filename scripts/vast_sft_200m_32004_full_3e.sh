#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

BASE_CHECKPOINT="${BASE_CHECKPOINT:-/workspace/tr-hash-refinement}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/tr-hash-moe-200m-sft-v3-32004}"
SFT_BIN="${SFT_BIN:-${DATASET_ROOT}/tokenized/tr-hash-32k-v3-32004-2048}"
TOKENIZER="${TOKENIZER:-${SFT_BIN}/tokenizer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/tr_hash_moe_200m_sft_v3_32004_full_3e}"
DETECTED_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
NPROC_PER_NODE="${NPROC_PER_NODE:-${DETECTED_GPU_COUNT}}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
EPOCHS="${EPOCHS:-3}"
RUN_NAME="${RUN_NAME:-tr-hash-moe-200m-sft-v3-32004-full-3e}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export COMPLEXITY_REQUIRE_LIGER=1

if [[ "${NPROC_PER_NODE}" != "4" && "${NPROC_PER_NODE}" != "8" ]]; then
  echo "SFT v3 supports exactly 4 or 8 RTX 5090 ranks; got ${NPROC_PER_NODE}." >&2
  exit 2
fi
if (( NPROC_PER_NODE > DETECTED_GPU_COUNT )); then
  echo "Requested ${NPROC_PER_NODE} ranks but only ${DETECTED_GPU_COUNT} CUDA GPUs are visible." >&2
  exit 2
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
  echo "SFT v3 must start from the 32,004-row Refinement checkpoint; RESUME_FROM is forbidden." >&2
  exit 2
fi
echo "[preflight] ddp_world_size=${NPROC_PER_NODE} batch_per_gpu=${BATCH_SIZE_PER_GPU} global_batch=$((NPROC_PER_NODE * BATCH_SIZE_PER_GPU))"

# Refuse a silent resume or mixture with an older run. The sync supervisor may
# create OUTPUT_ROOT, but it may not populate it with training artifacts first.
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
  printf 'SFT v3 refuses stale training artifact: %s\n' "${stale_artifacts[@]}" >&2
  exit 2
fi
shopt -u nullglob

python -c 'from complexity.core.losses import has_liger_fused_linear_ce; assert has_liger_fused_linear_ce(); print("[preflight] liger_fused_linear_ce=required+available")'
python - "${BASE_CHECKPOINT}" "${SFT_BIN}/manifest.json" "${SFT_BIN}/train/sft.idx.json" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
train = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
config = json.loads((base / "config.json").read_text(encoding="utf-8"))
special = {
    "<|think_start|>": 32_000,
    "<|think_end|>": 32_001,
    "<|final_start|>": 32_002,
    "<|final_end|>": 32_003,
}
examples = int(train.get("examples", 0))
checks = {
    "base_vocab_size": int(config.get("vocab_size", -1)) == 32_004,
    "quality_status": manifest.get("quality_status") == "passed",
    "release_ready": manifest.get("release_quality", {}).get("ready") is True,
    "no_truncation": manifest.get("release_quality", {}).get("token_truncation") is False,
    "train_examples": examples >= 290_000,
    "manifest_vocab_size": manifest.get("tokenizer_vocab_size") == 32_004,
    "train_vocab_size": train.get("vocab_size") == 32_004,
    "sequence_length": manifest.get("sequence_length_cap") == 2_048,
    "chat_template": manifest.get("chat_template_id") == "complexity-chat-v3-32004",
    "special_token_ids": manifest.get("special_token_ids") == special,
    "supervised_markers": train.get("special_token_label_counts")
    == {token: examples for token in special},
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"SFT v3 preflight failed: {failed}")
print(
    f"[preflight] sft-v3 train={examples} vocab=32004 seq=2048 "
    "template=complexity-chat-v3-32004 markers=supervised truncation=false"
)
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
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE_PER_GPU}" \
  --seq-len 2048 \
  --lr "${LEARNING_RATE}" \
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
  --run-name "${RUN_NAME}" \
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

touch "${OUTPUT_ROOT}/.training_complete"
