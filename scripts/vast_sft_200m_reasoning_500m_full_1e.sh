#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

BASE_CHECKPOINT="${BASE_CHECKPOINT:-/workspace/tr-hash-refinement}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/tr-hash-moe-200m-reasoning-sft-500m}"
SFT_BIN="${SFT_BIN:-${DATASET_ROOT}/tokenized/tr-hash-32k-v2-2048}"
TOKENIZER="${TOKENIZER:-${SFT_BIN}/tokenizer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/tr_hash_moe_200m_reasoning_sft_500m_full_1e}"
DETECTED_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
NPROC_PER_NODE="${NPROC_PER_NODE:-${DETECTED_GPU_COUNT}}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
LR="${LR:-5e-6}"
SAVE_STEPS="${SAVE_STEPS:-250}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export COMPLEXITY_REQUIRE_LIGER=1

if [[ "${NPROC_PER_NODE}" != "1" && "${NPROC_PER_NODE}" != "4" && "${NPROC_PER_NODE}" != "8" ]]; then
  echo "Reasoning SFT supports 1, 4 or 8 RTX 5090 ranks; got ${NPROC_PER_NODE}." >&2
  exit 2
fi
if (( NPROC_PER_NODE > DETECTED_GPU_COUNT )); then
  echo "Requested ${NPROC_PER_NODE} ranks but only ${DETECTED_GPU_COUNT} GPUs are visible." >&2
  exit 2
fi
if [[ -f "${OUTPUT_ROOT}/.training_complete" ]]; then
  echo "[launch] reasoning SFT is already complete; nothing to resume"
  exit 0
fi

# A first launch always initializes from the audited Refinement checkpoint.
# After an interruption, only this run's own complete top-level step checkpoint
# may be resumed.  Directories under best/ are intentionally excluded because
# best-model selection and exact optimizer/data-cursor recovery are separate
# concerns.
RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  if [[ ! -s "${RESUME_FROM}/checkpoint.pt" && ! -s "${RESUME_FROM}" ]]; then
    echo "Requested resume checkpoint is missing or incomplete: ${RESUME_FROM}" >&2
    exit 2
  fi
  RESUME_ARGS=(--resume "${RESUME_FROM}")
elif [[ -d "${OUTPUT_ROOT}" ]]; then
  LATEST_RESUME="$(find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 \
    -type f -name checkpoint.pt -path "${OUTPUT_ROOT}/step_*/*" -size +0c \
    -print 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${LATEST_RESUME}" ]]; then
    RESUME_ARGS=(--resume "$(dirname "${LATEST_RESUME}")")
  fi
fi

python -c 'from complexity.core.losses import has_liger_fused_linear_ce; assert has_liger_fused_linear_ce(); print("[preflight] liger_fused_linear_ce=required+available")'
python - "${BASE_CHECKPOINT}" "${SFT_BIN}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
train = manifest.get("partitions", {}).get("train", {})
actual = int(manifest.get("actual_unique_formatted_tokens", train.get("num_tokens", 0)))
checks = {
    "base_weights": (base / "model.safetensors").is_file(),
    "release_ready": manifest.get("release_quality", {}).get("ready") is True,
    "no_truncation": manifest.get("release_quality", {}).get("token_truncation") is False,
    "token_target": 500_000_000 <= actual < 500_020_000,
    "vocab_size": manifest.get("tokenizer_vocab_size") == 32_000,
    "sequence_length": manifest.get("sequence_length_cap") == 2_048,
    "chat_eos": manifest.get("chat_template_eos_token") == "</s>",
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Reasoning SFT preflight failed: {failed}; actual_tokens={actual}")
print(f"[preflight] refinement=step-8156 unique_tokens={actual:,} vocab=32000 seq=2048")
PY

if (( ${#RESUME_ARGS[@]} > 0 )); then
  echo "[resume] exact SFT state from ${RESUME_ARGS[1]}"
else
  echo "[launch] fresh initialization from ${BASE_CHECKPOINT}"
fi
echo "[launch] ranks=${NPROC_PER_NODE} batch_per_gpu=${BATCH_SIZE_PER_GPU} lr=${LR} epochs=1"
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node "${NPROC_PER_NODE}" \
  -m scripts.sft_tr \
  --checkpoint "${BASE_CHECKPOINT}" \
  "${RESUME_ARGS[@]}" \
  --source-stage refinement \
  --sft-bin "${SFT_BIN}" \
  --require-release-ready \
  --tokenizer "${TOKENIZER}" \
  --pack-sequences \
  --steps 0 \
  --epochs 1 \
  --batch-size "${BATCH_SIZE_PER_GPU}" \
  --seq-len 2048 \
  --lr "${LR}" \
  --weight-decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --warmup-ratio 0.03 \
  --no-reset-lr-each-epoch \
  --bf16 \
  --no-sft-fp32-loss \
  --sft-liger-loss \
  --loss-chunk-tokens 1024 \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit 24 \
  --save-best \
  --save-dir "${OUTPUT_ROOT}" \
  --run-name tr-hash-moe-200m-reasoning-sft-500m-full-1e \
  --seed 42 \
  --use-custom-kernels true \
  --full-parameter \
  --expert-lr-multiplier 1.0 \
  --eval-steps "${SAVE_STEPS}" \
  --eval-batches 0 \
  --min-eval-fraction 0.002 \
  --eval-at-start \
  --early-stopping-min-epochs 1 \
  --early-stopping-patience 0

touch "${OUTPUT_ROOT}/.training_complete"
