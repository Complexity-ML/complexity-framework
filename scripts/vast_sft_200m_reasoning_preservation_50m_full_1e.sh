#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

BASE_CHECKPOINT="${BASE_CHECKPOINT:-/workspace/tr-hash-sft-v2}"
SFT_BIN="${SFT_BIN:-/workspace/tr-hash-reasoning-preservation-50m-mix}"
TOKENIZER="${TOKENIZER:-${SFT_BIN}/tokenizer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/tr_hash_moe_200m_reasoning_preservation_50m_full_1e}"
RUN_NAME="${RUN_NAME:-tr-hash-moe-200m-reasoning-preservation-50m-full-1e}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
LR="${LR:-1.5e-7}"
SAVE_STEPS="${SAVE_STEPS:-250}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export COMPLEXITY_REQUIRE_LIGER=1

if [[ "${NPROC_PER_NODE}" != "4" && "${NPROC_PER_NODE}" != "8" ]]; then
  echo "Expected 4 or 8 RTX 5090 ranks; got ${NPROC_PER_NODE}." >&2
  exit 2
fi
if [[ -f "${OUTPUT_ROOT}/.training_complete" ]]; then
  echo "[launch] training already complete"
  exit 0
fi

RESUME_ARGS=()
LATEST_RESUME="$(find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -type f \
  -name checkpoint.pt -path "${OUTPUT_ROOT}/step_*/*" -size +0c \
  -print 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -n "${LATEST_RESUME}" ]]; then
  RESUME_ARGS=(--resume "$(dirname "${LATEST_RESUME}")")
fi

python -c 'from complexity.core.losses import has_liger_fused_linear_ce; assert has_liger_fused_linear_ce(); print("[preflight] liger=required+available")'
python - "${BASE_CHECKPOINT}" "${SFT_BIN}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
selection = manifest.get("selection", {})
checks = {
    "base_weights": (base / "model.safetensors").is_file(),
    "release_ready": manifest.get("release_quality", {}).get("ready") is True,
    "token_total": 200_000_000 <= int(manifest.get("actual_unique_formatted_tokens", 0)) < 200_020_000,
    "general_replay": 150_000_000 <= int(selection.get("general_tokens", 0)) < 150_020_000,
    "reasoning": 50_000_000 <= int(selection.get("reasoning_tokens", 0)) < 50_020_000,
    "vocab": manifest.get("tokenizer_vocab_size") == 32_000,
    "context": manifest.get("sequence_length_cap") == 2_048,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Preservation SFT preflight failed: {failed}")
print("[preflight] source=SFT-v2 general=150M reasoning=50M total=200M")
PY

if (( ${#RESUME_ARGS[@]} > 0 )); then
  echo "[resume] ${RESUME_ARGS[1]}"
else
  echo "[launch] fresh from released SFT v2"
fi
echo "[launch] ranks=${NPROC_PER_NODE} batch_per_gpu=${BATCH_SIZE_PER_GPU} lr=${LR} epochs=1"

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node "${NPROC_PER_NODE}" \
  -m scripts.sft_tr \
  --checkpoint "${BASE_CHECKPOINT}" \
  "${RESUME_ARGS[@]}" \
  --source-stage supervised-finetuning \
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
  --save-total-limit 12 \
  --save-best \
  --save-dir "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --seed 20260823 \
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
