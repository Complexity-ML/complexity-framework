#!/usr/bin/env bash
set -euo pipefail

# Separate supervised GSM8K probe. This script never uploads artifacts.
# Reuse already-downloaded assets by setting CHECKPOINT, TOKENIZER and SFT_BIN.

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the pinned refinement snapshot directory}"
TOKENIZER="${TOKENIZER:?Set TOKENIZER to the pinned Agentic tokenizer snapshot directory}"
SFT_BIN="${SFT_BIN:?Set SFT_BIN to the prepared GSM8K Agentic shard root}"
OUTPUT="${OUTPUT:-artifacts/tr_hash_agentic_100m_gsm8k_sft}"
RUN_NAME="${RUN_NAME:-tr-hash-agentic-100m-gsm8k-supervised-probe}"
NPROC="${NPROC:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-5}"

[[ -f "$CHECKPOINT/config.json" && -f "$CHECKPOINT/model.safetensors" ]] || {
  echo "Checkpoint must be a complete local HF snapshot: $CHECKPOINT" >&2
  exit 2
}
[[ -f "$TOKENIZER/tokenizer.json" && -f "$TOKENIZER/tokenizer_config.json" && -f "$TOKENIZER/chat_template.json" ]] || {
  echo "Tokenizer must include the pinned Agentic tokenizer and chat template: $TOKENIZER" >&2
  exit 2
}
[[ -f "$SFT_BIN/train/sft.idx.json" && -f "$SFT_BIN/chat_template.json" && ! -e "$SFT_BIN/test" ]] || {
  echo "SFT_BIN must contain train/sft.idx.json, chat_template.json and no test partition: $SFT_BIN" >&2
  exit 2
}

python - "$TOKENIZER/chat_template.json" "$SFT_BIN/chat_template.json" "$SFT_BIN/train/sft.idx.json" <<'PY'
import json
import sys

expected = "tr-hash-agentic-chat-v1"
tokenizer_template = json.load(open(sys.argv[1], encoding="utf-8"))
dataset_template = json.load(open(sys.argv[2], encoding="utf-8"))
dataset_index = json.load(open(sys.argv[3], encoding="utf-8"))
if tokenizer_template != dataset_template:
    raise SystemExit("Tokenizer and GSM8K dataset chat-template contracts differ")
for source, value in (
    ("tokenizer", tokenizer_template.get("id")),
    ("dataset", dataset_template.get("id")),
    ("training index", dataset_index.get("chat_template_id")),
):
    if value != expected:
        raise SystemExit(f"{source} chat template is {value!r}; expected {expected!r}")
print(f"Validated native chat template: {expected}")
PY

COMMON_ARGS=(
  -m scripts.sft_500m_32k_tr
  --checkpoint "$CHECKPOINT"
  --source-stage refinement
  --tokenizer "$TOKENIZER"
  --sft-bin "$SFT_BIN"
  --seq-len 2048
  --steps 0
  --epochs 3
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --full-parameter
  --bf16
  --pack-sequences
  --sft-liger-loss
  --no-sft-fp32-loss
  --no-eval-at-start
  --no-reset-lr-each-epoch
  --save-every-epoch
  --save-total-limit 4
  --save-dir "$OUTPUT"
  --run-name "$RUN_NAME"
  --tensorboard-dir "runs/$RUN_NAME/tensorboard"
  --log-steps 10
  --empty-cache-every 0
  --num-workers 0
  --use-custom-kernels auto
)

mkdir -p "$OUTPUT"
cp configs/tr_hash_agentic_100m_gsm8k_sft.json "$OUTPUT/experiment_contract.json"

if (( NPROC > 1 )); then
  torchrun --standalone --nproc_per_node "$NPROC" "${COMMON_ARGS[@]}"
else
  python -u "${COMMON_ARGS[@]}"
fi
