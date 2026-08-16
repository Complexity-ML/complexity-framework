#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

BASE_CHECKPOINT="${BASE_CHECKPOINT:-artifacts/tr_hash_moe_500m_20b_hf}"
SFT_BIN="${SFT_BIN:-artifacts/complexity_atlas_posttrain/tokenized/32k-v2}"
TOKENIZER="${TOKENIZER:-$SFT_BIN/tokenizer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/tr_hash_500m_32k_v2_lora_3e_packed}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-24}"

if [[ -n "${RESUME_FROM:-}" ]]; then
  echo "This clean packed LoRA run must start from the pretrained base; RESUME_FROM is not allowed." >&2
  exit 2
fi

exec /venv/main/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  -m scripts.sft_tr \
  --checkpoint "$BASE_CHECKPOINT" \
  --sft-bin "$SFT_BIN" \
  --require-release-ready \
  --tokenizer "$TOKENIZER" \
  --curriculum-config configs/sft_500m_32k_v2_balanced.yaml \
  --curriculum-stage full-shard-weighted \
  --pack-sequences \
  --steps 0 \
  --epochs 3 \
  --batch-size "$BATCH_SIZE_PER_GPU" \
  --seq-len 512 \
  --lr 3e-6 \
  --weight-decay 0.0 \
  --bf16 \
  --grad-ckpt \
  --loss-chunk-tokens 1024 \
  --save-steps 0 \
  --save-every-epoch \
  --save-total-limit 3 \
  --save-dir "$OUTPUT_ROOT" \
  --run-name tr-hash-500m-32k-v2-lora-3e-packed \
  --seed 42 \
  --use-custom-kernels auto \
  --lora-rank 32 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-targets q_proj,k_proj,v_proj,o_proj,shared_gate,shared_up,shared_down,expert_gate,expert_up,expert_down \
  --expert-lr-multiplier 0.25 \
  --eval-steps 0 \
  --eval-every-epoch \
  --eval-batches 0 \
  --min-eval-fraction 0.01 \
  --no-eval-at-start \
  --save-best \
  --early-stopping-min-epochs 3 \
  --early-stopping-patience 0
