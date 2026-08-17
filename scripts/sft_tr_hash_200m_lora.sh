#!/bin/bash
# LoRA post-training for TR-Hash 200M on AETHORIA-AI/complexity-atlas-posttrain.
#
# Text SFT is LoRA-only for this project (complexity/training/finetuning.py) --
# the base checkpoint stays frozen except for the low-rank adapters, so a
# comparatively small post-training corpus (~19M supervised tokens) can't
# catastrophically overwrite the 130B-token pretrain.
#
# Requires the pretrain run to have produced a final checkpoint, and the
# dataset's tokenized/32k-v2/ shards downloaded locally as --sft-bin (they are
# already in the exact input_ids.bin/labels.bin/examples.jsonl/
# loss_metadata.jsonl/sft.idx.json layout this pipeline expects).
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the 200M pretrain final checkpoint dir}"
SFT_BIN="${SFT_BIN:?set SFT_BIN to the complexity-atlas-posttrain tokenized/32k-v2 dir}"
CURRICULUM_CONFIG="${CURRICULUM_CONFIG:-configs/sft_curriculum_200m_atlas_posttrain.yaml}"
THROUGH_STAGE="${THROUGH_STAGE:-full-extended}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/tr_hash_200m_atlas_posttrain_lora}"
TOKENIZER="${TOKENIZER:-tokenizer}"
WORLD_SIZE="${WORLD_SIZE:-1}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_LR_MULTIPLIER="${LORA_LR_MULTIPLIER:-1.0}"
EXPERT_LR_MULTIPLIER="${EXPERT_LR_MULTIPLIER:-0.25}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,v_proj,o_proj,shared_gate,shared_up,shared_down}"
SEED="${SEED:-42}"

exec python3 -m scripts.run_sft_curriculum \
  --checkpoint "$CHECKPOINT" \
  --sft-bin "$SFT_BIN" \
  --curriculum-config "$CURRICULUM_CONFIG" \
  --through-stage "$THROUGH_STAGE" \
  --output-root "$OUTPUT_ROOT" \
  --tokenizer "$TOKENIZER" \
  --world-size "$WORLD_SIZE" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --lora-lr-multiplier "$LORA_LR_MULTIPLIER" \
  --expert-lr-multiplier "$EXPERT_LR_MULTIPLIER" \
  --lora-targets "$LORA_TARGETS" \
  --seed "$SEED" \
  "$@"
