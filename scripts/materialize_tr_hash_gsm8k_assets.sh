#!/usr/bin/env bash
set -euo pipefail

# Downloads only the exact files required by the experiment. Set any of the
# *_SOURCE variables to reuse a file or directory already present on the host.

MODEL_REPO="AETHORIA-AI/TR-HASH-MoE-100M-70B-Agentic-Refinement"
MODEL_REVISION="99fee390916154ec9d8c0f049c3a890f81414e20"
MODEL_SUBFOLDER="token_pack_014_213622"
TOKENIZER_REPO="AETHORIA-AI/TR-HASH-Tokenizer-32K-Agentic"
TOKENIZER_REVISION="2fcbc2c5359ded0244ca14531f1b3806eebac55e"
GSM8K_REPO="openai/gsm8k"
GSM8K_REVISION="740312add88f781978c0658806c59bc2815b9866"
ASSET_ROOT="${ASSET_ROOT:-artifacts/gsm8k-supervised-assets}"

mkdir -p "$ASSET_ROOT/checkpoint" "$ASSET_ROOT/tokenizer" "$ASSET_ROOT/source"

if [[ -n "${CHECKPOINT_SOURCE:-}" ]]; then
  cp "$CHECKPOINT_SOURCE/config.json" "$ASSET_ROOT/checkpoint/config.json"
  cp "$CHECKPOINT_SOURCE/model.safetensors" "$ASSET_ROOT/checkpoint/model.safetensors"
else
  hf download "$MODEL_REPO" config.json "$MODEL_SUBFOLDER/model.safetensors" \
    --revision "$MODEL_REVISION" --local-dir "$ASSET_ROOT/model-download"
  cp "$ASSET_ROOT/model-download/config.json" "$ASSET_ROOT/checkpoint/config.json"
  cp "$ASSET_ROOT/model-download/$MODEL_SUBFOLDER/model.safetensors" \
    "$ASSET_ROOT/checkpoint/model.safetensors"
fi

if [[ -n "${TOKENIZER_SOURCE:-}" ]]; then
  cp "$TOKENIZER_SOURCE/tokenizer.json" "$ASSET_ROOT/tokenizer/tokenizer.json"
  cp "$TOKENIZER_SOURCE/tokenizer_config.json" "$ASSET_ROOT/tokenizer/tokenizer_config.json"
else
  hf download "$TOKENIZER_REPO" tokenizer.json tokenizer_config.json \
    --revision "$TOKENIZER_REVISION" --local-dir "$ASSET_ROOT/tokenizer"
fi

if [[ -n "${GSM8K_TRAIN_PARQUET:-}" && -n "${GSM8K_TEST_PARQUET:-}" ]]; then
  cp "$GSM8K_TRAIN_PARQUET" "$ASSET_ROOT/source/train.parquet"
  cp "$GSM8K_TEST_PARQUET" "$ASSET_ROOT/source/test.parquet"
else
  hf download "$GSM8K_REPO" main/train-00000-of-00001.parquet \
    main/test-00000-of-00001.parquet --repo-type dataset \
    --revision "$GSM8K_REVISION" --local-dir "$ASSET_ROOT/gsm8k-download"
  cp "$ASSET_ROOT/gsm8k-download/main/train-00000-of-00001.parquet" \
    "$ASSET_ROOT/source/train.parquet"
  cp "$ASSET_ROOT/gsm8k-download/main/test-00000-of-00001.parquet" \
    "$ASSET_ROOT/source/test.parquet"
fi

python -u -m scripts.prepare_tr_hash_gsm8k_sft \
  --train-parquet "$ASSET_ROOT/source/train.parquet" \
  --test-parquet "$ASSET_ROOT/source/test.parquet" \
  --tokenizer "$ASSET_ROOT/tokenizer" \
  --output "$ASSET_ROOT/dataset"

# The tokenizer repository publishes the Jinja template, while native training
# and evaluation consume the audited JSON contract generated with this shard.
cp "$ASSET_ROOT/dataset/chat_template.json" "$ASSET_ROOT/tokenizer/chat_template.json"

cat <<EOF
CHECKPOINT=$ASSET_ROOT/checkpoint
TOKENIZER=$ASSET_ROOT/tokenizer
SFT_BIN=$ASSET_ROOT/dataset
TEST_PARQUET=$ASSET_ROOT/source/test.parquet
EOF
