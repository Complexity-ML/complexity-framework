#!/usr/bin/env bash
set -euo pipefail

# Evaluate exactly the three end-of-epoch checkpoints against GSM8K test.
# The test parquet is supplied only to the evaluator and is never copied into
# the model or training-dataset directories.

CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the completed run directory}"
TOKENIZER="${TOKENIZER:?Set TOKENIZER to the pinned Agentic tokenizer snapshot}"
TEST_PARQUET="${TEST_PARQUET:?Set TEST_PARQUET to GSM8K main/test parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$CHECKPOINT_DIR/gsm8k_test_results}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

mapfile -t CHECKPOINTS < <(find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'step_*' | sort)
if [[ "${#CHECKPOINTS[@]}" -ne 3 ]]; then
  echo "Expected exactly 3 epoch checkpoints, found ${#CHECKPOINTS[@]}" >&2
  printf '%s\n' "${CHECKPOINTS[@]}" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
for index in 0 1 2; do
  epoch=$((index + 1))
  checkpoint="${CHECKPOINTS[$index]}"
  python -u -m scripts.eval_torch_gsm8k \
    "$checkpoint" \
    --tokenizer "$TOKENIZER" \
    --test-parquet "$TEST_PARQUET" \
    --split test \
    --limit 1319 \
    --output "$OUTPUT_DIR/epoch_${epoch}.json" \
    --device cuda \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --experiment-label supervised_gsm8k_sft
done
