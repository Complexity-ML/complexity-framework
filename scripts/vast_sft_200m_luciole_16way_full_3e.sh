#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

BASE_CHECKPOINT="${BASE_CHECKPOINT:-/workspace/tr-hash-refinement}"
TOKENIZER="${TOKENIZER:-/workspace/tr-hash-refinement}"
DATA_ROOT="${DATA_ROOT:-/workspace/luciole-16way-sft}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/tr_hash_moe_200m_160b_luciole_16way_full_sft_3e}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-48}"

if [[ -n "${RESUME_FROM:-}" ]]; then
  echo "This production run must start from the 160B refinement root; RESUME_FROM is not allowed." >&2
  exit 2
fi

python -c 'import liger_kernel; print("[preflight] liger_kernel=required+available")'
python - "$DATA_ROOT/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("train_examples") != 209_000:
    raise SystemExit(f"unexpected Luciole train size: {manifest.get('train_examples')!r}")
if len(manifest.get("sources", {})) != 16:
    raise SystemExit(f"unexpected Luciole source count: {len(manifest.get('sources', {}))}")
if manifest.get("assistant_supervision") != "final_assistant_only":
    raise SystemExit("Luciole manifest does not guarantee final-assistant-only supervision")
print("[preflight] dataset=luciole-16way train=209000 sources=16 supervision=final-assistant-only")
PY

exec python -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  -m scripts.sft_tr \
  --checkpoint "$BASE_CHECKPOINT" \
  --jsonl "$DATA_ROOT/train.jsonl" \
  --eval-jsonl "$DATA_ROOT/eval.jsonl" \
  --tokenizer "$TOKENIZER" \
  --pack-sequences \
  --steps 0 \
  --epochs 3 \
  --batch-size "$BATCH_SIZE_PER_GPU" \
  --seq-len 512 \
  --lr 2e-5 \
  --weight-decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --warmup-ratio 0.03 \
  --no-reset-lr-each-epoch \
  --bf16 \
  --loss-chunk-tokens 1024 \
  --save-steps 0 \
  --save-every-epoch \
  --save-total-limit 3 \
  --save-best \
  --save-dir "$OUTPUT_ROOT" \
  --run-name tr-hash-moe-200m-160b-luciole-16way-full-sft-3e \
  --seed 42 \
  --use-custom-kernels true \
  --full-parameter \
  --expert-lr-multiplier 1.0 \
  --eval-steps 0 \
  --eval-every-epoch \
  --eval-batches 0 \
  --min-eval-fraction 0.005 \
  --eval-at-start \
  --early-stopping-min-epochs 1 \
  --early-stopping-patience 0
