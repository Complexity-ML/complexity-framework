#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/complexity-framework}"
COCO_ROOT="${COCO_ROOT:-$REPO_ROOT/artifacts/COCO}"
HF_DATASET="${HF_DATASET:-manh6054/MSCOCO}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
READY_MARKER="$COCO_ROOT/.coco2017-ready"

expected_train=118287
expected_validation=5000

count_images() {
  local directory="$1"
  if [[ ! -d "$directory" ]]; then
    printf '0\n'
    return
  fi
  find "$directory" -maxdepth 1 -type f -name '*.jpg' -print | wc -l | tr -d ' '
}

validate_layout() {
  local train_count validation_count
  train_count="$(count_images "$COCO_ROOT/images/train2017")"
  validation_count="$(count_images "$COCO_ROOT/images/val2017")"
  [[ "$train_count" == "$expected_train" ]] || return 1
  [[ "$validation_count" == "$expected_validation" ]] || return 1
  [[ -f "$COCO_ROOT/annotations/instances_train2017.json" ]] || return 1
  [[ -f "$COCO_ROOT/annotations/instances_val2017.json" ]] || return 1
}

if [[ -f "$READY_MARKER" ]] && validate_layout; then
  echo "[skip] COCO 2017 is ready: $COCO_ROOT"
  exit 0
fi

mkdir -p "$COCO_ROOT/images"
if ! "$PYTHON_BIN" -c 'import huggingface_hub, hf_xet' >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install 'huggingface_hub[hf_xet]>=0.32.0'
fi
HF_BIN="$(dirname "$PYTHON_BIN")/hf"
if [[ ! -x "$HF_BIN" ]]; then
  echo "[error] Hugging Face CLI was not installed at $HF_BIN" >&2
  exit 2
fi

echo "[download] COCO 2017 through Hugging Face Xet"
HF_XET_HIGH_PERFORMANCE=1 "$HF_BIN" download "$HF_DATASET" \
  train2017.zip \
  val2017.zip \
  annotations_trainval2017.zip \
  --repo-type dataset \
  --local-dir "$COCO_ROOT"

(
  cd "$COCO_ROOT"
  md5sum -c - <<'CHECKSUMS'
cced6f7f71b7629ddf16f17bbcfab6b2  train2017.zip
442b8da7639aecaf257c1dceb8ba8c80  val2017.zip
f4bbac642086de4f52a3fdda2de5fa2c  annotations_trainval2017.zip
CHECKSUMS
)

echo "[extract] COCO train/validation images and instance annotations"
unzip -q -n "$COCO_ROOT/train2017.zip" -d "$COCO_ROOT/images"
unzip -q -n "$COCO_ROOT/val2017.zip" -d "$COCO_ROOT/images"
unzip -q -n "$COCO_ROOT/annotations_trainval2017.zip" -d "$COCO_ROOT"

if ! validate_layout; then
  echo "[error] extracted COCO 2017 layout or image counts are incomplete" >&2
  exit 2
fi

touch "$READY_MARKER"
echo "[ready] COCO 2017: $expected_train train, $expected_validation validation"
