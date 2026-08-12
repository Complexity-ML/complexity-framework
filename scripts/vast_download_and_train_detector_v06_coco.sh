#!/bin/bash
set -euo pipefail

source "${VENV_ACTIVATE:-/venv/main/bin/activate}"
cd "${REPO_ROOT:-/workspace/complexity-framework}"

dataset_root="artifacts/COCO"
mkdir -p "$dataset_root/images" "$dataset_root/annotations"

download_and_extract() {
  local url="$1"
  local archive="$dataset_root/${url##*/}"
  local marker="$2"
  if [[ -e "$marker" ]]; then
    printf '[dataset] ready: %s\n' "$marker"
    return
  fi

  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --continue=true \
      --max-connection-per-server=16 \
      --split=16 \
      --min-split-size=1M \
      --dir="$(dirname "$archive")" \
      --out="$(basename "$archive")" \
      "$url"
  else
    wget --continue --progress=dot:giga --output-document "$archive" "$url"
  fi
  unzip -q -n "$archive" -d "$dataset_root"
}

download_and_extract \
  "http://images.cocodataset.org/zips/train2017.zip" \
  "$dataset_root/train2017/000000000009.jpg"
download_and_extract \
  "http://images.cocodataset.org/zips/val2017.zip" \
  "$dataset_root/val2017/000000000139.jpg"
download_and_extract \
  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
  "$dataset_root/annotations/instances_train2017.json"

if [[ ! -e "$dataset_root/images/train2017" ]]; then
  ln -s ../train2017 "$dataset_root/images/train2017"
fi
if [[ ! -e "$dataset_root/images/val2017" ]]; then
  ln -s ../val2017 "$dataset_root/images/val2017"
fi

printf '[dataset] COCO 2017 ready; starting detector training\n'
exec scripts/vast_pretrain_detector_v06_coco.sh
