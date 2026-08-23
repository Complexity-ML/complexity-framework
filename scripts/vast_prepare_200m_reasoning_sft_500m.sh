#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

MODEL_REPO="${MODEL_REPO:-AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement}"
MODEL_REVISION="${MODEL_REVISION:-ad4e9217b637720fb939babe8c8ce285a804ade2}"
MODEL_DIR="${MODEL_DIR:-/workspace/tr-hash-refinement}"
DATASET_REPO="${DATASET_REPO:-AETHORIA-AI/TR-HASH-MoE-200M-Reasoning-SFT-500M}"
DATASET_REVISION="${DATASET_REVISION:-main}"
DATASET_DIR="${DATASET_DIR:-/workspace/tr-hash-moe-200m-reasoning-sft-500m}"
PIQA_DIR="${PIQA_DIR:-/workspace/physicaliqa-train-dev/physicaliqa-train-dev}"
ARC_DIR="${ARC_DIR:-/workspace/arc-evaluation-samples}"

if [[ ! -s /workspace/.hf_token ]]; then
  echo "Missing /workspace/.hf_token; create it with mode 600 first." >&2
  exit 2
fi
export HF_TOKEN="$(< /workspace/.hf_token)"
export HF_XET_HIGH_PERFORMANCE=1
umask 077

python - <<'PY'
import shutil
import torch

count = torch.cuda.device_count()
names = [torch.cuda.get_device_name(index) for index in range(count)]
if count not in {4, 8}:
    raise SystemExit(f"Expected exactly 4 or 8 CUDA GPUs, found {count}: {names}")
if not all("5090" in name for name in names):
    raise SystemExit(f"Expected only RTX 5090 GPUs, found: {names}")
free = shutil.disk_usage("/workspace").free
if free < 60 * 1024**3:
    raise SystemExit(f"At least 60 GiB free is required; found {free / 1024**3:.1f} GiB")
print(f"[preflight] cuda_gpus={count} names={names} free_disk_gib={free / 1024**3:.1f}")
PY

# Keep the CUDA-enabled torch supplied by the Vast image.
python -m pip install -e .
python -m pip install \
  "huggingface_hub[hf_xet]>=0.32.0" \
  "liger-kernel>=0.5.0" \
  "datasets>=3.0.0"

python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${MODEL_REPO}",
    revision="${MODEL_REVISION}",
    local_dir="${MODEL_DIR}",
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "model_config.yaml",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ],
    token=True,
)
snapshot_download(
    repo_id="${DATASET_REPO}",
    repo_type="dataset",
    revision="${DATASET_REVISION}",
    local_dir="${DATASET_DIR}",
    allow_patterns=[
        "README.md",
        "manifest.json",
        "metadata/release-audit.json",
        "metadata/recipe.json",
        "tokenized/tr-hash-32k-v2-2048/**",
    ],
    token=True,
)
PY

python - "${PIQA_DIR}" <<'PY'
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

target = Path(sys.argv[1])
required = (target / "dev.jsonl", target / "dev-labels.lst")
if not all(path.is_file() for path in required):
    url = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(target.parent)
examples = sum(1 for _ in required[0].open(encoding="utf-8"))
labels = sum(1 for _ in required[1].open(encoding="utf-8"))
if (examples, labels) != (1_838, 1_838):
    raise SystemExit(f"Expected 1,838 PIQA examples and labels, found {examples}/{labels}")
print("[preflight] piqa_validation_examples=1838")
PY

if [[ ! -s "${ARC_DIR}/manifest.json" ]]; then
  python -m scripts.prepare_arc_eval_samples --output "${ARC_DIR}"
fi

python - "${MODEL_DIR}" "${DATASET_DIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

model_dir, dataset_dir = map(Path, sys.argv[1:])
sft = dataset_dir / "tokenized" / "tr-hash-32k-v2-2048"
required_model = ("config.json", "model.safetensors", "model_config.yaml")
missing_model = [name for name in required_model if not (model_dir / name).is_file()]
if missing_model:
    raise SystemExit(f"Missing Refinement model files: {missing_model}")
manifest = json.loads((sft / "manifest.json").read_text(encoding="utf-8"))
train = manifest["partitions"]["train"]
actual = int(manifest.get("actual_unique_formatted_tokens", train["num_tokens"]))
checks = {
    "release_ready": manifest.get("release_quality", {}).get("ready") is True,
    "no_truncation": manifest.get("release_quality", {}).get("token_truncation") is False,
    "unique_target": 500_000_000 <= actual < 500_020_000,
    "vocab_size": manifest.get("tokenizer_vocab_size") == 32_000,
    "sequence_length": manifest.get("sequence_length_cap") == 2_048,
    "chat_eos": manifest.get("chat_template_eos_token") == "</s>",
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Dataset contract failed: {failed}; actual_tokens={actual}")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

hash_failures = []
for partition in ("train", "eval"):
    expected = manifest["partitions"][partition]["files"]
    for filename in ("input_ids.bin", "labels.bin", "examples.jsonl"):
        path = sft / partition / filename
        if not path.is_file():
            hash_failures.append(f"missing {path}")
        elif sha256(path) != expected[filename]:
            hash_failures.append(f"sha256 mismatch {path}")
tokenizer = sft / "tokenizer" / "tokenizer.json"
if sha256(tokenizer) != manifest["tokenizer_sha256"]:
    hash_failures.append("tokenizer/tokenizer.json sha256 mismatch")
if hash_failures:
    raise SystemExit(f"Dataset file verification failed: {hash_failures}")
print(f"[preflight] refinement=step-8156 unique_tokens={actual:,} hashes=verified")
PY

python - <<'PY'
import liger_kernel
import torch
import triton
from complexity.core.losses import has_liger_fused_linear_ce
from complexity.utils.device import get_backend_info

info = get_backend_info("cuda")
if not info.custom_triton:
    raise SystemExit(f"Custom Triton is unavailable: {info}")
if not has_liger_fused_linear_ce():
    raise SystemExit("Liger fused linear cross-entropy is unavailable")
print(f"[preflight] torch={torch.__version__} triton={triton.__version__} liger={liger_kernel.__file__}")
print(f"[preflight] backend={info}")
PY

mkdir -p /workspace/complexity-framework/artifacts
for name in \
  tr_hash_200m_reasoning_sft_500m_full_1e \
  tr_hash_200m_reasoning_sft_500m_hf_sync \
  tr_hash_200m_reasoning_sft_500m_eval; do
  install -m 0644 "deploy/supervisor/${name}.conf" "/etc/supervisor/conf.d/${name}.conf"
done

supervisorctl reread
supervisorctl update
supervisorctl status \
  tr_hash_200m_reasoning_sft_500m_full_1e \
  tr_hash_200m_reasoning_sft_500m_hf_sync \
  tr_hash_200m_reasoning_sft_500m_eval
