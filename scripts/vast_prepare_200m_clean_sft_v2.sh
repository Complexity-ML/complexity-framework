#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

MODEL_REPO="AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement"
MODEL_REVISION="ad4e9217b637720fb939babe8c8ce285a804ade2"
MODEL_DIR="/workspace/tr-hash-refinement"
DATASET_REPO="AETHORIA-AI/TR-HASH-MoE-200M-SFT-v2-300K"
DATASET_REVISION="084a658ec47e4ee872f6d67fdbad3602f599424b"
DATASET_DIR="/workspace/tr-hash-moe-200m-sft-v2-300k"
PIQA_DIR="/workspace/physicaliqa-train-dev/physicaliqa-train-dev"
OUTPUT_ROOT="/workspace/complexity-framework/artifacts/tr_hash_moe_200m_clean_sft_v2_full_3e"

if [[ ! -s /workspace/.hf_token ]]; then
  echo "Missing /workspace/.hf_token; create it with mode 600 before bootstrap." >&2
  exit 2
fi
export HF_TOKEN="$(< /workspace/.hf_token)"
export HF_XET_HIGH_PERFORMANCE=1
umask 077

python - <<'PY'
import shutil
import os
import torch

count = torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(count)]
if count not in {4, 8}:
    raise SystemExit(f"Expected exactly 4 or 8 CUDA GPUs, found {count}: {names}")
expected = os.environ.get("EXPECTED_GPU_COUNT")
if expected is not None and count != int(expected):
    raise SystemExit(f"EXPECTED_GPU_COUNT={expected}, but found {count}: {names}")
if not all("5090" in name for name in names):
    raise SystemExit(f"Expected only RTX 5090 GPUs, found: {names}")
free = shutil.disk_usage("/workspace").free
if free < 15 * 1024**3:
    raise SystemExit(f"At least 15 GiB free is required; found {free / 1024**3:.1f} GiB")
print(f"[preflight] cuda_gpus={count} names={names} free_disk_gib={free / 1024**3:.1f}")
PY

# Preserve the CUDA-enabled torch supplied by the Vast image. The framework
# intentionally does not depend on torch, and Liger is installed explicitly.
python -m pip install -e .
python -m pip install "huggingface_hub[hf_xet]>=0.32.0" "liger-kernel>=0.5.0"

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
        "manifest.json",
        "metadata/release-audit.json",
        "tokenized/tr-hash-32k-v2-2048/manifest.json",
        "tokenized/tr-hash-32k-v2-2048/chat_template.json",
        "tokenized/tr-hash-32k-v2-2048/tokenizer/**",
        "tokenized/tr-hash-32k-v2-2048/train/input_ids.bin",
        "tokenized/tr-hash-32k-v2-2048/train/labels.bin",
        "tokenized/tr-hash-32k-v2-2048/train/examples.jsonl",
        "tokenized/tr-hash-32k-v2-2048/train/sft.idx.json",
        "tokenized/tr-hash-32k-v2-2048/eval/input_ids.bin",
        "tokenized/tr-hash-32k-v2-2048/eval/labels.bin",
        "tokenized/tr-hash-32k-v2-2048/eval/examples.jsonl",
        "tokenized/tr-hash-32k-v2-2048/eval/sft.idx.json",
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
    url = (
        "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/"
        "physicaliqa-train-dev.zip"
    )
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(target.parent)
if not all(path.is_file() for path in required):
    raise SystemExit(f"PIQA probe download is incomplete under {target}")
examples = sum(1 for _ in required[0].open(encoding="utf-8"))
labels = sum(1 for _ in required[1].open(encoding="utf-8"))
if examples != 1_838 or labels != 1_838:
    raise SystemExit(f"Expected 1,838 PIQA examples and labels, found {examples}/{labels}")
print("[preflight] piqa_validation_examples=1838")
PY

python - "${MODEL_DIR}" "${DATASET_DIR}" "${OUTPUT_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

model_dir, dataset_dir, output_root = map(Path, sys.argv[1:])
required_model = {
    "config.json",
    "model.safetensors",
    "model_config.yaml",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
}
missing_model = sorted(name for name in required_model if not (model_dir / name).is_file())
if missing_model:
    raise SystemExit(f"Missing model files: {missing_model}")
if (model_dir / "checkpoints").exists() or any(model_dir.glob("optimizer*")):
    raise SystemExit("Refusing duplicated checkpoint/optimizer payload in minimal model download")

sft = dataset_dir / "tokenized" / "tr-hash-32k-v2-2048"
manifest = json.loads((sft / "manifest.json").read_text(encoding="utf-8"))
checks = {
    "release_ready": manifest.get("release_quality", {}).get("ready") is True,
    "no_truncation": manifest.get("release_quality", {}).get("token_truncation") is False,
    "train_examples": manifest.get("partitions", {}).get("train", {}).get("examples") == 300_000,
    "eval_examples": manifest.get("partitions", {}).get("eval", {}).get("examples") == 3_000,
    "vocab_size": manifest.get("tokenizer_vocab_size") == 32_000,
    "sequence_length": manifest.get("sequence_length_cap") == 2_048,
    "chat_eos": manifest.get("chat_template_eos_token") == "</s>",
}
required_dataset = [
    sft / "train" / "input_ids.bin",
    sft / "train" / "labels.bin",
    sft / "train" / "examples.jsonl",
    sft / "train" / "sft.idx.json",
    sft / "eval" / "input_ids.bin",
    sft / "eval" / "labels.bin",
    sft / "eval" / "examples.jsonl",
    sft / "eval" / "sft.idx.json",
    sft / "tokenizer" / "tokenizer.json",
    sft / "tokenizer" / "chat_template.jinja",
]
missing_dataset = [str(path) for path in required_dataset if not path.is_file()]
failed = [name for name, passed in checks.items() if not passed]
if failed or missing_dataset:
    raise SystemExit(f"Dataset verification failed checks={failed} missing={missing_dataset}")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

hash_failures = []
for partition in ("train", "eval"):
    expected_files = manifest["partitions"][partition]["files"]
    for filename in ("input_ids.bin", "labels.bin", "examples.jsonl"):
        path = sft / partition / filename
        actual = sha256(path)
        expected = expected_files[filename]
        if actual != expected:
            hash_failures.append(f"{partition}/{filename}: {actual} != {expected}")
tokenizer_path = sft / "tokenizer" / "tokenizer.json"
actual_tokenizer_hash = sha256(tokenizer_path)
if actual_tokenizer_hash != manifest["tokenizer_sha256"]:
    hash_failures.append(
        f"tokenizer/tokenizer.json: {actual_tokenizer_hash} != {manifest['tokenizer_sha256']}"
    )
if hash_failures:
    raise SystemExit(f"Dataset SHA256 verification failed: {hash_failures}")

if output_root.exists():
    stale = []
    for pattern in ("step_*", "best", "final", "final_*", "interrupted_*", "token_pack_*"):
        stale.extend(output_root.glob(pattern))
    if stale:
        raise SystemExit(f"Refusing stale SFT output artifacts: {[str(path) for path in stale]}")

print("[preflight] model=refinement-step-8156 dataset=sft-v2-300k revision-locked hashes=verified")
print("[preflight] train=300000 eval=3000 vocab=32000 seq=2048 eos=</s> truncation=false")
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

# Supervisor validates every stdout/stderr parent directory during `reread`.
# Create the shared artifacts directory before installing any program config;
# the training launcher still owns creation of its run-specific output folder.
mkdir -p /workspace/complexity-framework/artifacts

install -m 0644 \
  deploy/supervisor/tr_hash_200m_clean_sft_v2_full_3e.conf \
  /etc/supervisor/conf.d/tr_hash_200m_clean_sft_v2_full_3e.conf
install -m 0644 \
  deploy/supervisor/tr_hash_200m_clean_sft_v2_hf_sync.conf \
  /etc/supervisor/conf.d/tr_hash_200m_clean_sft_v2_hf_sync.conf
install -m 0644 \
  deploy/supervisor/tr_hash_200m_clean_sft_v2_eval.conf \
  /etc/supervisor/conf.d/tr_hash_200m_clean_sft_v2_eval.conf

supervisorctl reread
supervisorctl update
supervisorctl status \
  tr_hash_200m_clean_sft_v2_full_3e \
  tr_hash_200m_clean_sft_v2_hf_sync \
  tr_hash_200m_clean_sft_v2_eval
