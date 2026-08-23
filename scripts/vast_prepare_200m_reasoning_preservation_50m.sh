#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/complexity-framework

if [[ ! -s /workspace/.hf_token ]]; then
  echo "Missing /workspace/.hf_token" >&2
  exit 2
fi
export HF_TOKEN="$(< /workspace/.hf_token)"
export HF_XET_HIGH_PERFORMANCE=1
umask 077

python - <<'PY'
import shutil
import torch

names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 4 or not all("5090" in name for name in names):
    raise SystemExit(f"Expected 4 RTX 5090 GPUs, found {names}")
free = shutil.disk_usage("/workspace").free
if free < 35 * 1024**3:
    raise SystemExit(f"At least 35 GiB free required, found {free / 1024**3:.1f}")
print(f"[preflight] GPUs={names} free_disk={free / 1024**3:.1f}GiB")
PY

python -m pip install -e .
python -m pip install "huggingface_hub[hf_xet]>=0.32.0" "liger-kernel>=0.5.0"

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT",
    local_dir="/workspace/tr-hash-sft-v2",
    allow_patterns=[
        "config.json", "model.safetensors", "model_config.yaml", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
    ],
    token=True,
)
snapshot_download(
    repo_id="AETHORIA-AI/TR-HASH-MoE-200M-SFT-v2-300K",
    repo_type="dataset",
    local_dir="/workspace/tr-hash-moe-200m-sft-v2-300k",
    allow_patterns=["manifest.json", "tokenized/tr-hash-32k-v2-2048/**"],
    token=True,
)
snapshot_download(
    repo_id="AETHORIA-AI/TR-HASH-MoE-200M-Reasoning-SFT-500M",
    repo_type="dataset",
    local_dir="/workspace/tr-hash-moe-200m-reasoning-sft-500m",
    allow_patterns=["manifest.json", "tokenized/tr-hash-32k-v2-2048/**"],
    token=True,
)
PY

MIX=/workspace/tr-hash-reasoning-preservation-50m-mix
if [[ ! -s "${MIX}/manifest.json" ]]; then
  python scripts/build_tr_hash_200m_reasoning_preservation_50m.py \
    --general-shard /workspace/tr-hash-moe-200m-sft-v2-300k/tokenized/tr-hash-32k-v2-2048 \
    --reasoning-shard /workspace/tr-hash-moe-200m-reasoning-sft-500m/tokenized/tr-hash-32k-v2-2048 \
    --output "${MIX}"
fi

python - <<'PY'
from complexity.core.losses import has_liger_fused_linear_ce
from complexity.utils.device import get_backend_info

info = get_backend_info("cuda")
if not info.custom_triton or not has_liger_fused_linear_ce():
    raise SystemExit(f"Production kernels unavailable: {info}")
print(f"[preflight] kernels={info}")
PY

mkdir -p artifacts
for name in \
  tr_hash_200m_reasoning_preservation_50m_full_1e \
  tr_hash_200m_reasoning_preservation_50m_hf_sync; do
  install -m 0644 "deploy/supervisor/${name}.conf" "/etc/supervisor/conf.d/${name}.conf"
done
supervisorctl reread
supervisorctl update
supervisorctl status \
  tr_hash_200m_reasoning_preservation_50m_full_1e \
  tr_hash_200m_reasoning_preservation_50m_hf_sync
