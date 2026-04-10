#!/bin/bash
# Setup script for remote GPU instances (Verda, Lambda, RunPod, etc.)
# Usage: scp this script + .env to the server, then run it.
#
#   scp scripts/setup_remote.sh .env user@host:/root/
#   ssh user@host "bash /root/setup_remote.sh"
#
# Complexity-ML — 2026

set -e

echo "=== Complexity Framework — Remote Setup ==="

# Detect workspace
if [ -d "/workspace" ]; then
    WORKDIR="/workspace"
else
    WORKDIR="$HOME"
fi
echo "Workdir: $WORKDIR"

cd "$WORKDIR"

# Clone repo if not present
if [ ! -d "complexity-framework" ]; then
    echo "Cloning complexity-framework..."
    git clone https://github.com/Complexity-ML/complexity-framework.git
else
    echo "Repo exists, pulling latest..."
    cd complexity-framework && git pull && cd ..
fi

cd complexity-framework

# Install pip if needed (Ubuntu 24.04 externally-managed)
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip..."
    apt update && apt install -y python3-pip python3-venv
fi

# Install framework (editable) with all extras
echo "Installing complexity-framework..."
pip3 install --break-system-packages --ignore-installed typing-extensions -e ".[all]" 2>/dev/null \
    || pip3 install --break-system-packages -e ".[all]" 2>/dev/null \
    || pip3 install -e ".[all]"

# Extra dependencies used by training scripts
pip3 install --break-system-packages python-dotenv safetensors "huggingface_hub[cli]" datasets tokenizers 2>/dev/null \
    || pip3 install python-dotenv safetensors "huggingface_hub[cli]" datasets tokenizers

# Setup .env if available
if [ -f "$WORKDIR/.env" ]; then
    cp "$WORKDIR/.env" .env
    echo ".env copied"
elif [ -f "/root/.env" ]; then
    cp /root/.env .env
    echo ".env copied from /root"
fi

# HuggingFace login
if [ -f ".env" ]; then
    HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)
    if [ -n "$HF_TOKEN" ]; then
        huggingface-cli login --token "$HF_TOKEN"
        echo "HuggingFace authenticated"
    fi
fi

# GPU info
echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")
echo ""
echo "=== Ready ==="
echo "  cd $WORKDIR/complexity-framework"
echo ""
echo "  # Pretraining:"
echo "  torchrun --nproc_per_node=$NUM_GPUS scripts/train_hackathon_383m.py --target-tokens 1000000000 --batch-size 64"
echo ""
echo "  # GRPO (download model first):"
echo "  huggingface-cli download Pacific-i64/TR-MoE-400M --local-dir checkpoints/400m"
echo "  torchrun --nproc_per_node=$NUM_GPUS -m complexity.RL.grpo.train_grpo --model_path checkpoints/400m --dataset cais/mmlu --bf16"
echo ""
