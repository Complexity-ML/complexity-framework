#!/bin/bash
# =============================================================================
# Cluster Launcher Examples — Complexity Framework
#
# Choose your launcher based on what the cluster provides.
# All examples: 2 nodes × 8 GPUs, 1B MoE model (scripts/train_1b_moe_v1.py).
#
# Complexity-ML — 2026
# =============================================================================

# ── Option 1: SLURM (most HPC clusters) ─────────────────────────────────
#
# Submit with: sbatch scripts/launcher_example.sh
#
#SBATCH --job-name=complexity-1b-moe
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/train_%j.log

# SLURM auto-sets: MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK
# srun launches one process per node, torchrun spawns 8 GPU workers per node

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1) \
    --master_port=29500 \
    --node_rank=$SLURM_NODEID \
    scripts/train_1b_moe_v1.py


# ── Option 2: pdsh / SSH (bare metal, cloud VMs) ────────────────────────
#
# Requirements: passwordless SSH between nodes
# Run from node 0:
#
#   bash scripts/launcher_example.sh pdsh
#

launch_pdsh() {
    MASTER_ADDR=$(hostname -I | awk '{print $1}')
    NODES="node0,node1"  # <-- edit with your hostnames/IPs
    NNODES=2
    NPROC=8

    echo "Master: $MASTER_ADDR"
    echo "Nodes: $NODES"

    pdsh -w $NODES "
        cd ~/complexity-framework &&
        export PATH=\$HOME/.local/bin:\$PATH &&
        torchrun \
            --nnodes=$NNODES \
            --nproc_per_node=$NPROC \
            --master_addr=$MASTER_ADDR \
            --master_port=29500 \
            --node_rank=\$(echo $NODES | tr ',' '\n' | grep -n \$(hostname) | cut -d: -f1 | head -1) \
            scripts/train_1b_moe_v1.py
    "
}


# ── Option 3: Manual SSH (simplest, 2 terminals) ────────────────────────
#
# Terminal 1 (node 0):
#   torchrun --nnodes=2 --nproc_per_node=8 \
#       --master_addr=<NODE0_IP> --master_port=29500 --node_rank=0 \
#       scripts/train_1b_moe_v1.py
#
# Terminal 2 (node 1):
#   torchrun --nnodes=2 --nproc_per_node=8 \
#       --master_addr=<NODE0_IP> --master_port=29500 --node_rank=1 \
#       scripts/train_1b_moe_v1.py


# ── Option 4: Docker / Kubernetes ────────────────────────────────────────
#
# Dockerfile:
#   FROM nvcr.io/nvidia/pytorch:24.12-py3
#   RUN pip install complexity-framework
#   COPY scripts/ /app/scripts/
#   COPY tokenizer/ /app/tokenizer/
#   ENTRYPOINT ["torchrun"]
#
# Kubernetes (Volcano / KubeFlow):
#   apiVersion: batch.volcano.sh/v1alpha1
#   kind: Job
#   spec:
#     tasks:
#       - replicas: 2
#         template:
#           spec:
#             containers:
#               - name: worker
#                 image: complexity-ml/train:latest
#                 args: ["--nnodes=2", "--nproc_per_node=8", "scripts/train_1b_moe_v1.py"]
#                 resources:
#                   limits:
#                     nvidia.com/gpu: 8


# ── Option 5: Single node (test/dev) ────────────────────────────────────
#
#   torchrun --nproc_per_node=8 scripts/train_1b_moe_v1.py
#
# Or even single GPU:
#
#   python scripts/train_1b_moe_v1.py --batch-size 64


# ── Dispatcher ───────────────────────────────────────────────────────────

case "${1:-slurm}" in
    pdsh)   launch_pdsh ;;
    *)      echo "For SLURM: sbatch scripts/launcher_example.sh"
            echo "For pdsh:  bash scripts/launcher_example.sh pdsh"
            echo "For SSH:   see comments in this file"
            ;;
esac
