#!/usr/bin/env bash
# Multi-node torchrun launcher for AMD ROCm.
#
# Designed for the AMD Developer Cloud / TensorWave / Hot Aisle topology:
#   - 8x MI300X per node
#   - inter-node fabric is either RoCE (Mellanox) or TCP over 100GbE/200GbE
#
# Run this on EVERY node — pass --node-rank 0 on the master node, 1 on the
# second node, etc. The master node's IP must be reachable from the others
# on $MASTER_PORT.
#
# Usage:
#   # On master (node 0):
#   ./scripts/launch_rocm_multinode.sh \
#       --nnodes 2 --node-rank 0 --master-addr 10.0.0.5 \
#       --config configs/run_configs/1b_o200k_tr_1T_rocm_16gpu.yaml
#
#   # On second node:
#   ./scripts/launch_rocm_multinode.sh \
#       --nnodes 2 --node-rank 1 --master-addr 10.0.0.5 \
#       --config configs/run_configs/1b_o200k_tr_1T_rocm_16gpu.yaml

set -euo pipefail

NNODES=1
NODE_RANK=0
MASTER_ADDR=""
MASTER_PORT="29500"
NPROC_PER_NODE=""
CONFIG=""
SCRIPT="-m complexity.training.o200k_pretrain"
SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
IB_HCA="${NCCL_IB_HCA:-}"
RCCL_DEBUG="${COMPLEXITY_RCCL_DEBUG:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nnodes)          NNODES="$2"; shift 2 ;;
    --node-rank)       NODE_RANK="$2"; shift 2 ;;
    --master-addr)     MASTER_ADDR="$2"; shift 2 ;;
    --master-port)     MASTER_PORT="$2"; shift 2 ;;
    --nproc-per-node)  NPROC_PER_NODE="$2"; shift 2 ;;
    --config)          CONFIG="$2"; shift 2 ;;
    --script)          SCRIPT="$2"; shift 2 ;;
    --socket-ifname)   SOCKET_IFNAME="$2"; shift 2 ;;
    --ib-hca)          IB_HCA="$2"; shift 2 ;;
    --debug)           RCCL_DEBUG="1"; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

# Auto-detect GPUs per node if not passed.
if [[ -z "$NPROC_PER_NODE" ]]; then
  if command -v rocm-smi >/dev/null 2>&1; then
    NPROC_PER_NODE=$(rocm-smi --showid 2>/dev/null | grep -cE '^GPU\[[0-9]+\]' || true)
  fi
  if [[ -z "$NPROC_PER_NODE" || "$NPROC_PER_NODE" == "0" ]]; then
    NPROC_PER_NODE=8
  fi
fi

if [[ "$NNODES" -gt 1 && -z "$MASTER_ADDR" ]]; then
  echo "ERROR: --master-addr is required when --nnodes > 1" >&2
  exit 1
fi
if [[ "$NNODES" == "1" && -z "$MASTER_ADDR" ]]; then
  MASTER_ADDR="127.0.0.1"
fi

# ── ROCm / RCCL environment ──────────────────────────────────────────────────
# These are exported so they reach every torchrun child process AND every
# Python rank. configure_rccl() in the framework will not override them.
export NNODES
export MASTER_ADDR
export MASTER_PORT

# Pin the NIC for the RCCL bootstrap socket. Without this, RCCL probes
# every interface (including docker0/lo) and can pick the wrong one. If
# you don't know the right name: `ip -br link` on the node.
if [[ -n "$SOCKET_IFNAME" ]]; then
  export NCCL_SOCKET_IFNAME="$SOCKET_IFNAME"
fi

# IB HCA pinning (RoCE / InfiniBand). Leave unset if the cluster is TCP-only;
# the framework's configure_rccl() will then set NCCL_IB_DISABLE=1.
if [[ -n "$IB_HCA" ]]; then
  export NCCL_IB_HCA="$IB_HCA"
fi

if [[ "$RCCL_DEBUG" == "1" ]]; then
  export COMPLEXITY_RCCL_DEBUG=1
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
fi

# Common ROCm-side env. HSA_FORCE_FINE_GRAIN_PCIE is required for stable
# multi-node xGMI on MI300X; set early so torchrun children inherit it.
if [[ "$NNODES" -gt 1 ]]; then
  export HSA_FORCE_FINE_GRAIN_PCIE=1
fi

# Make HIP-visible devices match LOCAL_RANK that torchrun emits.
# (Most ROCm containers already do this; harmless if duplicated.)
unset ROCR_VISIBLE_DEVICES || true

# ── Launch ───────────────────────────────────────────────────────────────────
WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))
export WORLD_SIZE

echo "[launch] nnodes=$NNODES node_rank=$NODE_RANK nproc_per_node=$NPROC_PER_NODE world=$WORLD_SIZE"
echo "[launch] master=$MASTER_ADDR:$MASTER_PORT"
echo "[launch] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<auto>} NCCL_IB_HCA=${NCCL_IB_HCA:-<unset>}"
echo "[launch] script=$SCRIPT config=$CONFIG"

CMD=(torchrun
  --nnodes "$NNODES"
  --node_rank "$NODE_RANK"
  --nproc_per_node "$NPROC_PER_NODE"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
)

# torchrun expects either a script path or '-m module.path'. Split on first space.
if [[ "$SCRIPT" == -m\ * ]]; then
  CMD+=(${SCRIPT})
else
  CMD+=("$SCRIPT")
fi

if [[ -n "$CONFIG" ]]; then
  CMD+=(--config "$CONFIG")
fi

exec "${CMD[@]}"
