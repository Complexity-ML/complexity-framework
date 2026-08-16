"""
RCCL configuration for AMD ROCm multi-GPU and multi-node training.

ROCm uses RCCL (ROCm Communication Collectives Library) as its NCCL-compatible
collective backend. PyTorch exposes it through the same `nccl` backend name,
but most tuning knobs live behind RCCL-prefixed env vars *or* the NCCL_* names
that RCCL also honors.

This module mirrors `nccl_config.py` but with AMD-specific defaults:
  - GDR (GPU Direct RDMA) levels via NCCL_NET_GDR_LEVEL
  - HSA_FORCE_FINE_GRAIN_PCIE for stable P2P across MI300X xGMI links
  - NCCL_IB_HCA / NCCL_SOCKET_IFNAME for multi-node fabric selection
  - Longer timeouts (multi-node TCP/IB init can take >120s on cold start)

Complexity-ML — 2026
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def configure_rccl(
    *,
    multi_node: bool = False,
    socket_ifname: Optional[str] = None,
    ib_hca: Optional[str] = None,
    gdr_level: str = "PHB",
    timeout_s: int = 1800,
    p2p: bool = True,
    debug: bool = False,
) -> None:
    """
    Configure RCCL for AMD MI250/MI300 training.

    Args:
        multi_node: True when launching across >1 host. Enables IB plugin,
            forces fine-grain PCIe, and raises the heartbeat timeout.
        socket_ifname: NIC name for RCCL bootstrap (e.g. "ens6np0", "ibp0s0f0").
            On AMD cloud the right NIC is rarely eth0 — leaving this unset
            lets RCCL probe, but on multi-NIC nodes you almost always want to
            pin it explicitly.
        ib_hca: InfiniBand HCA(s) to use, e.g. "mlx5_0,mlx5_1". Only set if
            the node actually has IB/RoCE; otherwise leave None and RCCL will
            fall back to TCP via `socket_ifname`.
        gdr_level: GPU Direct RDMA level. "PHB" works on most MI300X nodes;
            "SYS" disables GDR (slowest, safest fallback).
        timeout_s: NCCL heartbeat timeout. 1800s (30min) is the safe default
            for multi-node — first all_reduce after dataset shard init can
            take minutes on cold caches.
        p2p: Enable xGMI/PCIe P2P between GPUs within a node.
        debug: Set NCCL_DEBUG=INFO to dump topology / fabric selection at init.

    All env vars are only set if not already present, so user overrides via
    `export NCCL_...` on the launch command stick.
    """
    env: dict[str, str] = {
        # Async error handling so a single-rank crash propagates instead of
        # leaving the other ranks hanging on a collective.
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_TIMEOUT": str(timeout_s),
        # P2P toggle (xGMI between MI300X within a node).
        "NCCL_P2P_DISABLE": "0" if p2p else "1",
    }

    if multi_node:
        # RDMA / IB
        if ib_hca:
            env["NCCL_IB_HCA"] = ib_hca
            env["NCCL_IB_DISABLE"] = "0"
        else:
            # No IB HCA pin → assume TCP fabric, disable IB probing to avoid
            # 30-second timeouts when the IB plugin is loaded but unusable.
            env["NCCL_IB_DISABLE"] = "1"

        # GPU Direct RDMA level (only relevant when IB is up, harmless otherwise).
        env["NCCL_NET_GDR_LEVEL"] = gdr_level

        # Fine-grain PCIe is required for stable RCCL multi-node on MI300X.
        # Without it, cross-node sends can intermittently stall.
        env["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"

    if socket_ifname:
        env["NCCL_SOCKET_IFNAME"] = socket_ifname

    if debug:
        env["NCCL_DEBUG"] = "INFO"
        env["NCCL_DEBUG_SUBSYS"] = "INIT,NET,ENV"

    applied = []
    for key, val in env.items():
        if key not in os.environ:
            os.environ[key] = val
            applied.append(f"{key}={val}")

    if applied:
        logger.info("[gpu] RCCL configured (%s): %s",
                    "multi-node" if multi_node else "single-node",
                    " ".join(applied))
    else:
        logger.info("[gpu] RCCL: all env vars already set by launcher, leaving as-is")
