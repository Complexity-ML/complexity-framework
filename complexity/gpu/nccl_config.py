"""
NCCL configuration for multi-GPU and multi-node training.

Complexity-ML — 2026
"""

import os
import logging

logger = logging.getLogger(__name__)


def configure_nccl(
    p2p: bool = True,
    ib: bool = True,
    socket_ifname: str = None,
    timeout_s: int = 1800,
) -> None:
    """
    Configure NCCL for optimal multi-GPU communication.

    Args:
        p2p: Enable GPU-to-GPU P2P (NVLink/PCIe direct, faster than host staging)
        ib: Enable InfiniBand (for multi-node, disable if not available)
        socket_ifname: Network interface for NCCL socket (e.g. "eth0", "ib0")
        timeout_s: NCCL operation timeout in seconds (default: 30min for large clusters)

    Environment variables set:
        NCCL_P2P_DISABLE=0          (enable P2P)
        NCCL_IB_DISABLE=0           (enable InfiniBand)
        NCCL_SOCKET_IFNAME=eth0     (network interface)
        NCCL_TIMEOUT=1800           (30min timeout)
        NCCL_ASYNC_ERROR_HANDLING=1 (non-blocking error handling)
    """
    env_vars = {
        "NCCL_P2P_DISABLE": "0" if p2p else "1",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_TIMEOUT": str(timeout_s),
    }

    if not ib:
        env_vars["NCCL_IB_DISABLE"] = "1"

    if socket_ifname:
        env_vars["NCCL_SOCKET_IFNAME"] = socket_ifname

    for key, val in env_vars.items():
        if key not in os.environ:
            os.environ[key] = val

    logger.info(f"[gpu] NCCL configured: p2p={p2p}, ib={ib}, timeout={timeout_s}s")
