"""Optional utilities built on top of the Token-Routed Triton kernels.

Keeping benchmarks and the historical robotics adapter here prevents the
kernel implementation from also becoming a demo and integration module.
Imports of these utilities remain available from
``complexity_cuda.triton_token_routed`` for compatibility.
"""

from __future__ import annotations

import time

import torch


def benchmark_token_routed_mlp(
    batch_size: int = 32,
    seq_len: int = 512,
    hidden_size: int = 1024,
    intermediate_size: int = 4096,
    num_experts: int = 4,
    vocab_size: int = 100000,
    n_iter: int = 100,
):
    """Benchmark CGGR against the historical BMM implementation."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return None

    from .triton_token_routed import TokenRoutedMLPTriton

    device = "cuda"
    cggr_mlp = TokenRoutedMLPTriton(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        vocab_size=vocab_size,
        use_cggr=True,
    ).to(device).eval()
    bmm_mlp = TokenRoutedMLPTriton(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        vocab_size=vocab_size,
        use_cggr=False,
    ).to(device).eval()

    hidden = torch.randn(batch_size, seq_len, hidden_size, device=device)
    token_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        device=device,
    )

    for _ in range(10):
        _ = cggr_mlp(hidden, token_ids)
        _ = bmm_mlp(hidden, token_ids)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        _ = cggr_mlp(hidden, token_ids)
    torch.cuda.synchronize()
    cggr_time = (time.perf_counter() - start) / n_iter * 1000

    start = time.perf_counter()
    for _ in range(n_iter):
        _ = bmm_mlp(hidden, token_ids)
    torch.cuda.synchronize()
    bmm_time = (time.perf_counter() - start) / n_iter * 1000

    print(
        f"\nToken-Routed MLP Benchmark "
        f"(batch={batch_size}, seq={seq_len}, h={hidden_size})"
    )
    print("=" * 60)
    print(f"  BMM:      {bmm_time:.3f} ms (v1)")
    print(f"  CGGR:     {cggr_time:.3f} ms (v2)")
    print(f"  Speedup:  {bmm_time / cggr_time:.2f}x")
    print("=" * 60)
    return cggr_time, bmm_time


class RoboticsTokenRoutedLayer(torch.nn.Module):
    """Historical robotics-style RMSNorm, routed MLP, residual adapter."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        vocab_size: int = 32000,
        eps: float = 1e-6,
    ):
        super().__init__()
        from .triton_token_routed import TokenRoutedMLPTriton

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.eps = eps
        self.norm_weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.mlp = TokenRoutedMLPTriton(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            vocab_size=vocab_size,
            use_cggr=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        from .triton_token_routed import fused_rmsnorm

        residual = x
        x_normed = fused_rmsnorm(x, self.norm_weight, self.eps)
        return residual + self.mlp(x_normed, token_ids=token_ids)


__all__ = ["RoboticsTokenRoutedLayer", "benchmark_token_routed_mlp"]
