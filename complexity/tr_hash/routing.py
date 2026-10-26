"""Zero-history routing anchor and compact pair metadata utilities."""

from __future__ import annotations

import torch

from .composed_routing import composed_mlp_routes
from .config import TRHashEngineConfig


def build_route_table(config: TRHashEngineConfig) -> torch.Tensor:
    """Return the zero-history anchor; actual routes use the causal sequence."""
    token_ids = torch.arange(config.vocab_size, dtype=torch.long, device="cpu").unsqueeze(1)
    return composed_mlp_routes(token_ids, config.layer_index).squeeze(-1)


def compile_top2_pair_metadata(
    routes: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compile a top-2 route table into compact CUDA dispatch metadata.

    The hash-native CUDA partition uses the explicit 3-bit pair index plus one
    orientation bit for up to four experts. Eight experts require 28 pairs, so
    they use a 5-bit pair index plus one orientation bit. Both layouts fit in
    one byte and remain distinguishable from the pair-table length, preserving
    existing checkpoint metadata.
    """

    if routes.ndim != 2 or routes.size(0) != 2:
        raise ValueError("routes must be shaped [2, vocab_size]")
    if not 2 <= num_experts <= 8:
        raise ValueError("compact pair metadata supports two to eight experts")
    if routes.numel() and (routes.min().item() < 0 or routes.max().item() >= num_experts):
        raise ValueError("routes contain an invalid expert index")

    expert_pairs = torch.combinations(
        torch.arange(num_experts, dtype=torch.int32, device="cpu"),
        r=2,
    )
    pair_count = int(expert_pairs.size(0))
    if pair_count > 32:
        raise ValueError("compact pair encoding supports at most 32 pairs")

    routes_cpu = routes.detach().to(device="cpu", dtype=torch.long)
    unordered = routes_cpu.sort(dim=0).values
    if torch.any(unordered[0] == unordered[1]):
        raise ValueError("top-2 pair routes must select two distinct experts")
    pair_matches = (unordered[0].unsqueeze(0) == expert_pairs[:, 0].long().unsqueeze(1)) & (
        unordered[1].unsqueeze(0) == expert_pairs[:, 1].long().unsqueeze(1)
    )
    if not torch.all(pair_matches.any(dim=0)):
        raise ValueError("routes contain an expert pair that cannot be encoded")

    pair_indices = pair_matches.to(torch.int64).argmax(dim=0)
    swap = routes_cpu[0].eq(unordered[1]).to(torch.int64)
    swap_bit = 0x8 if pair_count <= 8 else 0x20
    route_codes = (pair_indices | (swap * swap_bit)).to(torch.uint8)
    return route_codes, expert_pairs


def decode_top2_pair_metadata(
    route_codes: torch.Tensor,
    expert_pairs: torch.Tensor,
) -> torch.Tensor:
    """Decode compact pair metadata to the authoritative ``[2, vocab]`` form."""

    if route_codes.ndim != 1 or route_codes.dtype != torch.uint8:
        raise ValueError("route_codes must be a one-dimensional uint8 tensor")
    if expert_pairs.ndim != 2 or expert_pairs.size(1) != 2:
        raise ValueError("expert_pairs must be shaped [pair_count, 2]")
    codes = route_codes.to(torch.int64)
    pair_count = int(expert_pairs.size(0))
    pair_mask, swap_mask = (0x7, 0x8) if pair_count <= 8 else (0x1F, 0x20)
    pair_indices = codes.bitwise_and(pair_mask)
    if pair_indices.numel() and pair_indices.max().item() >= expert_pairs.size(0):
        raise ValueError("route_codes reference an unavailable expert pair")
    selected = expert_pairs[pair_indices.long()].long()
    swap = codes.bitwise_and(swap_mask).ne(0)
    return torch.stack(
        (
            torch.where(swap, selected[:, 1], selected[:, 0]),
            torch.where(swap, selected[:, 0], selected[:, 1]),
        ),
        dim=0,
    )
