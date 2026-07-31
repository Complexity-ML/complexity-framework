"""Deterministic routing helpers used by the legacy Token-Routed MLP.

This module is intentionally free of module parameters and model state.  It
keeps route construction and backend-policy decisions separate from
``TokenRoutedMLP`` while preserving the historical functions bit-for-bit for
checkpoint and ablation replay.
"""

from __future__ import annotations

import itertools

import torch

from ...utils.device import supports_custom_triton


def normalize_cggr_policy(policy: object) -> str:
    """Normalize CGGR policy values accepted by configs and CLIs."""
    if isinstance(policy, str):
        value = policy.strip().lower()
        if value in {"auto", "true", "false"}:
            return value
    if policy is True:
        return "true"
    if policy is False:
        return "false"
    return "auto"


def create_pair_coverage_hash_metadata(
    vocab_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the compact, layer-independent state for pair-coverage hashing."""
    pairs = torch.combinations(
        torch.arange(num_experts, dtype=torch.int32, device="cpu"),
        r=2,
    )
    pair_count = int(pairs.size(0))
    if num_experts < 2 or num_experts > 4:
        raise ValueError("token_id_pair_coverage_hash supports two to four experts")

    token_generator = torch.Generator().manual_seed(0xC04E2A63)
    token_order = torch.randperm(
        vocab_size,
        generator=token_generator,
        device="cpu",
    )
    token_rank = torch.empty(vocab_size, dtype=torch.int32, device="cpu")
    token_rank[token_order] = torch.arange(
        vocab_size,
        dtype=torch.int32,
        device="cpu",
    )

    # Every permutation has one unique cyclic rotation beginning with pair
    # zero. The historical constants and construction order are part of the
    # checkpoint-replay contract and must not change.
    base_orders = torch.tensor(
        [
            (0, *tail)
            for tail in itertools.permutations(range(1, pair_count))
        ],
        dtype=torch.int32,
        device="cpu",
    )
    base_generator = torch.Generator().manual_seed(0xC04E2A63 ^ 0x51A7)
    base_orders = base_orders[
        torch.randperm(
            base_orders.size(0),
            generator=base_generator,
            device="cpu",
        )
    ]
    return token_rank, pairs, base_orders


def pair_coverage_hash_routes_from_metadata(
    token_rank: torch.Tensor,
    pairs: torch.Tensor,
    base_orders: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Decode one layer's equal-weight expert pair from compact hash state."""
    pair_count = int(pairs.size(0))
    group_rank = torch.div(token_rank, pair_count, rounding_mode="floor")
    rotation = token_rank.remainder(pair_count)

    if layer_idx < pair_count:
        base_idx = group_rank.remainder(base_orders.size(0))
        position = (rotation + layer_idx).remainder(pair_count)
        pair_idx = base_orders[base_idx, position]
    else:
        route_code = torch.div(
            group_rank,
            base_orders.size(0),
            rounding_mode="floor",
        )
        digit_place = pair_count ** (layer_idx - pair_count)
        route_digit = torch.div(
            route_code,
            digit_place,
            rounding_mode="floor",
        ).remainder(pair_count)
        pair_idx = (rotation + route_digit).remainder(pair_count)

    selected = pairs[pair_idx.long()].long()
    swap = (token_rank.long() + int(layer_idx)).remainder(2).bool()
    primary = torch.where(swap, selected[:, 1], selected[:, 0])
    secondary = torch.where(swap, selected[:, 0], selected[:, 1])
    return torch.stack((primary, secondary), dim=0)


def create_pair_coverage_hash_routes(
    vocab_size: int,
    num_experts: int,
    layer_idx: int,
) -> torch.Tensor:
    """Return two token-ID routes with full early-layer pair coverage."""
    metadata = create_pair_coverage_hash_metadata(vocab_size, num_experts)
    return pair_coverage_hash_routes_from_metadata(*metadata, layer_idx)


def encode_pair_coverage_hash_routes(
    routes: torch.Tensor,
    expert_pairs: torch.Tensor,
) -> torch.Tensor:
    """Compile two expert routes into one byte per vocabulary entry."""
    if routes.ndim != 2 or routes.size(0) != 2:
        raise ValueError("routes must be shaped [2, vocab_size]")
    unordered = torch.sort(routes.long(), dim=0).values
    pair_matches = (
        (unordered[0].unsqueeze(0) == expert_pairs[:, 0].long().unsqueeze(1))
        & (unordered[1].unsqueeze(0) == expert_pairs[:, 1].long().unsqueeze(1))
    )
    if not torch.all(pair_matches.any(dim=0)):
        raise ValueError("routes contain a pair absent from expert_pairs")
    pair_indices = pair_matches.to(torch.int64).argmax(dim=0)
    swap = routes[0].long().eq(unordered[1]).to(torch.int64)
    return (pair_indices | (swap << 3)).to(torch.uint8)


def cggr_dispatch_decision(
    *,
    cggr_policy: object,
    kernel_policy: object,
    is_cuda: bool,
    has_cggr: bool,
    has_autograd: bool,
    static_dispatch: bool = False,
    supports_triton_fn=None,
) -> tuple[bool, list[str]]:
    """Return whether Token-Routed should use CGGR and why it may not."""
    mode = normalize_cggr_policy(cggr_policy)
    reasons: list[str] = []
    supports_triton_fn = supports_triton_fn or supports_custom_triton

    if mode == "false":
        reasons.append("config.use_cggr=False")
    if static_dispatch:
        reasons.append("static_expert_capacity=True")
    if not supports_triton_fn(kernel_policy):
        reasons.append(f"supports_custom_triton(policy={kernel_policy!r})=False")
    if not has_cggr:
        reasons.append("HAS_CGGR=False (complexity_cuda triton import failed)")
    if not is_cuda:
        reasons.append("flat_x.is_cuda=False")
    if not has_autograd:
        reasons.append("cggr_grouped_gemm_autograd=None")

    return mode != "false" and not reasons, reasons


# Private aliases preserve imports used by historical tests and supplements.
_normalize_cggr_policy = normalize_cggr_policy
_create_pair_coverage_hash_metadata = create_pair_coverage_hash_metadata
_pair_coverage_hash_routes_from_metadata = pair_coverage_hash_routes_from_metadata
_create_pair_coverage_hash_routes = create_pair_coverage_hash_routes
_encode_pair_coverage_hash_routes = encode_pair_coverage_hash_routes


__all__ = [
    "cggr_dispatch_decision",
    "create_pair_coverage_hash_metadata",
    "create_pair_coverage_hash_routes",
    "encode_pair_coverage_hash_routes",
    "normalize_cggr_policy",
    "pair_coverage_hash_routes_from_metadata",
]
