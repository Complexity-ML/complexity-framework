"""Deterministic route-table construction for TR-Hash."""

from __future__ import annotations

import torch

from .config import TRHashEngineConfig, TRHashStrategy


def _layer_permutation(num_experts: int, layer_index: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 104729 * int(layer_index))
    return torch.randperm(num_experts, generator=generator, device="cpu")


def _balanced_hash_routes(config: TRHashEngineConfig) -> torch.Tensor:
    """Assign collision-free tuples with exact marginal balance.

    Every complete ``num_experts``-token rank block uses each expert exactly
    once in every route position. A deterministic odd stride varies the tuple
    geometry between blocks while remaining collision-free for the supported
    power-of-two expert counts.
    """

    token_generator = torch.Generator(device="cpu").manual_seed(
        int(config.route_seed) + 104729 * int(config.layer_index)
    )
    token_order = torch.randperm(
        config.vocab_size,
        generator=token_generator,
        device="cpu",
    )
    rank = torch.arange(
        config.vocab_size,
        dtype=torch.long,
        device="cpu",
    )
    group = torch.div(rank, config.num_experts, rounding_mode="floor")
    position = rank.remainder(config.num_experts)
    odd_strides = torch.arange(
        1,
        config.num_experts,
        2,
        dtype=torch.long,
        device="cpu",
    )
    stride_index = _mix32(
        group + int(config.route_seed) + int(config.layer_index) * 0x9E3779B1
    ).remainder(odd_strides.numel())
    stride = odd_strides[stride_index]
    expert_permutation = _layer_permutation(
        config.num_experts,
        config.layer_index,
        config.route_seed ^ 0x5EC0D,
    )
    ranked_routes = torch.stack(
        tuple(
            expert_permutation[(position + route_index * stride).remainder(config.num_experts)]
            for route_index in range(config.top_k)
        ),
        dim=0,
    )
    routes = torch.empty(
        config.top_k,
        config.vocab_size,
        dtype=torch.long,
        device="cpu",
    )
    routes[:, token_order] = ranked_routes
    return routes


def _modulo_cyclic_routes(config: TRHashEngineConfig) -> torch.Tensor:
    primary = torch.arange(config.vocab_size, dtype=torch.long, device="cpu") % config.num_experts
    primary = _layer_permutation(
        config.num_experts,
        config.layer_index,
        config.route_seed ^ 0xC0DE,
    )[primary]
    return torch.stack(
        tuple(
            (primary + route_index).remainder(config.num_experts)
            for route_index in range(config.top_k)
        ),
        dim=0,
    )


def _mix32(values: torch.Tensor) -> torch.Tensor:
    values = values.bitwise_and(0xFFFFFFFF)
    values = (values ^ (values >> 16)).bitwise_and(0xFFFFFFFF)
    values = (values * 0x7FEB352D).bitwise_and(0xFFFFFFFF)
    values = (values ^ (values >> 15)).bitwise_and(0xFFFFFFFF)
    values = (values * 0x846CA68B).bitwise_and(0xFFFFFFFF)
    return (values ^ (values >> 16)).bitwise_and(0xFFFFFFFF)


def _affine_hash_routes(config: TRHashEngineConfig) -> torch.Tensor:
    """Compute compact power-of-two routes with an odd, collision-free stride."""

    token_ids = torch.arange(
        config.vocab_size,
        dtype=torch.int64,
        device="cpu",
    )
    layer_salt = (int(config.route_seed) + int(config.layer_index) * 0x9E3779B1) & 0xFFFFFFFF
    base = _mix32(token_ids ^ layer_salt).remainder(config.num_experts)
    stride = _mix32(token_ids + (layer_salt ^ 0x85EBCA77))
    # All supported expert counts are powers of two. An odd stride visits
    # distinct experts for top-1/2/4 without a collision-repair loop.
    stride = stride.bitwise_or(1).remainder(config.num_experts)
    return torch.stack(
        tuple(
            (base + route_index * stride).remainder(config.num_experts)
            for route_index in range(config.top_k)
        ),
        dim=0,
    ).long()


def build_route_table(config: TRHashEngineConfig) -> torch.Tensor:
    """Return a CPU ``[top_k, vocab_size]`` route table."""

    if config.routing_strategy is TRHashStrategy.BALANCED_HASH:
        routes = _balanced_hash_routes(config)
    elif config.routing_strategy is TRHashStrategy.MODULO_CYCLIC:
        routes = _modulo_cyclic_routes(config)
    elif config.routing_strategy is TRHashStrategy.AFFINE_HASH:
        routes = _affine_hash_routes(config)
    else:  # pragma: no cover - Enum validation makes this defensive.
        raise ValueError(f"unsupported TR-Hash strategy: {config.routing_strategy}")

    if routes.shape != (config.top_k, config.vocab_size):
        raise RuntimeError("route builder returned an invalid shape")
    if routes.min().item() < 0 or routes.max().item() >= config.num_experts:
        raise RuntimeError("route builder returned an invalid expert index")
    if config.top_k > 1:
        sorted_routes = routes.sort(dim=0).values
        if torch.any(sorted_routes[1:] == sorted_routes[:-1]):
            raise RuntimeError("route builder selected one expert twice")
    return routes


def compile_top2_pair_metadata(
    routes: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compile a top-2 route table into compact CUDA dispatch metadata.

    The current hash-native CUDA partition encodes one unordered expert-pair
    index in bits 0..2 and the primary/secondary orientation in bit 3. This
    covers all pairs for two, three, or four experts without storing two
    integer expert IDs per vocabulary item.
    """

    if routes.ndim != 2 or routes.size(0) != 2:
        raise ValueError("routes must be shaped [2, vocab_size]")
    if not 2 <= num_experts <= 4:
        raise ValueError("compact pair metadata supports two to four experts")
    if routes.numel() and (
        routes.min().item() < 0 or routes.max().item() >= num_experts
    ):
        raise ValueError("routes contain an invalid expert index")

    expert_pairs = torch.combinations(
        torch.arange(num_experts, dtype=torch.int32, device="cpu"),
        r=2,
    )
    if expert_pairs.size(0) > 8:
        raise ValueError("compact pair encoding supports at most eight pairs")

    routes_cpu = routes.detach().to(device="cpu", dtype=torch.long)
    unordered = routes_cpu.sort(dim=0).values
    if torch.any(unordered[0] == unordered[1]):
        raise ValueError("top-2 pair routes must select two distinct experts")
    pair_matches = (
        unordered[0].unsqueeze(0)
        == expert_pairs[:, 0].long().unsqueeze(1)
    ) & (
        unordered[1].unsqueeze(0)
        == expert_pairs[:, 1].long().unsqueeze(1)
    )
    if not torch.all(pair_matches.any(dim=0)):
        raise ValueError("routes contain an expert pair that cannot be encoded")

    pair_indices = pair_matches.to(torch.int64).argmax(dim=0)
    swap = routes_cpu[0].eq(unordered[1]).to(torch.int64)
    route_codes = (pair_indices | (swap << 3)).to(torch.uint8)
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
    pair_indices = codes.bitwise_and(0x7)
    if pair_indices.numel() and pair_indices.max().item() >= expert_pairs.size(0):
        raise ValueError("route_codes reference an unavailable expert pair")
    selected = expert_pairs[pair_indices.long()].long()
    swap = codes.bitwise_and(0x8).ne(0)
    return torch.stack(
        (
            torch.where(swap, selected[:, 1], selected[:, 0]),
            torch.where(swap, selected[:, 0], selected[:, 1]),
        ),
        dim=0,
    )
