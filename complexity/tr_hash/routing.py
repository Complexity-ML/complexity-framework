"""Deterministic route-table construction for TR-Hash."""

from __future__ import annotations

from dataclasses import replace

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

    if config.num_experts == 1:
        # A single expert has no stride/tuple geometry to vary — every route
        # position trivially selects expert 0.
        return torch.zeros(
            config.top_k, config.vocab_size, dtype=torch.long, device="cpu"
        )

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


def _multi_hash_routes(config: TRHashEngineConfig) -> torch.Tensor:
    """Select distinct experts by multi-hash rendezvous voting.

    Each token/expert pair receives one independent 32-bit score per hash
    channel. Summing the channels makes the final top-k decision depend on
    several token-ID hashes instead of one hash plus a derived stride. The
    result is still compiled into the ordinary ``[top_k, vocab]`` table, so
    dispatch and fused CUDA execution incur no additional runtime routing
    cost.
    """

    token_ids = torch.arange(
        config.vocab_size,
        dtype=torch.int64,
        device="cpu",
    ).unsqueeze(1)
    expert_ids = torch.arange(
        config.num_experts,
        dtype=torch.int64,
        device="cpu",
    ).unsqueeze(0)
    scores = torch.zeros(
        config.vocab_size,
        config.num_experts,
        dtype=torch.int64,
        device="cpu",
    )
    layer_salt = (
        int(config.route_seed) + int(config.layer_index) * 0x9E3779B1
    ) & 0xFFFFFFFF
    for hash_index in range(config.route_hash_count):
        channel_salt = (
            layer_salt + hash_index * 0x85EBCA77
        ) & 0xFFFFFFFF
        expert_salts = _mix32(
            expert_ids + (channel_salt ^ 0xC2B2AE3D)
        )
        scores.add_(_mix32(token_ids ^ expert_salts ^ channel_salt))
    return scores.topk(config.top_k, dim=1, largest=True, sorted=True).indices.T.long()


def expand_route_table_hierarchically(
    routes: torch.Tensor,
    *,
    source_num_experts: int,
    target_num_experts: int,
    layer_index: int,
    route_seed: int = 0x71D5A17,
) -> torch.Tensor:
    """Split every source expert into deterministic, function-preserving clones.

    Expert ``e`` is expanded to ``e + clone * source_num_experts``. If every
    clone starts with an exact copy of its parent weights, replacing a source
    route with any clone route preserves the model function while allowing
    the clones to specialize independently during subsequent training.
    """

    if routes.ndim != 2:
        raise ValueError("routes must be shaped [top_k, vocab_size]")
    if source_num_experts < 1:
        raise ValueError("source_num_experts must be positive")
    if target_num_experts <= source_num_experts:
        raise ValueError("target_num_experts must exceed source_num_experts")
    if target_num_experts % source_num_experts:
        raise ValueError("target_num_experts must be a multiple of source_num_experts")
    if routes.numel() and (
        routes.min().item() < 0 or routes.max().item() >= source_num_experts
    ):
        raise ValueError("routes contain an invalid source expert index")

    clone_count = target_num_experts // source_num_experts
    token_ids = torch.arange(routes.size(1), dtype=torch.int64, device="cpu").unsqueeze(0)
    route_indices = torch.arange(routes.size(0), dtype=torch.int64, device="cpu").unsqueeze(1)
    parents = routes.detach().to(device="cpu", dtype=torch.int64)
    layer_salt = (int(route_seed) + int(layer_index) * 0x9E3779B1) & 0xFFFFFFFF
    clone_ids = _mix32(
        token_ids
        ^ (parents * 0x85EBCA77)
        ^ (route_indices * 0xC2B2AE3D)
        ^ layer_salt
    ).remainder(clone_count)
    return (parents + clone_ids * source_num_experts).long()


def _hierarchical_hash_routes(config: TRHashEngineConfig) -> torch.Tensor:
    """Build balanced parent routes and select a hash clone within each family."""

    if config.num_experts < 4 or config.num_experts % 2:
        raise ValueError("hierarchical hash requires an even expert count of at least four")
    parent_count = config.num_experts // 2
    if config.top_k > parent_count:
        raise ValueError("top_k cannot exceed the hierarchical parent expert count")
    parent_config = replace(
        config,
        num_experts=parent_count,
        routing_strategy=TRHashStrategy.BALANCED_HASH,
    )
    return expand_route_table_hierarchically(
        _balanced_hash_routes(parent_config),
        source_num_experts=parent_count,
        target_num_experts=config.num_experts,
        layer_index=config.layer_index,
        route_seed=config.route_seed,
    )


def build_route_table(config: TRHashEngineConfig) -> torch.Tensor:
    """Return a CPU ``[top_k, vocab_size]`` route table."""

    if config.routing_strategy is TRHashStrategy.BALANCED_HASH:
        routes = _balanced_hash_routes(config)
    elif config.routing_strategy is TRHashStrategy.MODULO_CYCLIC:
        routes = _modulo_cyclic_routes(config)
    elif config.routing_strategy is TRHashStrategy.AFFINE_HASH:
        routes = _affine_hash_routes(config)
    elif config.routing_strategy is TRHashStrategy.MULTI_HASH:
        routes = _multi_hash_routes(config)
    elif config.routing_strategy is TRHashStrategy.HIERARCHICAL_HASH:
        routes = _hierarchical_hash_routes(config)
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

    The hash-native CUDA partition uses the legacy 3-bit pair index plus one
    orientation bit for up to four experts. Eight experts require 28 pairs, so
    they use a 5-bit pair index plus one orientation bit. Both layouts fit in
    one byte and remain distinguishable from the pair-table length, preserving
    existing checkpoint metadata.
    """

    if routes.ndim != 2 or routes.size(0) != 2:
        raise ValueError("routes must be shaped [2, vocab_size]")
    if not 2 <= num_experts <= 8:
        raise ValueError("compact pair metadata supports two to eight experts")
    if routes.numel() and (
        routes.min().item() < 0 or routes.max().item() >= num_experts
    ):
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
    pair_mask, swap_mask = ((0x7, 0x8) if pair_count <= 8 else (0x1F, 0x20))
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
