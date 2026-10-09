"""Inspect actual TR-HASH routes and expert contributions on local probe tensors.

These diagnostics use the persisted routes and active expert capacity. Sampling
is for inspecting vectors; full-probe statistics always include every token.
No plotting dependency, network fetch, or training launch is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",")))
    except (ValueError, AttributeError) as error:
        raise ValueError("layer indices must be comma-separated integers") from error
    if not layers or min(layers) < 0:
        raise ValueError("at least one nonnegative layer index is required")
    return layers


def load_tr_hash_model(checkpoint: str | Path, device: torch.device):
    """Strictly load a local native bundle and preserve its learned routes."""
    from complexity.models import ComplexityModel

    model = ComplexityModel.from_pretrained(checkpoint, device=str(device), dtype=torch.float32)
    if any(not hasattr(layer.mlp, "engine") for layer in model.layers):
        raise ValueError("routing diagnostics require a native TR-HASH engine bundle")
    return model.eval().requires_grad_(False)


def stratified_token_sample(route_table, token_ids, count, rng) -> np.ndarray:
    """Select distinct positions with every observed primary expert represented."""
    if type(count) is not int or count < 1:
        raise ValueError("sample count must be a positive integer")
    if token_ids.ndim != 1 or not token_ids.numel():
        raise ValueError("sampling requires a nonempty vector of token IDs")
    if route_table.ndim != 2 or not route_table.shape[0]:
        raise ValueError("route table must have shape routes by vocabulary")
    ids = token_ids.detach().to(device="cpu", dtype=torch.long)
    if ids.min() < 0 or ids.max() >= route_table.shape[1]:
        raise ValueError("token ID is outside the route vocabulary")
    primary = route_table.detach().cpu()[0, ids].numpy()
    expert_ids = np.unique(primary)
    count = min(count, len(primary))
    if count < len(expert_ids):
        raise ValueError("sample count cannot represent every observed primary expert")
    per_expert = max(1, count // len(expert_ids))
    chosen = []
    for expert in expert_ids:
        candidates = np.flatnonzero(primary == expert)
        chosen.extend(rng.choice(candidates, min(per_expert, len(candidates)), replace=False))
    remaining = count - len(chosen)
    if remaining:
        pool = np.setdiff1d(np.arange(len(primary)), chosen)
        chosen.extend(rng.choice(pool, remaining, replace=False))
    return np.asarray(sorted(chosen), dtype=np.int64)


def _validate_probe(mlp, hidden, token_ids, device):
    if hidden.ndim != 2 or token_ids.ndim != 1 or len(hidden) != len(token_ids):
        raise ValueError("hidden states and token IDs must describe matching flat token positions")
    if not len(token_ids) or hidden.shape[-1] != mlp.engine.config.hidden_size:
        raise ValueError("probe must contain tokens with the engine's hidden width")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token IDs must be integer tensors")
    if token_ids.min() < 0 or token_ids.max() >= mlp.engine.config.vocab_size:
        raise ValueError("token ID is outside the route vocabulary")
    if not torch.isfinite(hidden).all():
        raise ValueError("probe hidden states must be finite")
    device = torch.device(device)
    actual = mlp.engine.expert_gate.device
    if actual.type != device.type or (device.index is not None and actual.index != device.index):
        raise ValueError("move the TR-HASH engine to the requested probe device first")
    return device


def _contributions(mlp, hidden, token_ids, device):
    engine = mlp.engine
    x = hidden.to(device=device, dtype=engine.expert_gate.dtype)
    routes = engine._routes(token_ids.to(device=device, dtype=torch.long))
    gate, up, down = engine._active_expert_weights()
    for rank in range(engine.config.top_k):
        experts = routes[rank]
        contribution = torch.empty_like(x)
        for expert in range(engine.active_num_experts):
            mask = experts == expert
            if not mask.any():
                continue
            selected = x[mask]
            intermediate = F.silu(selected @ gate[expert]) * (selected @ up[expert])
            contribution[mask] = intermediate @ down[expert]
        contribution *= engine.config.routed_output_scale * engine.config.route_weights[rank]
        yield contribution, experts


def routed_contributions(mlp, hidden, token_ids, sample_indices, device):
    """Return unit vectors, real norms, routes and original sampled positions."""
    device = _validate_probe(mlp, hidden, token_ids, device)
    indices = np.asarray(sample_indices)
    if indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu":
        raise ValueError("sample indices must be a nonempty integer vector")
    if indices.min() < 0 or indices.max() >= len(token_ids):
        raise ValueError("sample index is outside the probe")
    positions = torch.as_tensor(indices.copy(), dtype=torch.long, device=hidden.device)
    ids = token_ids[positions.to(token_ids.device)]
    outputs, experts = [], []
    with torch.inference_mode():
        for contribution, route in _contributions(mlp, hidden[positions], ids, device):
            outputs.append(contribution.float().cpu().numpy())
            experts.append(route.cpu().numpy())
    vectors = np.concatenate(outputs)
    norms = np.linalg.norm(vectors, axis=1)
    return (
        vectors / np.maximum(norms[:, None], 1e-12),
        np.concatenate(experts),
        np.repeat(np.arange(len(outputs)), len(indices)),
        np.tile(indices, len(outputs)),
    ), norms


def _distribution_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("probe distributions must be nonempty and finite")
    return {
        "min": float(values.min()),
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def audit_full_probe_routing(mlp, hidden, residual, token_ids, device, *, batch_size=512):
    """Measure all actual routes, routed branch norms and residual/input ratios."""
    device = _validate_probe(mlp, hidden, token_ids, device)
    if residual.shape != hidden.shape or not torch.isfinite(residual).all():
        raise ValueError("residual states must match hidden states and be finite")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("probe batch size must be a positive integer")
    engine = mlp.engine
    count, top_k, expert_count = len(token_ids), engine.config.top_k, engine.active_num_experts
    route_counts = np.zeros((top_k, expert_count), dtype=np.int64)
    pairs = np.zeros((expert_count, expert_count), dtype=np.int64)
    norms = {
        key: []
        for key in (
            "routed_branch_norm",
            "residual_stream_norm",
            "mlp_input_norm",
            "routed_to_residual_norm_ratio",
            "routed_to_mlp_input_norm_ratio",
        )
    }
    repeated = 0
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            x = hidden[start:end].to(device=device, dtype=engine.expert_gate.dtype)
            routed, routes = torch.zeros_like(x), []
            for rank, (contribution, experts) in enumerate(
                _contributions(mlp, x, token_ids[start:end], device)
            ):
                routed += contribution
                row = experts.cpu().numpy()
                routes.append(row)
                route_counts[rank] += np.bincount(row, minlength=expert_count)
            routes = np.stack(routes)
            repeated += sum(len(np.unique(routes[:, i])) < top_k for i in range(end - start))
            if top_k >= 2:
                np.add.at(pairs, (routes[0], routes[1]), 1)
            routed_norm = routed.float().norm(dim=-1)
            residual_norm = residual[start:end].to(device).float().norm(dim=-1)
            input_norm = x.float().norm(dim=-1)
            for key, values in zip(
                norms,
                (
                    routed_norm,
                    residual_norm,
                    input_norm,
                    routed_norm / residual_norm.clamp_min(1e-12),
                    routed_norm / input_norm.clamp_min(1e-12),
                ),
            ):
                norms[key].extend(values.cpu().tolist())
    combined = route_counts.sum(axis=0)
    return {
        "tokens": count,
        "unique_token_ids": int(token_ids.unique().numel()),
        "sampling": "all_collected_probe_tokens",
        "active_num_experts": expert_count,
        "active_expert_width": engine.active_expert_width,
        "route_counts": route_counts.tolist(),
        "route_fractions": (route_counts / count).tolist(),
        "combined_route_counts": combined.tolist(),
        "combined_route_fractions": (combined / (count * top_k)).tolist(),
        "primary_secondary_pair_counts": pairs.tolist() if top_k >= 2 else None,
        "tokens_with_repeated_expert_across_routes": int(repeated),
        "all_experts_observed_per_route": bool((route_counts > 0).all()),
        **{key: _distribution_summary(values) for key, values in norms.items()},
    }
