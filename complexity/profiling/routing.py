"""Inspect causal TR-HASH routes and contributions on complete local sequences.

A probe records both preceding IDs at every original position. Contributions
and shared outputs are checked against the real forward before any sampling.
Statistics always cover the complete probe; no plotting or training is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from complexity.tr_hash.composed_routing import composed_mlp_routes


@dataclass(frozen=True)
class RoutingProbe:
    """Complete population in row-major sequence order, ready for inspection."""

    token_ids: np.ndarray  # [batch, sequence]; boundaries are retained.
    inputs: np.ndarray  # [positions, hidden]
    routes: np.ndarray  # [2, positions]
    contributions: np.ndarray  # [2, positions, hidden], including route weights and scale.
    shared: np.ndarray  # [positions, hidden], including the shared scale.
    actual_output: np.ndarray  # [positions, hidden]
    layer_index: int
    expert_width: int
    shared_width: int
    reconstruction_max_abs_error: float


def parse_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",")))
    except (ValueError, AttributeError) as error:
        raise ValueError("layer indices must be comma-separated integers") from error
    if not layers or min(layers) < 0:
        raise ValueError("at least one nonnegative layer index is required")
    return layers


def load_tr_hash_model(checkpoint: str | Path, device: torch.device):
    """Strictly load a local canonical checkpoint without routing conversion."""
    from complexity.core.mlp.tr_hash_engine import TRHashEngineMLP
    from complexity.models import ComplexityModel

    model = ComplexityModel.from_pretrained(checkpoint, device=str(device), dtype=torch.float32)
    if any(not isinstance(layer.mlp, TRHashEngineMLP) for layer in model.layers):
        raise ValueError("routing diagnostics require a native TR-HASH engine bundle")
    return model.eval().requires_grad_(False)


def _validate_probe(mlp, hidden, token_ids, device, previous_token_ids):
    if hidden.ndim != 3 or token_ids.ndim != 2 or hidden.shape[:2] != token_ids.shape:
        raise ValueError(
            "probe requires hidden [batch, sequence, hidden] and IDs [batch, sequence]"
        )
    engine = mlp.engine
    if not token_ids.numel() or hidden.shape[-1] != engine.config.hidden_size:
        raise ValueError("probe must contain tokens with the engine's hidden width")
    if token_ids.dtype != torch.long:
        raise ValueError("token IDs must be torch.long")
    if token_ids.min() < 0 or token_ids.max() >= engine.config.vocab_size:
        raise ValueError("token ID is outside the route vocabulary")
    if not torch.isfinite(hidden).all():
        raise ValueError("probe hidden states must be finite")
    if previous_token_ids is not None:
        if (
            previous_token_ids.shape != (token_ids.size(0), 2)
            or previous_token_ids.dtype != torch.long
            or previous_token_ids.device != token_ids.device
        ):
            raise ValueError("previous_token_ids must be same-device torch.long [batch, 2]")
        if previous_token_ids.min() < 0 or previous_token_ids.max() >= engine.config.vocab_size:
            raise ValueError("previous token ID is outside the route vocabulary")
    if engine.active_num_experts != 4 or engine.active_expert_width != engine.config.expert_width:
        raise ValueError("routing diagnostics require full configured expert capacity")
    device = torch.device(device)
    actual = engine.expert_gate.device
    if actual.type != device.type or (device.index is not None and actual.index != device.index):
        raise ValueError("move the TR-HASH engine to the requested probe device first")
    return device


def collect_routing_probe(
    mlp, hidden, token_ids, device, *, previous_token_ids=None, batch_size=512
) -> RoutingProbe:
    """Compute and verify every position before selecting any visualized points.

    Rows are independent original sequences. For a contiguous sequence chunk,
    previous_token_ids supplies its two true preceding IDs, oldest first.
    batch_size only partitions the pointwise projection work; it never resets
    routing history or joins the end of one row to the start of another.
    """
    device = _validate_probe(mlp, hidden, token_ids, device, previous_token_ids)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("probe batch size must be a positive integer")
    engine = mlp.engine
    inputs = hidden.to(device=device, dtype=engine.expert_gate.dtype)
    ids = token_ids.to(device=device)
    history = None if previous_token_ids is None else previous_token_ids.to(device=device)
    flat = inputs.reshape(-1, inputs.size(-1))
    with torch.inference_mode():
        routes = composed_mlp_routes(ids, engine.config.layer_index, history).reshape(2, -1)
        contributions = torch.zeros((2, *flat.shape), device=device, dtype=flat.dtype)
        shared = torch.empty_like(flat)
        for start in range(0, flat.size(0), batch_size):
            end = min(start + batch_size, flat.size(0))
            x = flat[start:end]
            for rank in range(2):
                for expert in range(4):
                    mask = routes[rank, start:end] == expert
                    selected = x[mask]
                    intermediate = F.silu(selected @ engine.expert_gate[expert]) * (
                        selected @ engine.expert_up[expert]
                    )
                    contributions[rank, start:end][mask] = (
                        intermediate @ engine.expert_down[expert]
                    ) * (engine.config.routed_output_scale * engine.config.route_weights[rank])
            shared[start:end] = (
                engine.shared_down(F.silu(engine.shared_gate(x)) * engine.shared_up(x))
                * engine.config.shared_output_scale
            )
        actual = mlp(inputs, ids, previous_token_ids=history).reshape_as(flat)
        reconstructed = shared + contributions.sum(dim=0)
        if not all(torch.isfinite(value).all() for value in (contributions, shared, actual)):
            raise ValueError("probe contributions and forward outputs must be finite")
        tolerance = {
            torch.float64: (1e-10, 1e-12),
            torch.float32: (1e-5, 1e-9),
            torch.float16: (5e-3, 1e-5),
            torch.bfloat16: (5e-2, 1e-4),
        }[flat.dtype]
        try:
            torch.testing.assert_close(reconstructed, actual, rtol=tolerance[0], atol=tolerance[1])
        except AssertionError as error:
            raise ValueError(
                "shared and routed contributions do not reconstruct the real forward"
            ) from error
        max_error = float((reconstructed - actual).abs().max())

    def numpy(value):
        return value.detach().float().cpu().numpy().copy()

    return RoutingProbe(
        token_ids=ids.cpu().numpy().copy(),
        inputs=numpy(flat),
        routes=routes.cpu().numpy().copy(),
        contributions=numpy(contributions),
        shared=numpy(shared),
        actual_output=numpy(actual),
        layer_index=engine.config.layer_index,
        expert_width=engine.config.expert_width,
        shared_width=engine.config.shared_width,
        reconstruction_max_abs_error=max_error,
    )


def stratified_token_sample(probe: RoutingProbe, count, rng) -> np.ndarray:
    """Select distinct original positions, covering every observed primary expert."""
    if not isinstance(probe, RoutingProbe):
        raise ValueError("collect a complete sequence probe before sampling")
    if type(count) is not int or count < 1:
        raise ValueError("sample count must be a positive integer")
    primary = probe.routes[0]
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


def routed_contributions(probe: RoutingProbe, sample_indices):
    """Return unit vectors, original norms and routes from the verified population."""
    if not isinstance(probe, RoutingProbe):
        raise ValueError("collect a complete sequence probe before sampling")
    indices = np.asarray(sample_indices)
    if indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu":
        raise ValueError("sample indices must be a nonempty integer vector")
    if indices.min() < 0 or indices.max() >= probe.inputs.shape[0]:
        raise ValueError("sample index is outside the probe")
    vectors = probe.contributions[:, indices].reshape(-1, probe.inputs.shape[-1])
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("contribution norms must be finite")
    normalized = np.divide(
        vectors, norms[:, None], out=np.zeros_like(vectors), where=norms[:, None] > 0
    )
    return (
        normalized,
        probe.routes[:, indices].reshape(-1).copy(),
        np.repeat(np.arange(2), len(indices)),
        np.tile(indices, 2),
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


def audit_full_probe_routing(probe: RoutingProbe, residual):
    """Measure full-population route frequencies, norms and residual/input ratios."""
    if not isinstance(probe, RoutingProbe):
        raise ValueError("collect a complete sequence probe before auditing")
    if isinstance(residual, torch.Tensor):
        residual = residual.detach().float().cpu().numpy()
    residual = np.asarray(residual)
    expected_shape = (*probe.token_ids.shape, probe.inputs.shape[-1])
    if residual.shape != expected_shape or not np.isfinite(residual).all():
        raise ValueError("residual states must match the original sequence shape and be finite")
    residual = residual.reshape(probe.inputs.shape).astype(np.float64)
    count = probe.inputs.shape[0]
    route_counts = np.stack([np.bincount(row, minlength=4) for row in probe.routes])
    pairs = np.zeros((4, 4), dtype=np.int64)
    np.add.at(pairs, (probe.routes[0], probe.routes[1]), 1)
    routed_norm = np.linalg.norm(probe.contributions.astype(np.float64).sum(axis=0), axis=1)
    residual_norm = np.linalg.norm(residual, axis=1)
    input_norm = np.linalg.norm(probe.inputs.astype(np.float64), axis=1)
    combined = route_counts.sum(axis=0)
    values = {
        "routed_branch_norm": routed_norm,
        "residual_stream_norm": residual_norm,
        "mlp_input_norm": input_norm,
        "routed_to_residual_norm_ratio": routed_norm / np.maximum(residual_norm, 1e-12),
        "routed_to_mlp_input_norm_ratio": routed_norm / np.maximum(input_norm, 1e-12),
    }
    return {
        "tokens": count,
        "sequence_shape": list(probe.token_ids.shape),
        "unique_token_ids": int(np.unique(probe.token_ids).size),
        "sampling": "all_collected_probe_tokens",
        "num_experts": 4,
        "expert_width": probe.expert_width,
        "shared_width": probe.shared_width,
        "route_counts": route_counts.tolist(),
        "route_fractions": (route_counts / count).tolist(),
        "combined_route_counts": combined.tolist(),
        "combined_route_fractions": (combined / (count * 2)).tolist(),
        "primary_secondary_pair_counts": pairs.tolist(),
        "tokens_with_repeated_expert_across_routes": int(
            (probe.routes[0] == probe.routes[1]).sum()
        ),
        "all_experts_observed_per_route": bool((route_counts > 0).all()),
        "reconstruction_max_abs_error": probe.reconstruction_max_abs_error,
        **{key: _distribution_summary(value) for key, value in values.items()},
    }
