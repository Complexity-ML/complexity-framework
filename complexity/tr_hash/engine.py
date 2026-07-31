"""General TR-Hash SwiGLU engine.

The universal PyTorch path is the correctness contract. CUDA CGGR and CUDA
Graph paths are optional execution strategies selected from the same config.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .buckets import GraphBucketPlanner
from .capabilities import BackendDecision, select_backend
from .config import GraphBucket, TRHashBackend, TRHashEngineConfig, TRHashPhase
from .routing import build_route_table

logger = logging.getLogger(__name__)

try:
    from complexity_cuda.triton_token_routed import (
        HAS_TRITON,
        cggr_grouped_gemm_autograd,
        sort_tokens_by_expert,
    )
except Exception:
    HAS_TRITON = False
    cggr_grouped_gemm_autograd = None
    sort_tokens_by_expert = None


def _silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def reference_topk_swiglu(
    flat_x: torch.Tensor,
    routes: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    activation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = _silu_mul,
    intermediate_transform: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> torch.Tensor:
    """Authoritative PyTorch definition shared with ``TokenRoutedMLP``."""

    if flat_x.ndim != 2:
        raise ValueError("flat_x must be [tokens, hidden]")
    if routes.ndim != 2 or routes.size(1) != flat_x.size(0):
        raise ValueError("routes must be [top_k, tokens]")
    if route_weights.shape != routes.shape:
        raise ValueError("route_weights must match routes")
    if gate_weights.shape != up_weights.shape:
        raise ValueError("gate and up expert weights must have the same shape")
    if gate_weights.ndim != 3 or down_weights.ndim != 3:
        raise ValueError("expert weights must be rank-3 tensors")
    if gate_weights.size(0) != down_weights.size(0):
        raise ValueError("expert weight tensors must have the same expert count")

    output = torch.zeros_like(flat_x)
    for expert_index in range(gate_weights.size(0)):
        token_weight = (routes.eq(expert_index).to(flat_x.dtype) * route_weights).sum(dim=0)
        active_x = flat_x * token_weight.ne(0).to(flat_x.dtype).unsqueeze(-1)
        intermediate = activation(
            active_x @ gate_weights[expert_index],
            active_x @ up_weights[expert_index],
        )
        if intermediate_transform is not None:
            intermediate = intermediate_transform(intermediate, expert_index)
        expert_output = intermediate @ down_weights[expert_index]
        output = output + expert_output.to(output.dtype) * token_weight.unsqueeze(-1)
    return output


class TRHashEngine(nn.Module):
    """Shared SwiGLU plus deterministic top-k residual experts.

    GQA and MHA produce the same ``[batch, sequence, hidden]`` contract, so the
    engine deliberately does not duplicate attention code.
    """

    def __init__(self, config: TRHashEngineConfig):
        super().__init__()
        self.config = config
        self.register_buffer("route_table", build_route_table(config))

        self.expert_gate = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                config.expert_width,
            )
        )
        self.expert_up = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                config.expert_width,
            )
        )
        self.expert_down = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.expert_width,
                config.hidden_size,
            )
        )
        if config.shared_width:
            self.shared_gate = nn.Linear(
                config.hidden_size,
                config.shared_width,
                bias=False,
            )
            self.shared_up = nn.Linear(
                config.hidden_size,
                config.shared_width,
                bias=False,
            )
            self.shared_down = nn.Linear(
                config.shared_width,
                config.hidden_size,
                bias=False,
            )
        else:
            self.shared_gate = None
            self.shared_up = None
            self.shared_down = None

        self.reset_parameters()
        self.last_backend: Optional[BackendDecision] = None
        self._graph_pool: Dict[GraphBucket, "_CapturedTRHashGraph"] = {}
        self._bucket_planner = (
            GraphBucketPlanner(config.graph_buckets) if config.graph_buckets else None
        )

    def reset_parameters(self) -> None:
        for parameter in (self.expert_gate, self.expert_up, self.expert_down):
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        for module in (self.shared_gate, self.shared_up, self.shared_down):
            if module is not None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _routes(self, token_ids: torch.Tensor) -> torch.Tensor:
        clamped = token_ids.clamp(0, self.config.vocab_size - 1)
        return self.route_table[:, clamped]

    def _shared(self, flat_x: torch.Tensor) -> torch.Tensor:
        if self.shared_gate is None:
            return torch.zeros_like(flat_x)
        return self.shared_down(_silu_mul(self.shared_gate(flat_x), self.shared_up(flat_x)))

    def _reference_routed(
        self,
        flat_x: torch.Tensor,
        routes: torch.Tensor,
    ) -> torch.Tensor:
        """Graph-friendly, autograd-correct fixed expert loop."""

        route_weights = torch.tensor(
            self.config.route_weights,
            dtype=flat_x.dtype,
            device=flat_x.device,
        )[:, None].expand_as(routes)
        return reference_topk_swiglu(
            flat_x,
            routes,
            self.expert_gate,
            self.expert_up,
            self.expert_down,
            route_weights,
        )

    def _cggr_routed(
        self,
        flat_x: torch.Tensor,
        routes: torch.Tensor,
    ) -> torch.Tensor:
        """One combined grouped dispatch for arbitrary supported top-k."""

        if not HAS_TRITON or cggr_grouped_gemm_autograd is None:
            raise RuntimeError("CGGR selected but Triton CGGR is unavailable")
        token_count, hidden_size = flat_x.shape
        expanded_x = (
            flat_x.unsqueeze(0)
            .expand(
                self.config.top_k,
                -1,
                -1,
            )
            .reshape(-1, hidden_size)
        )
        flat_routes = routes.reshape(-1)
        sorted_x, sorted_indices, expert_offsets, _ = sort_tokens_by_expert(
            expanded_x,
            flat_routes,
            self.config.num_experts,
        )
        gate = cggr_grouped_gemm_autograd(
            sorted_x,
            self.expert_gate,
            expert_offsets,
        )
        up = cggr_grouped_gemm_autograd(
            sorted_x,
            self.expert_up,
            expert_offsets,
        )
        intermediate = _silu_mul(gate, up)
        sorted_output = cggr_grouped_gemm_autograd(
            intermediate,
            self.expert_down,
            expert_offsets,
        )
        expanded_output = torch.empty_like(sorted_output)
        expanded_output[sorted_indices] = sorted_output
        expanded_output = expanded_output.view(
            self.config.top_k,
            token_count,
            hidden_size,
        )
        weights = torch.tensor(
            self.config.route_weights,
            device=flat_x.device,
            dtype=flat_x.dtype,
        ).view(self.config.top_k, 1, 1)
        return (expanded_output * weights).sum(dim=0)

    def _forward_eager(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        backend: TRHashBackend,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must be [batch, sequence, hidden]")
        if token_ids.shape != hidden_states.shape[:2]:
            raise ValueError("token_ids must match hidden batch/sequence dimensions")
        if hidden_states.size(-1) != self.config.hidden_size:
            raise ValueError(
                f"expected hidden size {self.config.hidden_size}, got {hidden_states.size(-1)}"
            )

        batch, sequence, hidden = hidden_states.shape
        flat_x = hidden_states.reshape(-1, hidden)
        routes = self._routes(token_ids).reshape(self.config.top_k, -1)
        shared = self._shared(flat_x)
        if backend is TRHashBackend.CGGR:
            routed = self._cggr_routed(flat_x, routes)
        else:
            routed = self._reference_routed(flat_x, routes)
        output = self.config.shared_output_scale * shared + self.config.routed_output_scale * routed
        return output.view(batch, sequence, hidden)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        decision = select_backend(
            self.config,
            device_type=hidden_states.device.type,
            cggr_available=bool(HAS_TRITON and cggr_grouped_gemm_autograd),
        )
        self.last_backend = decision
        if decision.selected is TRHashBackend.CUDA_GRAPH:
            return self._forward_cuda_graph(hidden_states, token_ids)
        return self._forward_eager(hidden_states, token_ids, decision.selected)

    def _forward_cuda_graph(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("CUDA Graph TR-Hash requires eval mode and torch.no_grad()")
        if hidden_states.device.type != "cuda":
            raise RuntimeError("CUDA Graph TR-Hash requires CUDA tensors")
        if self._bucket_planner is None:
            raise RuntimeError("CUDA Graph backend has no configured buckets")
        batch, sequence, _ = hidden_states.shape
        bucket = self._bucket_planner.select(batch, sequence)
        runner = self._graph_pool.get(bucket)
        if runner is None:
            runner = _CapturedTRHashGraph(self, bucket, hidden_states)
            self._graph_pool[bucket] = runner
        return runner.replay(hidden_states, token_ids)

    def capability_summary(self, device_type: str = "cpu") -> dict:
        """Return stable, serializable metadata for logs and run manifests."""

        decision = select_backend(
            self.config,
            device_type=device_type,
            cggr_available=bool(HAS_TRITON and cggr_grouped_gemm_autograd),
        )
        return {
            "experts": self.config.num_experts,
            "top_k": self.config.top_k,
            "shared_width": self.config.shared_width,
            "expert_width": self.config.expert_width,
            "stored_width": self.config.stored_mlp_width,
            "active_width": self.config.active_mlp_width,
            "active_width_fraction": self.config.active_width_fraction,
            "attention": self.config.attention_backbone.value,
            "routing": self.config.routing_strategy.value,
            "phase": self.config.phase.value,
            "precision": self.config.precision.value,
            "world_size": self.config.world_size,
            "parallelism": ("replicated_ddp" if self.config.world_size > 1 else "single_device"),
            "requested_backend": decision.requested.value,
            "selected_backend": decision.selected.value,
            "backend_reasons": list(decision.reasons),
            "cuda_graph_buckets": [
                [bucket.batch_size, bucket.sequence_length] for bucket in self.config.graph_buckets
            ],
        }


class _CapturedTRHashGraph:
    """One lazily captured, persistent-buffer inference graph."""

    def __init__(
        self,
        engine: TRHashEngine,
        bucket: GraphBucket,
        sample: torch.Tensor,
    ):
        if engine.config.phase is not TRHashPhase.INFERENCE:
            raise RuntimeError("only inference graphs can be captured")
        self.engine = engine
        self.bucket = bucket
        shape = (
            bucket.batch_size,
            bucket.sequence_length,
            engine.config.hidden_size,
        )
        self.hidden = torch.zeros(
            shape,
            dtype=sample.dtype,
            device=sample.device,
        )
        self.token_ids = torch.zeros(
            bucket.batch_size,
            bucket.sequence_length,
            dtype=torch.long,
            device=sample.device,
        )
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.output = engine._forward_eager(
                    self.hidden,
                    self.token_ids,
                    TRHashBackend.PYTORCH,
                )
        stream.synchronize()
        with torch.cuda.graph(self.graph):
            self.output = engine._forward_eager(
                self.hidden,
                self.token_ids,
                TRHashBackend.PYTORCH,
            )

    def replay(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        self.hidden.zero_()
        self.token_ids.zero_()
        self.hidden[:batch, :sequence].copy_(hidden_states)
        self.token_ids[:batch, :sequence].copy_(token_ids)
        self.graph.replay()
        return self.output[:batch, :sequence].clone()
