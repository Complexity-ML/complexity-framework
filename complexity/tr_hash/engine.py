"""Canonical contextual TR-HASH SwiGLU execution through PyTorch or CGGR."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .buckets import GraphBucketPlanner
from .capabilities import BackendDecision, select_backend, supports_fused_cuda
from .config import (
    TRHashBackend,
    TRHashEngineConfig,
)
from .composed_routing import composed_mlp_routes
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

HAS_FUSED_CUDA = False


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
    """Authoritative PyTorch definition of the routed expert branch."""

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
        self.register_buffer(
            "_route_weights",
            torch.tensor(config.route_weights, dtype=torch.float32),
            persistent=False,
        )
        route_codes, expert_pairs = self._compile_fused_metadata(self.route_table)
        self.register_buffer("fused_route_codes", route_codes)
        self.register_buffer("fused_expert_pairs", expert_pairs)

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
        self._graph_pool: dict = {}
        self._bucket_planner = (
            GraphBucketPlanner(config.graph_buckets) if config.graph_buckets else None
        )

        # The expert count is fixed. Width controls select a prefix of channels.
        self._active_num_experts = config.num_experts
        self._active_expert_width = config.expert_width
        self._active_route_table = self.route_table
        # LoRA freezes the shared base projections. When both SwiGLU inputs
        # are adapted, cache their concatenated base matrix once so training
        # executes one large GEMM instead of two independent base GEMMs.
        # This is intentionally a plain, non-persistent tensor: it is derived
        # from canonical weights and must not enter checkpoints or DDP buffer
        # broadcasts.
        self._fused_shared_gate_up_weight: Optional[torch.Tensor] = None

    def _apply(self, fn, recurse: bool = True):
        # ``_active_route_table`` is a plain attribute (sometimes just an
        # alias of the ``route_table`` buffer, sometimes an independently
        # built reduced-capacity table from ``set_active_capacity``), so the
        # base ``_apply`` -- which only walks parameters/buffers -- does not
        # move it. Without this it silently stays on the old device after
        # ``.to(device)``/``.cuda()``/etc., and every ``_routes`` lookup
        # against the (now-moved) route_ids raises a cross-device index error.
        was_aliased = self._active_route_table is self.route_table
        super()._apply(fn, recurse=recurse)
        self._active_route_table = self.route_table if was_aliased else fn(self._active_route_table)
        if self._fused_shared_gate_up_weight is not None:
            self._fused_shared_gate_up_weight = fn(self._fused_shared_gate_up_weight)
        return self

    @property
    def active_num_experts(self) -> int:
        return self._active_num_experts

    @property
    def active_expert_width(self) -> int:
        return self._active_expert_width

    def set_active_capacity(self, *, num_experts=None, expert_width=None) -> None:
        """Vary only the active width; the architecture always retains four experts."""
        if num_experts is not None and num_experts != 4:
            raise ValueError("TR-HASH requires four active experts")
        if expert_width is not None:
            if not 0 < expert_width <= self.config.expert_width:
                raise ValueError(f"active expert_width must be in (0, {self.config.expert_width}]")
            self._active_expert_width = expert_width
        self._graph_pool.clear()

    def _compile_fused_metadata(self, route_table):
        # Static pair metadata cannot represent contextual routing.
        return torch.empty(0, dtype=torch.uint8), torch.empty((0, 2), dtype=torch.int32)

    def load_route_table(self, route_table: torch.Tensor) -> None:
        """Accept only the exact canonical zero-history anchor."""
        expected = build_route_table(self.config)
        if (
            not isinstance(route_table, torch.Tensor)
            or route_table.dtype != torch.long
            or not torch.equal(route_table.cpu(), expected)
        ):
            raise ValueError("TR-HASH contextual routing cannot be replaced by a token-only table")
        self.route_table.copy_(route_table.to(device=self.route_table.device))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        anchor = state_dict.get(prefix + "route_table")
        expected = build_route_table(self.config)
        if anchor is not None and (
            not isinstance(anchor, torch.Tensor)
            or anchor.dtype != torch.long
            or not torch.equal(anchor.cpu(), expected)
        ):
            error_msgs.append(
                prefix + "route_table must match the canonical zero-history routing anchor"
            )
            return
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def reset_parameters(self) -> None:
        std = self.config.initializer_range
        for parameter in (self.expert_gate, self.expert_up, self.expert_down):
            nn.init.normal_(parameter, mean=0.0, std=std)
        for module in (self.shared_gate, self.shared_up, self.shared_down):
            if module is not None:
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def _routes(
        self,
        token_ids: torch.Tensor,
        previous_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return composed_mlp_routes(token_ids, self.config.layer_index, previous_token_ids)

    def _active_expert_weights(self):
        """Return (gate, up, down) views truncated to the active capacity."""

        num_experts = self._active_num_experts
        width = self._active_expert_width
        if num_experts == self.config.num_experts and width == self.config.expert_width:
            return self.expert_gate, self.expert_up, self.expert_down
        return (
            self.expert_gate[:num_experts, :, :width].contiguous(),
            self.expert_up[:num_experts, :, :width].contiguous(),
            self.expert_down[:num_experts, :width, :].contiguous(),
        )

    def _shared(self, flat_x: torch.Tensor) -> torch.Tensor:
        if self.shared_gate is None:
            return torch.zeros_like(flat_x)
        fused_weight = self._fused_shared_gate_up_weight
        gate_is_lora = getattr(self.shared_gate, "is_lora_adapter", False)
        up_is_lora = getattr(self.shared_up, "is_lora_adapter", False)
        if fused_weight is not None and gate_is_lora and up_is_lora:
            shared_width = int(self.shared_gate.base.out_features)
            base_gate, base_up = F.linear(flat_x, fused_weight).split(
                [shared_width, shared_width], dim=-1
            )
            gate = base_gate + self.shared_gate.lora_residual(flat_x)
            up = base_up + self.shared_up.lora_residual(flat_x)
        else:
            gate = self.shared_gate(flat_x)
            up = self.shared_up(flat_x)
        return self.shared_down(_silu_mul(gate, up))

    def refresh_fused_shared_lora_weight(self) -> bool:
        """Refresh the derived frozen gate/up matrix used by shared LoRA."""

        gate_is_lora = getattr(self.shared_gate, "is_lora_adapter", False)
        up_is_lora = getattr(self.shared_up, "is_lora_adapter", False)
        if not (gate_is_lora and up_is_lora):
            self._fused_shared_gate_up_weight = None
            return False
        self._fused_shared_gate_up_weight = torch.cat(
            [self.shared_gate.base.weight, self.shared_up.base.weight], dim=0
        ).detach()
        return True

    def _reference_routed(
        self,
        flat_x: torch.Tensor,
        routes: torch.Tensor,
    ) -> torch.Tensor:
        """Graph-friendly, autograd-correct fixed expert loop."""

        route_weights = self._route_weights.to(dtype=flat_x.dtype)[:, None].expand_as(routes)
        gate, up, down = self._active_expert_weights()
        return reference_topk_swiglu(
            flat_x,
            routes,
            gate,
            up,
            down,
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
        gate_w, up_w, down_w = self._active_expert_weights()
        sorted_x, sorted_indices, expert_offsets, _ = sort_tokens_by_expert(
            expanded_x,
            flat_routes,
            self._active_num_experts,
        )
        gate = cggr_grouped_gemm_autograd(
            sorted_x,
            gate_w,
            expert_offsets,
        )
        up = cggr_grouped_gemm_autograd(
            sorted_x,
            up_w,
            expert_offsets,
        )
        intermediate = _silu_mul(gate, up)
        sorted_output = cggr_grouped_gemm_autograd(
            intermediate,
            down_w,
            expert_offsets,
        )
        expanded_output = torch.empty_like(sorted_output)
        expanded_output[sorted_indices] = sorted_output
        expanded_output = expanded_output.view(
            self.config.top_k,
            token_count,
            hidden_size,
        )
        weights = self._route_weights.to(dtype=flat_x.dtype).view(self.config.top_k, 1, 1)
        return (expanded_output * weights).sum(dim=0)

    def _forward_eager(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        backend: TRHashBackend,
        previous_token_ids: torch.Tensor | None = None,
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
        is_reduced_capacity = (
            self._active_num_experts != self.config.num_experts
            or self._active_expert_width != self.config.expert_width
        )
        if is_reduced_capacity and backend is TRHashBackend.FUSED_CUDA:
            raise RuntimeError(
                "FUSED_CUDA requires the full allocated capacity: its "
                "precompiled pair metadata is tied to config.num_experts. "
                "Use TRHashBackend.PYTORCH or CGGR while active capacity is "
                f"reduced (active={self._active_num_experts}/{self.config.num_experts} "
                f"experts, {self._active_expert_width}/{self.config.expert_width} width)."
            )
        flat_x = hidden_states.reshape(-1, hidden)
        shared = self._shared(flat_x)
        routes = self._routes(token_ids, previous_token_ids).reshape(self.config.top_k, -1)
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
        previous_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        decision = select_backend(
            self.config,
            device_type=hidden_states.device.type,
            cggr_available=bool(HAS_TRITON and cggr_grouped_gemm_autograd),
            fused_cuda_available=HAS_FUSED_CUDA,
        )
        self.last_backend = decision
        return self._forward_eager(hidden_states, token_ids, decision.selected, previous_token_ids)

    def capability_summary(self, device_type: str = "cpu") -> dict:
        """Return stable, serializable metadata for logs and run manifests."""

        decision = select_backend(
            self.config,
            device_type=device_type,
            cggr_available=bool(HAS_TRITON and cggr_grouped_gemm_autograd),
            fused_cuda_available=HAS_FUSED_CUDA,
        )
        return {
            "experts": self.config.num_experts,
            "top_k": self.config.top_k,
            "shared_width": self.config.shared_width,
            "expert_width": self.config.expert_width,
            "stored_width": self.config.stored_mlp_width,
            "active_width": self.config.active_mlp_width,
            "active_width_fraction": self.config.active_width_fraction,
            # Runtime dynamic capacity (see set_active_capacity): a prefix of
            # the allocated experts/width actually used right now, distinct
            # from the top_k-driven sparsity fields above.
            "active_num_experts": self._active_num_experts,
            "active_expert_width": self._active_expert_width,
            "is_reduced_capacity": (
                self._active_num_experts != self.config.num_experts
                or self._active_expert_width != self.config.expert_width
            ),
            "attention": self.config.attention_backbone.value,
            "routing": self.config.routing_strategy.value,
            "routing_context_tokens": 2,
            "phase": self.config.phase.value,
            "precision": self.config.precision.value,
            "world_size": self.config.world_size,
            "parallelism": ("replicated_ddp" if self.config.world_size > 1 else "single_device"),
            "requested_backend": decision.requested.value,
            "selected_backend": decision.selected.value,
            "backend_reasons": list(decision.reasons),
            "fused_cuda_eligible": supports_fused_cuda(self.config),
            "cuda_graph_buckets": [
                [bucket.batch_size, bucket.sequence_length] for bucket in self.config.graph_buckets
            ],
        }
