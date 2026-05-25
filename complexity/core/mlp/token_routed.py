"""
Token-Routed MLP — Deterministic Mixture-of-Experts.

Innovation from Complexity-ML (2026):
  Each token is routed to exactly one expert based on its token ID.
  Routing is deterministic (no learned router, no load-balancing loss).

Features:
  - Zipf-balanced routing via greedy bin-packing on token frequencies
  - Shared Lexical Expert (dense SwiGLU all tokens pass through)
  - Sparse dispatch (loop over experts with masking)
  - Falls back to simple modulo routing without token frequencies

Usage:
    config = MLPConfig(hidden_size=512, intermediate_size=2048, num_experts=4)
    mlp = TokenRoutedMLP(config)
    out = mlp(hidden_states, token_ids=token_ids)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

from .base import MLPBase, MLPConfig
from .fused_activations import fused_silu_mul
from ..registry import register_mlp
from ...utils.device import supports_custom_triton

logger = logging.getLogger(__name__)

# Try to import CGGR acceleration
try:
    from complexity_cuda.triton_token_routed import (
        sort_tokens_by_expert,
        cggr_grouped_gemm_triton,
        cggr_grouped_gemm_autograd,
        grouped_gemm_pytorch,
        fused_swiglu_triton,
        HAS_TRITON,
    )
    HAS_CGGR = HAS_TRITON
except Exception:
    HAS_CGGR = False
    cggr_grouped_gemm_autograd = None

    def sort_tokens_by_expert(tokens, expert_ids, num_experts):
        """Pure-PyTorch fallback — stable sort + cumsum offsets.
        Used when complexity_cuda is not installed (Mac/CPU dev setups).
        """
        sorted_expert_ids, sorted_indices = torch.sort(expert_ids, stable=True)
        sorted_tokens = tokens[sorted_indices]
        expert_counts = torch.bincount(expert_ids, minlength=num_experts)
        torch._check(expert_counts.shape[0] == num_experts)
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=tokens.device)
        expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)
        return sorted_tokens, sorted_indices, expert_offsets, expert_counts


def _to_local(t: torch.Tensor) -> torch.Tensor:
    """Convert DTensor to local tensor (FSDP v2 compat)."""
    if hasattr(t, 'to_local'):
        return t.to_local()
    return t


def _normalize_cggr_policy(policy: object) -> str:
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


def cggr_dispatch_decision(
    *,
    cggr_policy: object,
    kernel_policy: object,
    is_cuda: bool,
    has_cggr: bool,
    has_autograd: bool,
    static_dispatch: bool = False,
) -> tuple[bool, list[str]]:
    """Return whether Token-Routed should use CGGR and why it may not."""
    mode = _normalize_cggr_policy(cggr_policy)
    reasons: list[str] = []

    if mode == "false":
        reasons.append("config.use_cggr=False")
    if static_dispatch:
        reasons.append("static_expert_capacity=True")
    if not supports_custom_triton(kernel_policy):
        reasons.append(f"supports_custom_triton(policy={kernel_policy!r})=False")
    if not has_cggr:
        reasons.append("HAS_CGGR=False (complexity_cuda triton import failed)")
    if not is_cuda:
        reasons.append("flat_x.is_cuda=False")
    if not has_autograd:
        reasons.append("cggr_grouped_gemm_autograd=None")

    return mode != "false" and not reasons, reasons


@register_mlp("token_routed")
@register_mlp("sort_split")
@register_mlp("sort_split_moe")
@register_mlp("deterministic_moe")
@register_mlp("complexity")
class TokenRoutedMLP(MLPBase):
    """
    Token-Routed MLP with Shared Lexical Expert.

    Routes tokens to experts deterministically (token_id -> expert_id)
    via Zipf-balanced bin-packing, then dispatches with sparse masking.
    """

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        self.num_experts = config.num_experts
        self.vocab_size = config.vocab_size
        self.expert_intermediate_size = self.intermediate_size // self.num_experts
        # Top-K deterministic: each token activates K precomputed Zipf-balanced
        # expert maps. K=1 is the classic single-expert Zipf routing. K>1 gives
        # more gradient coverage while keeping zero learned routing and zero
        # runtime router overhead.
        self.top_k = max(1, int(getattr(config, "top_k", 1)))
        primary_weight = getattr(config, "top_k_primary_weight", None)
        if primary_weight is None:
            primary_weight = 0.95
        self._primary_weight = (
            min(1.0, max(0.0, float(primary_weight))) if self.top_k > 1 else 1.0
        )

        # Routed expert weights: gate, up, down.
        # down_proj_w will be re-initialized with GPT-2 residual scaling by
        # ComplexityModel._init_residual_scaling() after the module tree is built.
        self.gate_proj_w = nn.Parameter(torch.empty(
            self.num_experts, self.hidden_size, self.expert_intermediate_size
        ))
        self.up_proj_w = nn.Parameter(torch.empty(
            self.num_experts, self.hidden_size, self.expert_intermediate_size
        ))
        self.down_proj_w = nn.Parameter(torch.empty(
            self.num_experts, self.expert_intermediate_size, self.hidden_size
        ))
        for expert_idx in range(self.num_experts):
            nn.init.kaiming_uniform_(self.gate_proj_w[expert_idx], a=5**0.5)
            nn.init.kaiming_uniform_(self.up_proj_w[expert_idx], a=5**0.5)
            nn.init.kaiming_uniform_(self.down_proj_w[expert_idx], a=5**0.5)

        # Shared lexical expert: dense SwiGLU all tokens pass through.
        # Default size = intermediate_size (full dense width). shared_down is
        # also rescaled by _init_residual_scaling() (residual output projection).
        self.use_shared_expert = getattr(config, 'shared_expert', False)
        self.use_shared_routed_gates = bool(getattr(config, "use_shared_routed_gates", False))
        if self.use_shared_expert:
            shared_size = getattr(config, 'shared_intermediate_size', None) or self.intermediate_size
            self.shared_gate = nn.Linear(self.hidden_size, shared_size, bias=False)
            self.shared_up = nn.Linear(self.hidden_size, shared_size, bias=False)
            self.shared_down = nn.Linear(shared_size, self.hidden_size, bias=False)
            if self.use_shared_routed_gates:
                self.shared_output_gate = nn.Parameter(
                    torch.tensor(float(getattr(config, "shared_gate_init", 1.0)))
                )
                self.routed_output_gate = nn.Parameter(
                    torch.tensor(float(getattr(config, "routed_gate_init", 1.0)))
                )

        # Token -> expert mapping (Zipf-balanced or modulo). In meta-init
        # contexts these buffers do not affect parameter counts, so avoid
        # materializing o200k routing tables on CPU.
        if self.gate_proj_w.is_meta:
            token_to_expert = torch.empty(self.vocab_size, dtype=torch.long, device="meta")
            topk_token_to_expert = torch.empty(
                self.top_k, self.vocab_size, dtype=torch.long, device="meta"
            )
        else:
            token_to_expert = self._create_token_mapping(self.vocab_size, self.num_experts)
            topk_token_to_expert = self._create_topk_token_mapping(
                token_to_expert,
                self.vocab_size,
                self.num_experts,
                self.top_k,
            )
        self.register_buffer(
            "token_to_expert",
            token_to_expert,
        )
        self.register_buffer(
            "topk_token_to_expert",
            topk_token_to_expert,
        )

        # Optional context-signature routing overlay: (sig, cur_id) -> expert,
        # where sig is a polynomial hash of K previous token classes. Wider
        # context than bigram while still a pure function of token ids.
        cfg_ctx_keys = getattr(config, "ctx_sig_keys", None)
        cfg_ctx_experts = getattr(config, "ctx_sig_experts", None)
        cfg_ctx_classes = getattr(config, "token_class_table", None)
        cfg_ctx_window = int(getattr(config, "ctx_window", 0))
        cfg_ctx_buckets = int(getattr(config, "ctx_num_buckets", 0))
        if (
            cfg_ctx_keys is not None
            and cfg_ctx_experts is not None
            and cfg_ctx_classes is not None
            and cfg_ctx_window > 0
            and cfg_ctx_buckets > 0
            and not self.gate_proj_w.is_meta
        ):
            layer_idx = int(getattr(self.config, "layer_idx", 0))
            g = torch.Generator().manual_seed(0xC0DE + layer_idx)
            permutation = torch.randperm(self.num_experts, generator=g, device="cpu")
            permuted_ctx_experts = permutation[cfg_ctx_experts.detach().cpu().long()]
            self.register_buffer(
                "ctx_sig_keys", cfg_ctx_keys.detach().cpu().long(), persistent=False
            )
            self.register_buffer(
                "ctx_sig_experts", permuted_ctx_experts, persistent=False
            )
            self.register_buffer(
                "token_class_table",
                cfg_ctx_classes.detach().cpu().long(),
                persistent=False,
            )
            self.ctx_window = cfg_ctx_window
            self.ctx_num_buckets = cfg_ctx_buckets
            self.register_buffer(
                "ctx_class_weights",
                torch.tensor(
                    [8**i for i in range(cfg_ctx_window)], dtype=torch.long
                ),
                persistent=False,
            )
            self.has_ctx_sig_routing = True
        else:
            self.has_ctx_sig_routing = False

        # Expert utilization counters — accumulated across forward calls.
        # Reset manually via reset_expert_counts() (e.g. once per CSV log step).
        # Non-persistent so checkpoint save/load doesn't snapshot stale counters.
        self.register_buffer(
            "expert_counts",
            torch.zeros(self.num_experts, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("last_shared_rms", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("last_routed_rms", torch.tensor(float("nan")), persistent=False)

    def reset_expert_counts(self) -> None:
        """Zero the expert utilization counter. Call once per log interval."""
        self.expert_counts.zero_()

    def get_expert_counts(self) -> torch.Tensor:
        """Return current expert counts [num_experts] on-device."""
        return self.expert_counts

    def set_top_k_primary_weight(self, weight: float) -> None:
        """Update the primary/auxiliary blend for scheduled specialization."""
        if self.top_k <= 1:
            self._primary_weight = 1.0
            return
        self._primary_weight = min(1.0, max(0.0, float(weight)))

    def _create_token_mapping(self, vocab_size: int, num_experts: int) -> torch.Tensor:
        """
        Create deterministic mapping from token ID to expert ID.

        With token_frequencies: greedy bin-packing so each expert gets
        equal corpus frequency load (Zipf-balanced).
        Without: simple modulo fallback (token_id % E).

        When config.per_layer_routing is True, a deterministic permutation
        of the expert indices specific to this layer is applied after the
        mapping is built. This keeps Zipf balance intact (a permutation
        preserves load distribution) while giving each layer a different
        token→expert assignment, enriching specialization.
        """
        freqs = getattr(self.config, 'token_frequencies', None)
        classes = getattr(self.config, 'token_classes', None)
        strategy = getattr(self.config, 'routing_strategy', 'zipf')
        if freqs is not None or strategy == "zipf_token_class":
            if freqs is None:
                freqs = torch.ones(vocab_size, dtype=torch.float32, device="cpu")
            if strategy == "zipf_token_class" and classes is not None:
                mapping = self._create_token_class_mapping(freqs, classes, vocab_size, num_experts)
            else:
                freqs = freqs.detach().cpu().float()
                sorted_indices = freqs.argsort(descending=True)
                mapping = torch.empty(vocab_size, dtype=torch.long, device="cpu")
                expert_loads = [0.0] * num_experts
                for rank_pos in range(vocab_size):
                    token_id = sorted_indices[rank_pos].item()
                    e = min(range(num_experts), key=lambda i: expert_loads[i])
                    mapping[token_id] = e
                    expert_loads[e] += freqs[token_id].item()
        else:
            mapping = torch.arange(vocab_size, dtype=torch.long, device="cpu") % num_experts

        # Per-layer routing is always on: a deterministic layer-dependent
        # permutation of expert indices. Preserves Zipf load balance (a
        # permutation is measure-preserving) while forcing each layer to
        # route differently → richer specialization, zero runtime cost.
        layer_idx = int(getattr(self.config, 'layer_idx', 0))
        g = torch.Generator().manual_seed(0xC0DE + layer_idx)
        permutation = torch.randperm(num_experts, generator=g, device="cpu")
        mapping = permutation[mapping]
        return mapping

    def _routing_frequencies(self, vocab_size: int) -> torch.Tensor:
        freqs = getattr(self.config, "token_frequencies", None)
        if freqs is None:
            return torch.ones(vocab_size, dtype=torch.float32, device="cpu")
        if freqs.numel() != vocab_size:
            raise ValueError(
                f"token_frequencies length ({freqs.numel()}) must match vocab_size ({vocab_size})"
            )
        return freqs.detach().cpu().float().clamp_min(0.0)

    def _create_topk_token_mapping(
        self,
        primary_mapping: torch.Tensor,
        vocab_size: int,
        num_experts: int,
        top_k: int,
    ) -> torch.Tensor:
        """Build deterministic Zipf-balanced auxiliary expert maps.

        For k>0 each token is assigned to an expert different from all earlier
        routes for that token. The assignment greedily balances corpus frequency
        load per auxiliary route, so secondary experts are not just a fixed
        cyclic neighbor of the primary expert.
        """

        routes = torch.empty(top_k, vocab_size, dtype=torch.long, device="cpu")
        routes[0] = primary_mapping.detach().to(device="cpu", dtype=torch.long)
        if top_k == 1:
            return routes

        freqs = self._routing_frequencies(vocab_size)
        sorted_indices = freqs.argsort(descending=True)

        for route_idx in range(1, top_k):
            expert_loads = [0.0] * num_experts
            mapping = torch.empty(vocab_size, dtype=torch.long, device="cpu")
            for rank_pos in range(vocab_size):
                token_id = int(sorted_indices[rank_pos].item())
                blocked = {
                    int(routes[prev_idx, token_id].item())
                    for prev_idx in range(route_idx)
                }
                candidates = [idx for idx in range(num_experts) if idx not in blocked]
                expert = min(candidates, key=lambda idx: expert_loads[idx])
                mapping[token_id] = expert
                expert_loads[expert] += float(freqs[token_id].item())
            routes[route_idx] = mapping
        return routes

    def _create_token_class_mapping(
        self,
        freqs: torch.Tensor,
        classes: torch.Tensor,
        vocab_size: int,
        num_experts: int,
    ) -> torch.Tensor:
        """Greedy routing that balances total Zipf load and coarse token classes."""

        freqs = freqs.detach().cpu().float()
        classes = classes.detach().cpu().long()
        sorted_indices = freqs.argsort(descending=True)
        mapping = torch.empty(vocab_size, dtype=torch.long, device="cpu")
        expert_loads = [0.0] * num_experts
        class_ids = sorted(set(int(x) for x in classes.tolist()))
        class_loads = {
            class_id: [0.0] * num_experts
            for class_id in class_ids
        }

        for rank_pos in range(vocab_size):
            token_id = int(sorted_indices[rank_pos].item())
            cls = int(classes[token_id].item())
            weight = float(freqs[token_id].item())
            e = min(
                range(num_experts),
                key=lambda idx: (class_loads[cls][idx], expert_loads[idx]),
            )
            mapping[token_id] = e
            expert_loads[e] += weight
            class_loads[cls][e] += weight
        return mapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with sparse dispatch.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            token_ids: [batch, seq_len] — original input token IDs

        Returns:
            output: [batch, seq_len, hidden_size]
                    = SharedMLP(x) + Expert_e(x)
        """
        B, S, H = hidden_states.shape

        if token_ids is None:
            return self._forward_all_experts(hidden_states)

        # Look up expert assignment per token
        token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
        expert_ids = self.token_to_expert[token_ids_clamped]  # [B, S]
        route_expert_ids = self.topk_token_to_expert[:, token_ids_clamped]  # [K, B, S]

        # Context-signature overlay. Routing key = (sig, cur_id), where sig is
        # a polynomial hash of K previous token classes. Applied as a soft
        # bias on the secondary top-k route (top_k >= 2 required): the unigram
        # primary stays intact so the unigram bin-pack load balance is
        # preserved, and the contextual expert receives the (1 - primary_w)
        # share of the gradient signal.
        if getattr(self, "has_ctx_sig_routing", False) and self.top_k >= 2:
            B_, S_ = token_ids_clamped.shape
            K = self.ctx_window
            class_table = self.token_class_table.to(token_ids_clamped.device)
            class_weights = self.ctx_class_weights.to(token_ids_clamped.device)
            class_seq = class_table[token_ids_clamped]
            padded = torch.cat(
                [
                    torch.zeros(
                        (B_, K),
                        dtype=class_seq.dtype,
                        device=class_seq.device,
                    ),
                    class_seq,
                ],
                dim=1,
            )  # padded[:, K+s] == class_seq[:, s]
            windows = padded.unfold(dimension=1, size=K, step=1)[:, :S_, :]
            sig = (windows.long() * class_weights).sum(-1) % self.ctx_num_buckets
            keys = sig.long() * self.vocab_size + token_ids_clamped.long()
            flat_keys = keys.view(-1)
            ctx_keys_dev = self.ctx_sig_keys.to(flat_keys.device)
            ctx_experts_dev = self.ctx_sig_experts.to(flat_keys.device)
            idx = torch.searchsorted(ctx_keys_dev, flat_keys)
            idx_safe = idx.clamp(max=ctx_keys_dev.numel() - 1)
            found = ctx_keys_dev[idx_safe] == flat_keys
            ctx_exp = ctx_experts_dev[idx_safe].to(expert_ids.dtype)
            flat_primary = expert_ids.view(-1)
            flat_secondary = route_expert_ids[1].view(-1)
            distinct_from_primary = ctx_exp != flat_primary
            new_secondary = torch.where(
                found & distinct_from_primary, ctx_exp, flat_secondary
            ).view(B_, S_)
            route_expert_ids = torch.cat(
                [
                    route_expert_ids[0:1],
                    new_secondary.unsqueeze(0),
                    route_expert_ids[2:],
                ],
                dim=0,
            )

        flat_x = hidden_states.view(-1, H)
        flat_expert_ids = expert_ids.view(-1)
        static_dispatch = bool(getattr(self.config, "static_expert_capacity", False))
        collect_telemetry = bool(getattr(self.config, "collect_moe_telemetry", False))

        # Track expert utilization (in-place, non-differentiable)
        if collect_telemetry and not static_dispatch:
            with torch.no_grad():
                batch_counts = torch.bincount(route_expert_ids.reshape(-1), minlength=self.num_experts)
                self.expert_counts += batch_counts.to(self.expert_counts.dtype)

        # Shared expert (dense, all tokens) — fused SwiGLU via Liger on CUDA
        if self.use_shared_expert:
            shared_out = self._shared_expert_forward(flat_x)
        else:
            shared_out = 0

        # Routed experts — CGGR Triton (autograd-aware) or sparse-loop fallback.
        # CGGRGroupedGEMM (in complexity_cuda.triton_token_routed) wraps the
        # forward-only Triton kernel with a proper torch.autograd.Function so
        # gradients flow back to gate/up/down_proj_w. fused_swiglu_triton stays
        # forward-only so we use plain F.silu(gate) * up which PyTorch
        # differentiates natively.
        gate_w = _to_local(self.gate_proj_w)
        up_w = _to_local(self.up_proj_w)
        down_w = _to_local(self.down_proj_w)

        # Verify expert weights are fully gathered before grouped matmuls.
        if not hasattr(self, "_fsdp_checked"):
            expected = (self.num_experts, self.hidden_size, self.expert_intermediate_size)
            if gate_w.shape != expected:
                raise RuntimeError(
                    f"Invalid gate_proj_w shape {tuple(gate_w.shape)}; expected {expected}. "
                    "Expert weights must be fully gathered before TokenRoutedMLP forward."
                )
            else:
                logger.debug(f"TokenRoutedMLP expert weights shape {tuple(gate_w.shape)}")
            self._fsdp_checked = True

        # Routing path selection:
        #   - masked_dense (default, universal): fixed expert loop, no CPU
        #     synchronization, autograd-friendly. It spends extra routed FLOPs
        #     to avoid stalling the device on per-batch bucket sizes.
        #   - CGGR Triton (CUDA + auto/opt-in): custom grouped-GEMM kernel,
        #     selected through config.use_cggr when custom kernels are allowed.
        kernel_policy = getattr(self.config, "use_custom_kernels", "auto")
        cggr_policy = getattr(self.config, "use_cggr", "auto")
        use_cggr, why_not_cggr = cggr_dispatch_decision(
            cggr_policy=cggr_policy,
            kernel_policy=kernel_policy,
            is_cuda=flat_x.is_cuda,
            has_cggr=HAS_CGGR,
            has_autograd=cggr_grouped_gemm_autograd is not None,
            static_dispatch=static_dispatch,
        )
        self.last_dispatch_path = "cggr" if use_cggr else "masked_dense"

        # Log path selection once per path/device/policy combo. Helps diagnose
        # whether a run is on CGGR or the universal no-sync fallback.
        log_key = (self.last_dispatch_path, str(cggr_policy), str(kernel_policy), flat_x.device.type)
        logged_keys = getattr(self.__class__, "_path_logged_keys", set())
        if log_key not in logged_keys:
            logged_keys.add(log_key)
            self.__class__._path_logged_keys = logged_keys
            if use_cggr:
                logger.info("[TokenRoutedMLP] dispatch path = CGGR (Triton grouped-GEMM, no sync)")
            else:
                logger.info(
                    "[TokenRoutedMLP] dispatch path = masked_dense (no CPU sync) "
                    f"(CGGR rejected: {', '.join(why_not_cggr) if why_not_cggr else 'unknown'})"
                )

        # Top-K deterministic Zipf: dispatch K precomputed expert maps.
        # Primary keeps the configured weight; auxiliaries share the remainder.
        if static_dispatch:
            if self.top_k == 1:
                routed_out = self._dispatch_once(
                    flat_x, flat_expert_ids, gate_w, up_w, down_w, use_cggr, H,
                )
            else:
                secondary_w = (1.0 - self._primary_weight) / (self.top_k - 1)
                routed_out = torch.zeros_like(flat_x)
                for k in range(self.top_k):
                    w = self._primary_weight if k == 0 else secondary_w
                    expert_ids_k = route_expert_ids[k].view(-1)
                    part = self._dispatch_once(
                        flat_x, expert_ids_k, gate_w, up_w, down_w, use_cggr, H,
                    )
                    routed_out = routed_out + w * part
        else:
            sorted_x, sorted_idx, expert_offsets, expert_counts = sort_tokens_by_expert(
                flat_x, flat_expert_ids, self.num_experts
            )
            if self.top_k == 1:
                routed_out = self._dispatch_sorted(
                    flat_x, sorted_x, sorted_idx, expert_offsets, expert_counts,
                    gate_w, up_w, down_w, use_cggr, H,
                )
            else:
                secondary_w = (1.0 - self._primary_weight) / (self.top_k - 1)
                routed_out = torch.zeros_like(flat_x)
                for k in range(self.top_k):
                    w = self._primary_weight if k == 0 else secondary_w
                    if k == 0:
                        sorted_k = (sorted_x, sorted_idx, expert_offsets, expert_counts)
                    else:
                        sorted_k = sort_tokens_by_expert(
                            flat_x,
                            route_expert_ids[k].view(-1),
                            self.num_experts,
                        )
                    sorted_x_k, sorted_idx_k, expert_offsets_k, expert_counts_k = sorted_k
                    part = self._dispatch_sorted(
                        flat_x, sorted_x_k, sorted_idx_k, expert_offsets_k, expert_counts_k,
                        gate_w, up_w, down_w, use_cggr, H,
                    )
                    routed_out = routed_out + w * part

        if collect_telemetry and not static_dispatch:
            # Per-layer RMS diagnostics. Each .pow(2).mean().sqrt() is a small
            # GPU op + writes to a buffer that later needs to be read on CPU
            # for logging, so this adds non-trivial overhead in the
            # routing-heavy training loop. Off by default; enable explicitly
            # for ablation studies.
            with torch.no_grad():
                if isinstance(shared_out, torch.Tensor):
                    self.last_shared_rms.copy_(shared_out.detach().float().pow(2).mean().sqrt())
                else:
                    self.last_shared_rms.fill_(float("nan"))
                self.last_routed_rms.copy_(routed_out.detach().float().pow(2).mean().sqrt())

        if self.use_shared_expert and self.use_shared_routed_gates:
            out = self.shared_output_gate * shared_out + self.routed_output_gate * routed_out
        else:
            out = shared_out + routed_out
        return out.view(B, S, H)

    def _shared_expert_forward(self, flat_x: torch.Tensor) -> torch.Tensor:
        """Run the dense shared expert, optionally chunked over tokens.

        The shared expert is dense over every token, so large train batches can
        create very large gate/up activations. Chunking keeps the exact same
        math while lowering the peak live activation size without model-wide
        gradient checkpointing.
        """

        chunk_tokens = int(getattr(self.config, "shared_expert_chunk_tokens", 0) or 0)
        if chunk_tokens <= 0 or flat_x.size(0) <= chunk_tokens:
            return self.shared_down(
                fused_silu_mul(self.shared_gate(flat_x), self.shared_up(flat_x))
            ).to(flat_x.dtype)

        parts = []
        for start in range(0, flat_x.size(0), chunk_tokens):
            x_chunk = flat_x[start:start + chunk_tokens]
            parts.append(
                self.shared_down(
                    fused_silu_mul(self.shared_gate(x_chunk), self.shared_up(x_chunk))
                ).to(flat_x.dtype)
            )
        return torch.cat(parts, dim=0)

    def _dispatch_once(
        self,
        flat_x: torch.Tensor,
        expert_ids: torch.Tensor,
        gate_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
        use_cggr: bool,
        H: int,
    ) -> torch.Tensor:
        """Run one expert-dispatch pass for a given [N] expert assignment.

        Returns an [N, H] tensor in the same token order as flat_x.
        """
        if bool(getattr(self.config, "static_expert_capacity", False)):
            return self._dispatch_static_all_experts(flat_x, expert_ids, gate_w, up_w, down_w)

        sorted_x, sorted_idx, expert_offsets, expert_counts = sort_tokens_by_expert(
            flat_x, expert_ids, self.num_experts
        )

        return self._dispatch_sorted(
            flat_x, sorted_x, sorted_idx, expert_offsets, expert_counts,
            gate_w, up_w, down_w, use_cggr, H,
        )

    def _dispatch_sorted(
        self,
        flat_x: torch.Tensor,
        sorted_x: torch.Tensor,
        sorted_idx: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_counts: torch.Tensor,
        gate_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
        use_cggr: bool,
        H: int,
    ) -> torch.Tensor:
        """Run dispatch from a pre-sorted token layout."""
        if use_cggr:
            gate_out = cggr_grouped_gemm_autograd(sorted_x, gate_w, expert_offsets)
            up_out = cggr_grouped_gemm_autograd(sorted_x, up_w, expert_offsets)
            intermediate = fused_silu_mul(gate_out, up_out)
            sorted_routed = cggr_grouped_gemm_autograd(intermediate, down_w, expert_offsets)

            out = torch.empty_like(flat_x)
            out[sorted_idx] = sorted_routed.to(out.dtype)
            return out

        return self._dispatch_masked_dense(
            flat_x, sorted_x, sorted_idx, expert_counts, gate_w, up_w, down_w,
        )

    def _dispatch_masked_dense(
        self,
        flat_x: torch.Tensor,
        sorted_x: torch.Tensor,
        sorted_idx: torch.Tensor,
        expert_counts: torch.Tensor,
        gate_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
    ) -> torch.Tensor:
        """Universal no-sync fallback for Token-Routed dispatch.

        The old padded-bmm fallback needed ``expert_counts.cpu().tolist()`` to
        allocate per-batch buckets, creating a device/host sync every dispatch.
        This path keeps all control data on-device. It computes each expert over
        the full sorted token block and masks inactive rows, trading extra routed
        FLOPs for stable step times and no CPU synchronization.
        """

        sorted_expert_ids = torch.repeat_interleave(
            torch.arange(self.num_experts, device=sorted_x.device),
            expert_counts.to(device=sorted_x.device),
            output_size=sorted_x.size(0),
        )
        sorted_routed = torch.zeros_like(flat_x)
        for e in range(self.num_experts):
            mask = (sorted_expert_ids == e).unsqueeze(-1).to(sorted_x.dtype)
            x_e = sorted_x * mask
            gate = x_e @ gate_w[e]
            up = x_e @ up_w[e]
            expert_out = fused_silu_mul(gate, up) @ down_w[e]
            sorted_routed = sorted_routed + expert_out.to(sorted_routed.dtype) * mask

        out = torch.empty_like(flat_x)
        out[sorted_idx] = sorted_routed.to(out.dtype)
        return out

    def _dispatch_static_all_experts(
        self,
        flat_x: torch.Tensor,
        expert_ids: torch.Tensor,
        gate_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
    ) -> torch.Tensor:
        """Export-friendly Token-Routed dispatch.

        This path keeps deterministic token routing but computes every expert
        over the local token block and masks the selected expert output. It is
        intended for torch.export / pipeline tracing, where the sparse path's
        data-dependent bucket sizes cannot be represented robustly.
        """
        expert_ids = expert_ids.clamp(0, self.num_experts - 1)
        expanded_x = flat_x.unsqueeze(0).expand(self.num_experts, -1, -1)

        gate = torch.bmm(expanded_x, gate_w)
        up = torch.bmm(expanded_x, up_w)
        inter = fused_silu_mul(gate, up)
        expert_out = torch.bmm(inter, down_w)

        expert_mask = F.one_hot(expert_ids, num_classes=self.num_experts)
        expert_mask = expert_mask.transpose(0, 1).unsqueeze(-1).to(expert_out.dtype)
        return (expert_out * expert_mask).sum(dim=0).to(flat_x.dtype)

    def _forward_all_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fallback: average all experts (inference without token_ids)."""
        flat = hidden_states.view(-1, self.hidden_size)
        gate_w = _to_local(self.gate_proj_w)
        up_w = _to_local(self.up_proj_w)
        down_w = _to_local(self.down_proj_w)
        out = torch.zeros_like(flat)
        for e in range(self.num_experts):
            gate_e = flat @ gate_w[e]
            up_e = flat @ up_w[e]
            out = out + fused_silu_mul(gate_e, up_e) @ down_w[e]
        out = out / self.num_experts
        if self.use_shared_expert:
            shared = self._shared_expert_forward(flat)
            if self.use_shared_routed_gates:
                out = self.shared_output_gate * shared + self.routed_output_gate * out
            else:
                out = out + shared
        return out.view_as(hidden_states)
