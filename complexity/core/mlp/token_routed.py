"""Token-Routed MLP — deterministic residual experts selected by token ID.

Each layer uses a fixed token-ID lookup rather than a learned contextual
router. A shared dense SwiGLU path remains active for every token.

Features:
  - fixed modulo or balanced token-hash lookup tables
  - layer-specific deterministic expert assignments
  - shared dense SwiGLU plus sparse routed residuals
  - historical frequency-aware strategies retained only for ablation replay

Usage:
    config = MLPConfig(hidden_size=512, intermediate_size=2048, num_experts=4)
    mlp = TokenRoutedMLP(config)
    out = mlp(hidden_states, token_ids=token_ids)
"""

import itertools
import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MLPBase, MLPConfig
from .fused_activations import fused_silu_mul
from ..registry import register_mlp
from ...utils.device import supports_custom_triton

logger = logging.getLogger(__name__)

# Try to import CGGR acceleration
try:
    from complexity_cuda.triton_token_routed import (
        sort_tokens_by_expert,
        pair_coverage_hash_expert_ids,
        sort_pair_hash_by_expert,
        cggr_grouped_gemm_triton,
        cggr_grouped_gemm_autograd,
        hash_cggr_grouped_gemm_autograd,
        pair_hash_reduce_autograd,
        pair_hash_weighted_reduce_autograd,
        grouped_gemm_pytorch,
        fused_swiglu_triton,
        HAS_TRITON,
    )
    HAS_CGGR = HAS_TRITON
except Exception:
    HAS_CGGR = False
    cggr_grouped_gemm_autograd = None
    hash_cggr_grouped_gemm_autograd = None
    pair_hash_reduce_autograd = None
    pair_hash_weighted_reduce_autograd = None
    try:
        from complexity_cuda.triton_token_routed import (
            pair_coverage_hash_expert_ids,
            sort_pair_hash_by_expert,
            hash_cggr_grouped_gemm_autograd,
            pair_hash_reduce_autograd,
            pair_hash_weighted_reduce_autograd,
            grouped_gemm_pytorch,
        )
    except Exception:
        pair_coverage_hash_expert_ids = None
        sort_pair_hash_by_expert = None
        hash_cggr_grouped_gemm_autograd = None
        pair_hash_reduce_autograd = None
        pair_hash_weighted_reduce_autograd = None
        grouped_gemm_pytorch = None

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


def _create_pair_coverage_hash_metadata(
    vocab_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the compact, layer-independent state for pair-coverage hashing.

    ``token_rank`` is the seeded permutation hash of token identity. The other
    two tensors are tiny decode tables shared conceptually by every layer:
    canonical unordered expert pairs and the 120 possible six-pair schedules.
    Model construction uses them to compile the per-layer uint8 hash codes
    consumed by CUDA instead of two full int64 expert-ID tables.
    """

    pairs = torch.combinations(
        torch.arange(num_experts, dtype=torch.int32, device="cpu"),
        r=2,
    )
    pair_count = int(pairs.size(0))
    if num_experts < 2 or num_experts > 4:
        raise ValueError(
            "token_id_pair_coverage_hash supports two to four experts"
        )

    token_generator = torch.Generator().manual_seed(0xC04E2A63)
    token_order = torch.randperm(
        vocab_size, generator=token_generator, device="cpu"
    )
    token_rank = torch.empty(vocab_size, dtype=torch.int32, device="cpu")
    token_rank[token_order] = torch.arange(
        vocab_size, dtype=torch.int32, device="cpu"
    )

    # Every permutation has one unique cyclic rotation beginning with pair
    # zero. Assign one such base order to each six-token rank group, then
    # rotate it by the token's rank inside that group. Consequently:
    #   * every token sees every unordered pair once in the first six layers;
    #   * every complete rank group uses every pair once in every layer;
    #   * all pair loads differ by at most one for an incomplete last group;
    #   * the 120 base orders x six rotations retain all 720 early-layer
    #     route signatures instead of collapsing tokens onto six schedules.
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


def _pair_coverage_hash_routes_from_metadata(
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
        # The remaining layers encode the quotient left after selecting one
        # of the canonical base orders. Four base-6 digits plus the 720 early
        # schedules provide 933,120 distinct ten-layer signatures, enough for
        # the 200,019-token vocabulary. Rotation keeps every layer balanced
        # within each six-token group while the full signature remains
        # collision-free.
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
    # Pair weights are exactly 0.5/0.5, so orientation does not change the
    # function. Still alternate it deterministically to keep primary and
    # secondary telemetry approximately symmetric without a second V-entry
    # lookup table in the CUDA planner.
    swap = (token_rank.long() + int(layer_idx)).remainder(2).bool()
    primary = torch.where(swap, selected[:, 1], selected[:, 0])
    secondary = torch.where(swap, selected[:, 0], selected[:, 1])
    return torch.stack((primary, secondary), dim=0)


def _create_pair_coverage_hash_routes(
    vocab_size: int,
    num_experts: int,
    layer_idx: int,
) -> torch.Tensor:
    """Return two token-ID routes with full pair coverage across early layers.

    For four experts there are six unordered expert pairs. Each token receives
    a deterministic permutation of those six pairs over the first six layers,
    so it visits every pair exactly once before any pair can repeat. The
    compact metadata is also consumed directly by the hash-aware CGGR planner.
    """

    metadata = _create_pair_coverage_hash_metadata(vocab_size, num_experts)
    return _pair_coverage_hash_routes_from_metadata(*metadata, layer_idx)


def _encode_pair_coverage_hash_routes(
    routes: torch.Tensor,
    expert_pairs: torch.Tensor,
) -> torch.Tensor:
    """Compile two expert routes into one byte per vocabulary entry.

    Bits 0..2 store the unordered pair index and bit 3 stores orientation.
    Keeping orientation makes the CUDA planner bit-identical to the reference
    lookup even though equal 0.5/0.5 weights make it functionally irrelevant.
    """

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

    Routes tokens through a fixed layer-specific lookup
    (token_id -> expert IDs), then dispatches with sparse masking.
    """

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        self.num_experts = config.num_experts
        self.vocab_size = config.vocab_size
        self.expert_intermediate_size = self.intermediate_size // self.num_experts
        # Top-K deterministic: each token activates K precomputed expert maps.
        # K>1 gives broader conditional capacity while preserving zero learned
        # routing and zero runtime router-network overhead.
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
        expert_initialization = getattr(
            config, "expert_initialization", "gpt_normal"
        )
        if expert_initialization == "legacy_kaiming":
            # Historical checkpoints used the raw tensor layout below with
            # Kaiming's second dimension interpreted as fan-in. Keep this mode
            # only for exact reproduction of those runs.
            for expert_idx in range(self.num_experts):
                nn.init.kaiming_uniform_(
                    self.gate_proj_w[expert_idx], a=5**0.5
                )
                nn.init.kaiming_uniform_(
                    self.up_proj_w[expert_idx], a=5**0.5
                )
                nn.init.kaiming_uniform_(
                    self.down_proj_w[expert_idx], a=5**0.5
                )
        else:
            # Raw nn.Parameter tensors are not visited by the model's
            # nn.Linear initializer. Initialize them explicitly so routed and
            # dense/shared input projections start on the same GPT-style scale.
            std = float(getattr(config, "initializer_range", 0.02))
            nn.init.normal_(self.gate_proj_w, mean=0.0, std=std)
            nn.init.normal_(self.up_proj_w, mean=0.0, std=std)
            nn.init.normal_(self.down_proj_w, mean=0.0, std=std)

        # Shared lexical expert: dense SwiGLU all tokens pass through.
        # Default size = intermediate_size (full dense width). shared_down is
        # also rescaled by _init_residual_scaling() (residual output projection).
        self.use_shared_expert = getattr(config, 'shared_expert', False)
        self.use_shared_routed_gates = bool(getattr(config, "use_shared_routed_gates", False))
        self.shared_output_scale = float(getattr(config, "shared_output_scale", 1.0))
        self.routed_output_scale = float(getattr(config, "routed_output_scale", 1.0))
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
        self.has_pair_coverage_hash = (
            str(getattr(config, "routing_strategy", "")).lower()
            == "token_id_pair_coverage_hash"
        )
        self.has_compact_pair_hash = str(
            getattr(config, "routing_strategy", "")
        ).lower() in {
            "token_id_balanced_hash",
            "token_id_pair_coverage_hash",
        }
        self.learn_hash_pair_gates = bool(
            getattr(config, "learn_hash_pair_gates", False)
        )
        if self.learn_hash_pair_gates and (
            not self.has_compact_pair_hash or self.top_k != 2
        ):
            raise ValueError(
                "learn_hash_pair_gates requires a compact token-ID hash "
                "routing strategy with top_k=2"
            )
        if self.has_compact_pair_hash:
            pair_count = self.num_experts * (self.num_experts - 1) // 2
            if self.gate_proj_w.is_meta:
                pair_hash_expert_pairs = torch.empty(
                    pair_count, 2, dtype=torch.int32, device="meta"
                )
                pair_hash_route_codes = torch.empty(
                    self.vocab_size, dtype=torch.uint8, device="meta"
                )
            else:
                pair_hash_expert_pairs = torch.combinations(
                    torch.arange(self.num_experts, dtype=torch.int32),
                    r=2,
                )
                pair_hash_route_codes = _encode_pair_coverage_hash_routes(
                    topk_token_to_expert,
                    pair_hash_expert_pairs,
                )
            self.register_buffer(
                "pair_hash_route_codes",
                pair_hash_route_codes,
                persistent=True,
            )
            self.register_buffer(
                "pair_hash_expert_pairs",
                pair_hash_expert_pairs,
                persistent=True,
            )
            if self.learn_hash_pair_gates:
                initial_weight = float(
                    getattr(config, "hash_pair_gate_init", 0.5)
                )
                initial_logit = math.log(
                    initial_weight / (1.0 - initial_weight)
                )
                self.hash_pair_gate_logits = nn.Parameter(
                    torch.full(
                        (pair_count,),
                        initial_logit,
                        dtype=torch.float32,
                        device=self.gate_proj_w.device,
                    )
                )

        # Semantic LSH routing: route the primary (and top-k neighbour) experts
        # on a fixed random-hyperplane hash of the hidden state, not the token
        # id. The planes are seeded per layer and saved, so routing is fully
        # deterministic with no learned gate — the expert choice now depends on
        # the contextual representation h (which carries meaning via attention).
        _lsh_layer_idx = int(getattr(config, "layer_idx", 0))
        self.has_lsh_routing = bool(getattr(config, "lsh_routing", False)) and (
            _lsh_layer_idx >= int(getattr(config, "lsh_from_layer", 0))
        )
        if self.has_lsh_routing:
            bits = int(getattr(config, "lsh_bits", 0)) or max(
                1, math.ceil(math.log2(max(2, self.num_experts)))
            )
            self.lsh_bits = bits
            if not self.gate_proj_w.is_meta:
                layer_idx = int(getattr(self.config, "layer_idx", 0))
                g = torch.Generator().manual_seed(0x15B5 + layer_idx)
                planes = torch.randn(bits, self.hidden_size, generator=g)
                planes = planes / planes.norm(dim=1, keepdim=True).clamp_min(1e-12)
                # bucket -> expert: spread the 2**bits buckets across experts.
                buckets = torch.arange(2**bits, dtype=torch.long)
                bucket_to_expert = buckets % self.num_experts
            else:
                planes = torch.empty(bits, self.hidden_size, device="meta")
                bucket_to_expert = torch.empty(2**bits, dtype=torch.long, device="meta")
            self.register_buffer("lsh_planes", planes, persistent=True)
            self.register_buffer("lsh_bucket_to_expert", bucket_to_expert, persistent=True)
            self.register_buffer(
                "lsh_bit_values",
                torch.tensor([2**i for i in range(bits)], dtype=torch.long),
                persistent=False,
            )

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

    def training_control_capabilities(self) -> frozenset[str]:
        capabilities = {"topk_primary_weight", "expert_diversity"}
        if self.use_shared_routed_gates:
            capabilities.add("shared_routed_gates")
        return frozenset(capabilities)

    def training_telemetry(self) -> dict[str, float]:
        values = {"topk_w": float(self._primary_weight)}
        if self.use_shared_routed_gates:
            values.update(
                shared_gate=float(self.shared_output_gate.detach().float().item()),
                routed_gate=float(self.routed_output_gate.detach().float().item()),
            )
        return values

    def _create_token_mapping(self, vocab_size: int, num_experts: int) -> torch.Tensor:
        """
        Create deterministic mapping from token ID to expert ID.

        Canonical routing is token_id modulo E. Frequency-aware construction
        remains available only to replay historical ablations.

        When config.per_layer_routing is True, a deterministic permutation
        of the expert indices specific to this layer is applied after the
        mapping is built. This preserves route cardinality while giving each
        layer a different
        token→expert assignment, enriching specialization.
        """
        strategy = str(
            getattr(self.config, "routing_strategy", "modulo_cyclic")
        ).lower()
        strategy = {
            "modulo_cyclic": "modulo",
        }.get(strategy, strategy)
        freqs = getattr(self.config, 'token_frequencies', None)
        if strategy == "zipf" and freqs is not None:
            freqs = freqs.detach().cpu().double()
            sorted_indices = freqs.argsort(descending=True)
            mapping = torch.empty(vocab_size, dtype=torch.long, device="cpu")
            expert_loads = [0.0] * num_experts
            for rank_pos in range(vocab_size):
                token_id = sorted_indices[rank_pos].item()
                e = min(range(num_experts), key=lambda i: expert_loads[i])
                mapping[token_id] = e
                expert_loads[e] += freqs[token_id].item()
        elif strategy in {
            "zipf",
            "modulo",
            "modulo_balanced_secondary",
            "modulo_frequency_balanced_secondary",
            "lsh_hidden",
        }:
            # zipf without frequencies intentionally falls back to modulo.
            # lsh_hidden replaces this lexical bootstrap in forward(); it still
            # needs a valid table while the module and auxiliary routes are
            # constructed.
            mapping = torch.arange(vocab_size, dtype=torch.long, device="cpu") % num_experts
        elif strategy == "token_id_balanced_hash":
            # Hash the complete token ID into a different exactly-balanced
            # lexical partition at every layer. Unlike merely permuting expert
            # labels after token_id % E, this changes which tokens share an
            # expert while remaining a fixed, cacheable lookup table.
            layer_idx = int(getattr(self.config, "layer_idx", 0))
            token_generator = torch.Generator().manual_seed(
                0x71D5A17 + 104729 * layer_idx
            )
            token_order = torch.randperm(
                vocab_size, generator=token_generator, device="cpu"
            )
            pairs = torch.tensor(
                list(itertools.permutations(range(num_experts), 2)),
                dtype=torch.long,
                device="cpu",
            )
            pair_generator = torch.Generator().manual_seed(
                0x5EC0D + 104729 * layer_idx
            )
            pairs = pairs[
                torch.randperm(
                    pairs.size(0), generator=pair_generator, device="cpu"
                )
            ]
            mapping = torch.empty(vocab_size, dtype=torch.long, device="cpu")
            pair_indices = (
                torch.arange(vocab_size, dtype=torch.long, device="cpu")
                % pairs.size(0)
            )
            mapping[token_order] = pairs[pair_indices, 0]
            return mapping
        elif strategy == "token_id_pair_coverage_hash":
            if self.top_k != 2:
                raise ValueError(
                    "token_id_pair_coverage_hash requires top_k=2"
                )
            return _create_pair_coverage_hash_routes(
                vocab_size,
                num_experts,
                int(getattr(self.config, "layer_idx", 0)),
            )[0]
        elif strategy == "round_robin":
            # Round-robin over frequency rank (or token id if no frequencies),
            # giving a routing-table control distinct from token-id modulo.
            if freqs is not None:
                order = freqs.detach().cpu().double().argsort(descending=True)
            else:
                order = torch.arange(vocab_size, dtype=torch.long, device="cpu")
            mapping = torch.empty(vocab_size, dtype=torch.long, device="cpu")
            mapping[order] = torch.arange(vocab_size, dtype=torch.long, device="cpu") % num_experts
        elif strategy == "random":
            # Deterministic random lexical partition: stable across runs/layers,
            # then layer permutation below still varies expert labels by layer.
            g = torch.Generator().manual_seed(0xA61A710)
            mapping = torch.randint(0, num_experts, (vocab_size,), generator=g, dtype=torch.long, device="cpu")
        else:
            raise ValueError(f"Unsupported lexical routing_strategy: {strategy}")

        # Per-layer routing is always on: a deterministic layer-dependent
        # permutation of expert indices. It preserves route cardinality while forcing each layer to
        # route differently → richer specialization, zero runtime cost.
        layer_idx = int(getattr(self.config, 'layer_idx', 0))
        g = torch.Generator().manual_seed(0xC0DE + layer_idx)
        permutation = torch.randperm(num_experts, generator=g, device="cpu")
        mapping = permutation[mapping]
        return mapping

    def _routing_frequencies(self, vocab_size: int) -> torch.Tensor:
        freqs = getattr(self.config, "token_frequencies", None)
        if freqs is None:
            if str(getattr(self.config, "routing_strategy", "")).lower() == (
                "modulo_frequency_balanced_secondary"
            ):
                raise ValueError(
                    "modulo_frequency_balanced_secondary requires token_frequencies "
                    "from the training partition"
                )
            return torch.ones(vocab_size, dtype=torch.float32, device="cpu")
        if freqs.numel() != vocab_size:
            raise ValueError(
                f"token_frequencies length ({freqs.numel()}) must match vocab_size ({vocab_size})"
            )
        return freqs.detach().cpu().double().clamp_min(0.0)

    def _create_topk_token_mapping(
        self,
        primary_mapping: torch.Tensor,
        vocab_size: int,
        num_experts: int,
        top_k: int,
    ) -> torch.Tensor:
        """Build deterministic auxiliary expert maps.

        For k>0 each token is assigned to an expert different from all earlier
        routes for that token. Canonical modulo routing uses cyclic successors.
        Frequency-balanced auxiliary routes exist only for historical
        ablation replay.
        """

        routes = torch.empty(top_k, vocab_size, dtype=torch.long, device="cpu")
        routes[0] = primary_mapping.detach().to(device="cpu", dtype=torch.long)
        if top_k == 1:
            return routes

        strategy = str(
            getattr(self.config, "routing_strategy", "modulo_cyclic")
        ).lower()
        strategy = {
            "modulo_cyclic": "modulo",
        }.get(strategy, strategy)
        if strategy in {"modulo", "round_robin"}:
            for route_idx in range(1, top_k):
                routes[route_idx] = (routes[0] + route_idx) % num_experts
            return routes

        if strategy == "token_id_balanced_hash":
            # Assign balanced ordered expert tuples to a layer-specific token
            # permutation. For top-2 and four experts this cycles over all 12
            # distinct ordered pairs, so both marginal expert loads and pair
            # loads differ by at most one token.
            layer_idx = int(getattr(self.config, "layer_idx", 0))
            token_generator = torch.Generator().manual_seed(
                0x71D5A17 + 104729 * layer_idx
            )
            token_order = torch.randperm(
                vocab_size, generator=token_generator, device="cpu"
            )
            expert_tuples = torch.tensor(
                list(itertools.permutations(range(num_experts), top_k)),
                dtype=torch.long,
                device="cpu",
            )
            tuple_generator = torch.Generator().manual_seed(
                0x5EC0D + 104729 * layer_idx
            )
            expert_tuples = expert_tuples[
                torch.randperm(
                    expert_tuples.size(0),
                    generator=tuple_generator,
                    device="cpu",
                )
            ]
            tuple_indices = (
                torch.arange(vocab_size, dtype=torch.long, device="cpu")
                % expert_tuples.size(0)
            )
            routes[:, token_order] = expert_tuples[tuple_indices].T
            return routes

        if strategy == "token_id_pair_coverage_hash":
            if top_k != 2:
                raise ValueError(
                    "token_id_pair_coverage_hash requires top_k=2"
                )
            return _create_pair_coverage_hash_routes(
                vocab_size,
                num_experts,
                int(getattr(self.config, "layer_idx", 0)),
            )

        if strategy == "random":
            for route_idx in range(1, top_k):
                g = torch.Generator().manual_seed(0xA61A710 + 7919 * route_idx)
                mapping = torch.randint(
                    0,
                    num_experts - route_idx,
                    (vocab_size,),
                    generator=g,
                    dtype=torch.long,
                    device="cpu",
                )
                for token_id in range(vocab_size):
                    blocked = sorted(int(routes[prev_idx, token_id].item()) for prev_idx in range(route_idx))
                    candidate = int(mapping[token_id].item())
                    for blocked_expert in blocked:
                        if candidate >= blocked_expert:
                            candidate += 1
                    routes[route_idx, token_id] = candidate
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
        hash_cggr_candidate = bool(
            self.has_compact_pair_hash
            and not self.has_lsh_routing
            and token_ids_clamped.is_cuda
            and HAS_CGGR
            and pair_coverage_hash_expert_ids is not None
            and sort_pair_hash_by_expert is not None
            and hash_cggr_grouped_gemm_autograd is not None
            and pair_hash_reduce_autograd is not None
            and (
                not self.learn_hash_pair_gates
                or pair_hash_weighted_reduce_autograd is not None
            )
        )
        if hash_cggr_candidate:
            # Defer the route entirely: the hash-aware CGGR counting partition
            # consumes token IDs and uint8 route codes directly.
            expert_ids = None
            route_expert_ids = None
        else:
            expert_ids = self.token_to_expert[token_ids_clamped]  # [B, S]
            route_expert_ids = self.topk_token_to_expert[
                :, token_ids_clamped
            ]  # [K, B, S]

        # Semantic LSH overlay: replace the lexical routing with a fixed
        # random-hyperplane hash of h. proj = h . planes; the sign bits index a
        # bucket -> expert. The top-k neighbour routes flip the lowest-margin
        # bits (the most ambiguous boundaries), i.e. the nearest buckets.
        if self.has_lsh_routing:
            planes = self.lsh_planes.to(hidden_states.dtype)
            b2e = self.lsh_bucket_to_expert
            bit_vals = self.lsh_bit_values.to(hidden_states.device)
            proj = torch.matmul(hidden_states, planes.t())  # [B, S, bits]
            if getattr(self.config, "lsh_threshold_mode", "zero") == "zero":
                thresh = torch.zeros(proj.shape[-1], dtype=proj.dtype, device=proj.device)
            else:
                # Threshold each plane at the median of the projection over the
                # batch tokens (not 0): guarantees each bit splits ~50/50, which
                # absorbs the residual-stream common mode and prevents the deep-layer
                # bucket collapse a through-origin hyperplane would cause.
                flat_proj = proj.reshape(-1, proj.shape[-1])
                thresh = flat_proj.median(dim=0).values  # [bits]
            bits = (proj > thresh).long()
            bucket = (bits * bit_vals).sum(-1)  # [B, S]
            expert_ids = b2e[bucket]
            routes = [expert_ids]
            if self.top_k >= 2:
                margins = (proj - thresh).abs()
                # flip the j-th smallest-margin bit for each extra route
                flip_order = torch.argsort(margins, dim=-1)  # [B, S, bits]
                for j in range(1, self.top_k):
                    flip_bit = flip_order[..., min(j - 1, self.lsh_bits - 1)]  # [B, S]
                    nb = bucket ^ bit_vals[flip_bit]
                    routes.append(b2e[nb])
            route_expert_ids = torch.stack(routes, dim=0)  # [K, B, S]

        flat_x = hidden_states.view(-1, H)
        flat_expert_ids = (
            expert_ids.view(-1) if expert_ids is not None else None
        )
        static_dispatch = bool(getattr(self.config, "static_expert_capacity", False))
        collect_telemetry = bool(getattr(self.config, "collect_moe_telemetry", False))

        # Track expert utilization (in-place, non-differentiable)
        if collect_telemetry and not static_dispatch:
            with torch.no_grad():
                if route_expert_ids is None:
                    route_expert_ids = pair_coverage_hash_expert_ids(
                        token_ids_clamped,
                        self.pair_hash_route_codes,
                        self.pair_hash_expert_pairs,
                        vocab_size=self.vocab_size,
                    )
                    expert_ids = route_expert_ids[0]
                    flat_expert_ids = expert_ids.view(-1)
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
        use_equal_top2_pair = (
            not static_dispatch
            and self.top_k == 2
            and abs(self._primary_weight - 0.5) <= 1e-12
        )
        use_fused_pair_cggr = use_equal_top2_pair and use_cggr
        use_fused_pair_fallback = use_equal_top2_pair and not use_cggr
        use_hash_route_kernel = (
            hash_cggr_candidate and use_fused_pair_cggr
        )
        if route_expert_ids is None and not use_hash_route_kernel:
            expert_ids = self.token_to_expert[token_ids_clamped]
            route_expert_ids = self.topk_token_to_expert[
                :, token_ids_clamped
            ]
            flat_expert_ids = expert_ids.view(-1)
        if use_fused_pair_cggr:
            self.last_dispatch_path = (
                "top2_hash_native_cggr"
                if use_hash_route_kernel
                else "top2_pair_cggr"
            )
        elif use_fused_pair_fallback:
            self.last_dispatch_path = "top2_pair_fused"
        else:
            self.last_dispatch_path = "cggr" if use_cggr else "masked_dense"

        # Log path selection once per path/device/policy combo. Helps diagnose
        # whether a run is on CGGR or the universal no-sync fallback.
        log_key = (self.last_dispatch_path, str(cggr_policy), str(kernel_policy), flat_x.device.type)
        logged_keys = getattr(self.__class__, "_path_logged_keys", set())
        if log_key not in logged_keys:
            logged_keys.add(log_key)
            self.__class__._path_logged_keys = logged_keys
            if use_cggr:
                if use_fused_pair_cggr and use_hash_route_kernel:
                    logger.info(
                        "[TokenRoutedMLP] dispatch path = "
                        "top2_hash_native_cggr (token-ID hash partition, "
                        "indirect grouped projections, fused route reduction)"
                    )
                elif use_fused_pair_cggr:
                    logger.info(
                        "[TokenRoutedMLP] dispatch path = top2_pair_cggr "
                        "(one combined partition, Triton grouped-GEMM)"
                    )
                else:
                    logger.info(
                        "[TokenRoutedMLP] dispatch path = CGGR "
                        "(Triton grouped-GEMM, no sync)"
                    )
            elif use_fused_pair_fallback:
                logger.info(
                    "[TokenRoutedMLP] dispatch path = top2_pair_fused "
                    "(equal-weight pair, one expert pass)"
                )
            else:
                logger.info(
                    "[TokenRoutedMLP] dispatch path = masked_dense (no CPU sync) "
                    f"(CGGR rejected: {', '.join(why_not_cggr) if why_not_cggr else 'unknown'})"
                )

        # Top-K deterministic lookup: dispatch K precomputed expert maps.
        # Primary keeps the configured weight; auxiliaries share the remainder.
        if use_hash_route_kernel:
            routed_out = self._dispatch_equal_top2_hash_cggr(
                flat_x,
                token_ids_clamped.view(-1),
                gate_w,
                up_w,
                down_w,
                H,
            )
        elif use_fused_pair_cggr:
            primary_weights = (
                self._hash_pair_primary_weights(token_ids_clamped.view(-1))
                if self.learn_hash_pair_gates
                else None
            )
            routed_out = self._dispatch_equal_top2_pair_cggr(
                flat_x,
                route_expert_ids.view(2, -1),
                gate_w,
                up_w,
                down_w,
                H,
                primary_weights=primary_weights,
            )
        elif use_fused_pair_fallback:
            primary_weights = (
                self._hash_pair_primary_weights(token_ids_clamped.view(-1))
                if self.learn_hash_pair_gates
                else None
            )
            routed_out = self._dispatch_equal_top2_pair(
                flat_x,
                route_expert_ids.view(2, -1),
                gate_w,
                up_w,
                down_w,
                primary_weights=primary_weights,
            )
        elif static_dispatch:
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
            out = (
                self.shared_output_scale * self.shared_output_gate * shared_out
                + self.routed_output_scale * self.routed_output_gate * routed_out
            )
        else:
            out = (
                self.shared_output_scale * shared_out
                + self.routed_output_scale * routed_out
            )
        return out.view(B, S, H)

    def _dispatch_equal_top2_pair(
        self,
        flat_x: torch.Tensor,
        route_expert_ids: torch.Tensor,
        gate_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
        *,
        primary_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dispatch an equal-weight top-2 pair with one pass per expert.

        The generic top-k fallback dispatches the primary and secondary maps
        separately. With fixed 0.5/0.5 routing that evaluates every expert
        twice over the full token block. A pair hash already exposes both
        selected experts, so compute each expert once and apply its per-token
        multiplicity after the non-linearity. This preserves the exact
        mathematical function, supports repeated routes defensively, avoids
        both sorting passes, and halves routed fallback GEMMs.
        """

        if route_expert_ids.ndim != 2 or route_expert_ids.size(0) != 2:
            raise ValueError(
                "equal top-2 pair dispatch expects expert IDs shaped [2, N]"
            )
        if route_expert_ids.size(1) != flat_x.size(0):
            raise ValueError(
                "route expert IDs must match the flattened token count"
            )

        routed_out = torch.zeros_like(flat_x)
        if primary_weights is None:
            route_weights = torch.full(
                route_expert_ids.shape,
                0.5,
                dtype=flat_x.dtype,
                device=flat_x.device,
            )
        else:
            route_weights = torch.stack(
                (primary_weights, 1.0 - primary_weights),
                dim=0,
            ).to(flat_x.dtype)
        for expert_idx in range(self.num_experts):
            route_weight = (
                route_expert_ids.eq(expert_idx).to(flat_x.dtype)
                * route_weights
            ).sum(dim=0)
            active = route_weight.ne(0).unsqueeze(-1).to(flat_x.dtype)
            expert_x = flat_x * active
            gate = expert_x @ gate_w[expert_idx]
            up = expert_x @ up_w[expert_idx]
            expert_out = (
                fused_silu_mul(gate, up) @ down_w[expert_idx]
            ).to(routed_out.dtype)
            routed_out = (
                routed_out
                + expert_out * route_weight.unsqueeze(-1)
            )
        return routed_out

    def _dispatch_equal_top2_pair_cggr(
        self,
        flat_x: torch.Tensor,
        route_expert_ids: torch.Tensor,
        gate_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
        hidden_size: int,
        *,
        use_cggr: bool = True,
        primary_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dispatch both equal-weight routes through one grouped batch.

        The generic CGGR path sorts and launches grouped GEMMs once per route.
        A fixed pair hash allows both assignments to be concatenated, sorted
        once, and processed by one gate/up/down grouped-GEMM sequence. The
        final mean restores the 0.5/0.5 mixture.
        """

        if route_expert_ids.ndim != 2 or route_expert_ids.size(0) != 2:
            raise ValueError(
                "equal top-2 CGGR dispatch expects expert IDs shaped [2, N]"
            )
        if route_expert_ids.size(1) != flat_x.size(0):
            raise ValueError(
                "route expert IDs must match the flattened token count"
            )

        pair_x = flat_x.unsqueeze(0).expand(2, -1, -1).reshape(
            -1, hidden_size
        )
        pair_expert_ids = route_expert_ids.reshape(-1)
        sorted_x, sorted_idx, expert_offsets, expert_counts = (
            sort_tokens_by_expert(
                pair_x,
                pair_expert_ids,
                self.num_experts,
            )
        )
        pair_out = self._dispatch_sorted(
            pair_x,
            sorted_x,
            sorted_idx,
            expert_offsets,
            expert_counts,
            gate_w,
            up_w,
            down_w,
            use_cggr,
            hidden_size,
        )
        pair_out = pair_out.view(2, flat_x.size(0), hidden_size)
        if primary_weights is None:
            return pair_out.mean(dim=0)
        primary_weights = primary_weights.to(pair_out.dtype).unsqueeze(-1)
        return (
            pair_out[0] * primary_weights
            + pair_out[1] * (1.0 - primary_weights)
        )

    def _hash_pair_primary_weights(
        self,
        flat_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the learned primary weight for each fixed token-ID pair."""

        route_codes = self.pair_hash_route_codes[
            flat_token_ids.clamp(0, self.vocab_size - 1)
        ].to(torch.long)
        pair_indices = route_codes.bitwise_and(0x7)
        expert_a_weights = torch.sigmoid(
            self.hash_pair_gate_logits[pair_indices]
        )
        return torch.where(
            route_codes.bitwise_and(0x8).ne(0),
            1.0 - expert_a_weights,
            expert_a_weights,
        )

    def _dispatch_equal_top2_hash_cggr(
        self,
        flat_x: torch.Tensor,
        flat_token_ids: torch.Tensor,
        gate_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
        hidden_size: int,
        *,
        use_cggr: bool = True,
    ) -> torch.Tensor:
        """Run equal top-2 routing through a fully hash-native CGGR path.

        The hash is counting-partitioned once. Gate/up read the original token
        matrix indirectly, so no duplicated ``[2*N, hidden]`` sorted input is
        created. The down projection stays expert-contiguous and a final
        Triton reduction writes the pair mean directly in token order.
        """

        if flat_token_ids.ndim != 1:
            raise ValueError("hash CGGR expects flattened token IDs")
        if flat_token_ids.numel() != flat_x.size(0):
            raise ValueError("token IDs must match the flattened token count")
        (
            assignment_indices,
            inverse_indices,
            expert_offsets,
            _expert_counts,
        ) = (
            sort_pair_hash_by_expert(
                flat_token_ids,
                self.pair_hash_route_codes,
                self.pair_hash_expert_pairs,
                vocab_size=self.vocab_size,
                num_experts=self.num_experts,
                return_inverse=True,
            )
        )
        token_count = flat_x.size(0)
        gate_out = hash_cggr_grouped_gemm_autograd(
            flat_x,
            gate_w,
            expert_offsets,
            assignment_indices,
            inverse_indices,
        )
        up_out = hash_cggr_grouped_gemm_autograd(
            flat_x,
            up_w,
            expert_offsets,
            assignment_indices,
            inverse_indices,
        )
        intermediate = fused_silu_mul(gate_out, up_out)
        if use_cggr and cggr_grouped_gemm_autograd is not None:
            sorted_routed = cggr_grouped_gemm_autograd(
                intermediate,
                down_w,
                expert_offsets,
            )
        else:
            sorted_routed = grouped_gemm_pytorch(
                intermediate,
                down_w,
                expert_offsets,
                torch.diff(expert_offsets),
            )
        if self.learn_hash_pair_gates:
            primary_weights = self._hash_pair_primary_weights(flat_token_ids)
            return pair_hash_weighted_reduce_autograd(
                sorted_routed,
                assignment_indices,
                inverse_indices,
                primary_weights,
                token_count,
            )
        return pair_hash_reduce_autograd(
            sorted_routed,
            assignment_indices,
            inverse_indices,
            token_count,
            scale=0.5,
        )

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
