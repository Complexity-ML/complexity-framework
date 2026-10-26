"""Model-block adapter for the canonical shared-plus-four-expert TR-HASH engine."""

from __future__ import annotations
from ...tr_hash import TRHashBackend, TRHashEngine, TRHashEngineConfig, TRHashPrecision
from ..registry import register_mlp
from .base import MLPBase, MLPConfig


def _backend(config: MLPConfig) -> TRHashBackend:
    kernel_policy = config.use_custom_kernels
    cggr_policy = config.use_cggr
    if isinstance(kernel_policy, str):
        kernel_policy = kernel_policy.strip().lower()
    if isinstance(cggr_policy, str):
        cggr_policy = cggr_policy.strip().lower()
    if kernel_policy in {False, "false"}:
        return TRHashBackend.PYTORCH
    if cggr_policy in {True, "true"}:
        return TRHashBackend.CGGR
    return TRHashBackend.AUTO


@register_mlp("tr_hash_engine")
@register_mlp("tr_hash_moe")
class TRHashEngineMLP(MLPBase):
    def __init__(self, config: MLPConfig):
        super().__init__(config)
        if config.expert_initialization != "gpt_normal" or config.hidden_act != "silu":
            raise ValueError("TR-HASH requires normal initialization and SwiGLU")
        if (
            config.use_shared_routed_gates
            or config.learn_hash_pair_gates
            or config.learn_hash_channel_modulation
            or config.lsh_routing
            or config.hash_routing
            or config.token_frequencies is not None
        ):
            raise ValueError("TR-HASH uses fixed token-ID routes and fixed branch weights")
        if config.shared_gate_init != 1.0 or config.routed_gate_init != 1.0:
            raise ValueError("TR-HASH does not use learned branch gates")
        if not config.shared_expert:
            raise ValueError("TR-HASH requires its shared expert")
        if config.intermediate_size % 4:
            raise ValueError("TR-HASH routed width must be divisible by four")
        self.top_k = 2
        self._primary_weight = 0.5
        self.engine = TRHashEngine(
            TRHashEngineConfig(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                num_experts=config.num_experts,
                top_k=config.top_k,
                shared_width=config.shared_intermediate_size or 2 * config.hidden_size,
                expert_width=config.intermediate_size // 4,
                initializer_range=float(config.initializer_range),
                routing_strategy=config.routing_strategy,
                layer_index=int(config.layer_idx),
                route_weights=(0.5, 0.5)
                if config.top_k_primary_weight is None
                else (config.top_k_primary_weight, 1 - config.top_k_primary_weight),
                shared_output_scale=float(config.shared_output_scale),
                routed_output_scale=float(config.routed_output_scale),
                precision=TRHashPrecision.BF16,
                backend=_backend(config),
            )
        )
        if config.active_num_experts is not None or config.active_expert_width is not None:
            self.engine.set_active_capacity(
                num_experts=config.active_num_experts, expert_width=config.active_expert_width
            )

    def forward(self, hidden_states, token_ids=None, **kwargs):
        if token_ids is None:
            raise ValueError("TR-HASH requires original token_ids")
        return self.engine(
            hidden_states, token_ids, previous_token_ids=kwargs.get("previous_token_ids")
        )

    def set_top_k_primary_weight(self, weight: float) -> None:
        if float(weight) != 0.5:
            raise ValueError("TR-HASH requires a fixed top_k_primary_weight of 0.5")

    def set_active_experts(self, num_experts: int) -> None:
        self.engine.set_active_capacity(num_experts=num_experts)

    def set_active_expert_width(self, width: int) -> None:
        self.engine.set_active_capacity(expert_width=width)

    def training_control_capabilities(self) -> frozenset[str]:
        return frozenset({"active_expert_width"})

    def training_telemetry(self) -> dict[str, float]:
        return {
            "topk_w": 0.5,
            "active_num_experts": 4.0,
            "active_expert_width": float(self.engine.active_expert_width),
        }

    def capability_summary(self, device_type: str = "cpu") -> dict:
        return self.engine.capability_summary(device_type)
