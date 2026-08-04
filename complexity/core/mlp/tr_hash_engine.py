"""MLP adapter that makes the canonical TR-Hash engine a model block.

The historical ``token_routed`` module predates :mod:`complexity.tr_hash` and
contains its own dispatch implementation.  This adapter is the explicit path
for new models that should execute through :class:`TRHashEngine` itself.
"""

from __future__ import annotations

from dataclasses import replace

from .base import MLPBase, MLPConfig
from ..registry import register_mlp
from ...tr_hash import (
    AttentionBackbone,
    TRHashBackend,
    TRHashEngine,
    TRHashEngineConfig,
    TRHashPhase,
    TRHashPrecision,
    TRHashStrategy,
)


def _route_weights(top_k: int, primary_weight: float | None) -> tuple[float, ...]:
    if top_k == 1:
        return (1.0,)
    primary = 1.0 / top_k if primary_weight is None else float(primary_weight)
    primary = min(1.0, max(0.0, primary))
    remainder = (1.0 - primary) / (top_k - 1)
    return (primary, *(remainder for _ in range(top_k - 1)))


def _strategy(value: str) -> TRHashStrategy:
    aliases = {
        "modulo": TRHashStrategy.MODULO_CYCLIC,
        "modulo_cyclic": TRHashStrategy.MODULO_CYCLIC,
        "token_id_balanced_hash": TRHashStrategy.BALANCED_HASH,
    }
    try:
        return aliases[str(value).lower()]
    except KeyError as exc:
        raise ValueError(
            "tr_hash_engine supports modulo_cyclic or token_id_balanced_hash; "
            f"got {value!r}"
        ) from exc


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
    """Shared SwiGLU plus fixed token-ID experts executed by ``TRHashEngine``."""

    def __init__(self, config: MLPConfig):
        super().__init__(config)
        if config.intermediate_size % config.num_experts:
            raise ValueError(
                "TR-Hash routed width must be divisible by num_experts"
            )
        shared_width = (
            int(config.shared_intermediate_size)
            if config.shared_expert and config.shared_intermediate_size
            else 0
        )
        self.top_k = int(config.top_k)
        self._primary_weight = (
            1.0
            if self.top_k == 1
            else float(
                1.0 / self.top_k
                if config.top_k_primary_weight is None
                else config.top_k_primary_weight
            )
        )
        self.engine = TRHashEngine(
            TRHashEngineConfig(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                num_experts=config.num_experts,
                top_k=self.top_k,
                shared_width=shared_width,
                expert_width=config.intermediate_size // config.num_experts,
                routing_strategy=_strategy(config.routing_strategy),
                layer_index=int(config.layer_idx),
                route_weights=_route_weights(
                    self.top_k,
                    config.top_k_primary_weight,
                ),
                shared_output_scale=float(config.shared_output_scale),
                routed_output_scale=float(config.routed_output_scale),
                attention_backbone=AttentionBackbone.GQA,
                phase=TRHashPhase.TRAINING,
                precision=TRHashPrecision.BF16,
                backend=_backend(config),
            )
        )

    def forward(self, hidden_states, token_ids=None, **kwargs):
        if token_ids is None:
            raise ValueError("tr_hash_engine requires token_ids")
        return self.engine(hidden_states, token_ids)

    def set_top_k_primary_weight(self, weight: float) -> None:
        if self.top_k == 1:
            self._primary_weight = 1.0
            return
        self._primary_weight = min(1.0, max(0.0, float(weight)))
        self.engine.config = replace(
            self.engine.config,
            route_weights=_route_weights(self.top_k, self._primary_weight),
        )

    def training_control_capabilities(self) -> frozenset[str]:
        return frozenset({"topk_primary_weight"})

    def training_telemetry(self) -> dict[str, float]:
        return {"topk_w": float(self._primary_weight)}

    def capability_summary(self, device_type: str = "cpu") -> dict:
        return self.engine.capability_summary(device_type)
