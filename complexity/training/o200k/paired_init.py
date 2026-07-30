"""Paired dense-to-routed initialization for controlled architecture studies."""

from __future__ import annotations

from dataclasses import replace

import torch

from complexity.config import ModelConfig
from complexity.core.mlp.standard import SwiGLUMLP
from complexity.core.mlp.token_routed import TokenRoutedMLP
from complexity.models import ComplexityModel


def dense_reference_config(routed_config: ModelConfig) -> ModelConfig:
    """Return the parameter-matched dense config for a shared+routed model."""

    if routed_config.mlp_type != "token_routed":
        raise ValueError("paired dense initialization requires mlp_type=token_routed")
    if not routed_config.shared_expert:
        raise ValueError("paired dense initialization requires a shared expert")
    shared_width = int(routed_config.shared_intermediate_size or 0)
    routed_width = int(routed_config.intermediate_size or 0)
    if shared_width <= 0 or routed_width <= 0:
        raise ValueError("shared and routed intermediate widths must be positive")
    return replace(
        routed_config,
        mlp_type="swiglu",
        intermediate_size=shared_width + routed_width,
        shared_expert=False,
        shared_intermediate_size=shared_width + routed_width,
    )


@torch.no_grad()
def initialize_token_routed_from_dense_reference(
    routed_model: ComplexityModel,
    dense_model: ComplexityModel,
) -> dict[str, int]:
    """Copy one dense initialization into its shared+routed reparameterization.

    Common embedding, attention and normalization tensors are copied exactly.
    Each dense SwiGLU is then split into a shared prefix and equal contiguous
    routed-expert blocks. This removes architecture-dependent RNG drift from a
    matched Dense-vs-TR experiment.
    """

    if len(routed_model.layers) != len(dense_model.layers):
        raise ValueError("dense and routed models must have the same depth")

    routed_state = routed_model.state_dict()
    dense_state = dense_model.state_dict()
    common_tensors = 0
    for name, target in routed_state.items():
        if ".mlp." in name:
            continue
        source = dense_state.get(name)
        if source is None or source.shape != target.shape:
            continue
        target.copy_(source)
        common_tensors += 1

    split_layers = 0
    for layer_index, (routed_layer, dense_layer) in enumerate(
        zip(routed_model.layers, dense_model.layers, strict=True)
    ):
        routed_mlp = routed_layer.mlp
        dense_mlp = dense_layer.mlp
        if not isinstance(routed_mlp, TokenRoutedMLP):
            raise TypeError(f"layer {layer_index} is not TokenRoutedMLP")
        if not isinstance(dense_mlp, SwiGLUMLP):
            raise TypeError(f"dense layer {layer_index} is not SwiGLUMLP")
        if not routed_mlp.use_shared_expert:
            raise ValueError(f"routed layer {layer_index} has no shared expert")

        shared_width = routed_mlp.shared_gate.out_features
        expert_width = routed_mlp.expert_intermediate_size
        routed_width = routed_mlp.num_experts * expert_width
        dense_width = dense_mlp.gate_proj.out_features
        if shared_width + routed_width != dense_width:
            raise ValueError(
                "dense width must equal shared width plus all routed widths: "
                f"{dense_width} != {shared_width} + {routed_width}"
            )

        routed_mlp.shared_gate.weight.copy_(
            dense_mlp.gate_proj.weight[:shared_width]
        )
        routed_mlp.shared_up.weight.copy_(
            dense_mlp.up_proj.weight[:shared_width]
        )
        routed_mlp.shared_down.weight.copy_(
            dense_mlp.down_proj.weight[:, :shared_width]
        )

        for expert_index in range(routed_mlp.num_experts):
            start = shared_width + expert_index * expert_width
            end = start + expert_width
            routed_mlp.gate_proj_w[expert_index].copy_(
                dense_mlp.gate_proj.weight[start:end].transpose(0, 1)
            )
            routed_mlp.up_proj_w[expert_index].copy_(
                dense_mlp.up_proj.weight[start:end].transpose(0, 1)
            )
            routed_mlp.down_proj_w[expert_index].copy_(
                dense_mlp.down_proj.weight[:, start:end].transpose(0, 1)
            )
        split_layers += 1

    return {
        "common_tensors": common_tensors,
        "split_layers": split_layers,
    }
