"""Pure causal expert routing, head selection and value-channel binding."""

from __future__ import annotations

import torch


def _validate_token_ids(token_ids: torch.Tensor) -> None:
    if token_ids.ndim != 2 or token_ids.dtype != torch.long:
        raise ValueError("token_ids must be a torch.long [batch, sequence] tensor")


def _routing_history(
    token_ids: torch.Tensor,
    previous_token_ids: torch.Tensor | None,
) -> torch.Tensor:
    _validate_token_ids(token_ids)
    if previous_token_ids is None:
        return token_ids.new_zeros((token_ids.size(0), 2))
    if (
        previous_token_ids.shape != (token_ids.size(0), 2)
        or previous_token_ids.dtype != torch.long
        or previous_token_ids.device != token_ids.device
    ):
        raise ValueError("previous_token_ids must be torch.long [batch, 2] on the token_ids device")
    return previous_token_ids


def update_routing_history(
    token_ids: torch.Tensor,
    previous_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the last two IDs, oldest first, without mutating either input."""

    history = _routing_history(token_ids, previous_token_ids)
    return torch.cat((history, token_ids), dim=1)[:, -2:].clone()


def composed_mlp_routes(
    token_ids: torch.Tensor,
    layer_index: int,
    previous_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return exact causal TR-HASH MLP routes as ``[2, batch, sequence]``.

    A missing prefix is zero padded. The special M1 formula applies only to
    absolute layer index one; subsequent layers use the Source formula's
    general branch rather than repeating a two-layer schedule.
    """

    if layer_index < 0:
        raise ValueError("layer_index must be non-negative")
    history = _routing_history(token_ids, previous_token_ids)
    combined = torch.cat((history, token_ids), dim=1)
    sequence = token_ids.size(1)
    previous = combined[:, 1 : sequence + 1]
    before_previous = combined[:, :sequence]
    context = (previous + 5 * before_previous).remainder(9)
    if layer_index == 1:
        # Adding two modulo four flips the same bit as XOR with two.
        primary = ((token_ids + 1).remainder(4) + 2 * context.remainder(3).remainder(2)).remainder(4)
        stride = 1 + (token_ids // 4 + 2 * (context // 3) + 1).remainder(3)
    else:
        primary = (token_ids + layer_index).remainder(4)
        context_term = (1 + layer_index % 2) * context.remainder(3)
        stride = 1 + (token_ids // 4 + context_term + layer_index).remainder(3)
    return torch.stack((primary, (primary + stride).remainder(4)), dim=0)


def composed_head_routes(
    token_ids: torch.Tensor,
    layer_index: int,
    num_heads: int,
) -> torch.Tensor:
    """Return deterministic selected heads as ``[batch, sequence, min(2,N)]``.

    The architecture uses four query heads and two selected heads.
    """

    _validate_token_ids(token_ids)
    if layer_index < 0:
        raise ValueError("layer_index must be non-negative")
    if num_heads != 4:
        raise ValueError("TR-HASH requires exactly four query heads")
    if layer_index == 1:
        primary = (token_ids // (num_heads * (num_heads - 1))).remainder(num_heads)
        secondary = (primary + 1).remainder(num_heads)
    else:
        primary = (token_ids + layer_index).remainder(num_heads)
        stride = 1 + (token_ids // num_heads + layer_index).remainder(num_heads - 1)
        secondary = (primary + stride).remainder(num_heads)
    return torch.stack((primary, secondary), dim=-1)


def bind_value_channels(
    hidden_states: torch.Tensor,
    token_ids: torch.Tensor,
    layer_index: int,
) -> torch.Tensor:
    """Rotate V's input channel quarters at layer zero using original IDs."""

    _validate_token_ids(token_ids)
    if layer_index < 0:
        raise ValueError("layer_index must be non-negative")
    if (
        hidden_states.ndim != 3
        or hidden_states.shape[:2] != token_ids.shape
        or hidden_states.device != token_ids.device
    ):
        raise ValueError("hidden_states and token_ids must share batch, sequence, and device")
    if layer_index != 0:
        return hidden_states
    hidden_size = hidden_states.size(-1)
    if hidden_size == 0 or hidden_size % 4:
        raise ValueError("TR-HASH value binding requires hidden_size divisible by four")
    channels = torch.arange(hidden_size, device=token_ids.device)
    indices = (channels + (hidden_size // 4) * token_ids.remainder(4).unsqueeze(-1)).remainder(
        hidden_size
    )
    return hidden_states.gather(-1, indices)
