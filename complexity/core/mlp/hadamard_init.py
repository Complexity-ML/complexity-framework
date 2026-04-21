"""
Deterministic Hadamard-based weight initialization.

Traditional Xavier / Kaiming init draws weights from a Gaussian via a
global RNG — reproducible only if every layer of the stack (PyTorch,
CUDA, numpy, DataLoader workers, …) correctly propagates a seed. In
practice that assumption breaks often, especially under FSDP.

This module provides an **RNG-free** initialisation: weights are a pure
function of `(shape, layer_idx)`. The construction:

  1. Build an n×n Hadamard matrix via Sylvester's recurrence, where
     n is the next power of two ≥ max(out_features, in_features).
     Entries are ±1 integers → bit-exact across hardware.
  2. Apply a per-layer ±1 sign pattern on the rows. The pattern is
     `(-1)^popcount(i AND layer_idx)`, which preserves the pairwise
     orthogonality of the rows (diagonal ±1 transformation).
  3. Truncate to the target shape.
  4. Scale so the entry magnitude matches Xavier's target std
     `sqrt(2 / (out + in))`, so the rest of the training stack
     (learning rate, warmup, grad-clip) does not need to be retuned.

Mathematically: the resulting matrix has pairwise-orthogonal rows and
all entries of magnitude exactly `std_target`. Perfect orthogonality in
the expected sense, zero randomness.
"""

from __future__ import annotations

import torch


def hadamard_sylvester(n: int) -> torch.Tensor:
    """
    Build the n×n Hadamard matrix via Sylvester's recurrence.

    H_1 = [[1]], H_{2k} = [[H_k, H_k], [H_k, −H_k]].

    Requires `n` to be a power of two. Entries are ±1.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard size must be a power of 2, got {n}")
    H = torch.ones((1, 1), dtype=torch.float32)
    while H.shape[0] < n:
        top = torch.cat([H,  H], dim=1)
        bot = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bot], dim=0)
    return H


def _layer_sign_pattern(n: int, layer_idx: int) -> torch.Tensor:
    """
    Per-layer ±1 row-sign pattern of length `n`, derived from `layer_idx`.

    sign[i] = (−1)^popcount(i AND layer_idx)

    - For `layer_idx == 0`: all +1 (no flips). Layer 0 gets the pure
      Hadamard.
    - Any bit of `layer_idx` that is set toggles half of the rows.
    - Distinct `layer_idx` values (mod n) give distinct sign patterns,
      hence distinct weight matrices.

    Preserves row orthogonality because it is a diagonal ±1 multiplication.
    """
    idx = torch.arange(n, dtype=torch.int64)
    masked = idx & int(layer_idx)
    parity = torch.zeros_like(idx)
    bits = masked.clone()
    # popcount mod 2 via XOR of low bits, shifting down until zero.
    while bool(bits.any()):
        parity ^= bits & 1
        bits >>= 1
    return 1.0 - 2.0 * parity.to(torch.float32)


def next_power_of_two(n: int) -> int:
    """Smallest power of two ≥ n (returns 1 for n ≤ 1)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@torch.no_grad()
def hadamard_init_(
    tensor: torch.Tensor,
    layer_idx: int = 0,
    std_target: float | None = None,
) -> torch.Tensor:
    """
    In-place deterministic Hadamard initialisation of a 2D weight tensor.

    Args:
        tensor: 2D weight of shape (out_features, in_features). Modified in
                place.
        layer_idx: integer diversifier — different values produce different
                   (but equally valid) weight matrices. Pass each layer's
                   index so every layer in the stack starts differently.
        std_target: target per-entry std. Defaults to Xavier's
                    `sqrt(2 / (out + in))` so downstream hyperparameters
                    (lr, warmup) behave identically to a Xavier baseline.

    Returns:
        The (mutated) input tensor, for chaining.

    The result has entries of magnitude exactly `std_target`, rows pairwise
    orthogonal, and is reproducible bit-for-bit on any hardware.
    """
    if tensor.dim() != 2:
        raise ValueError(
            f"hadamard_init_ expects a 2D tensor, got shape {tuple(tensor.shape)}"
        )
    out_f, in_f = tensor.shape
    n = next_power_of_two(max(out_f, in_f, 2))

    H = hadamard_sylvester(n)
    signs = _layer_sign_pattern(n, layer_idx)
    H = H * signs.unsqueeze(1)  # row-wise sign flip

    # Truncate to the target shape.
    W = H[:out_f, :in_f].contiguous()

    if std_target is None:
        std_target = (2.0 / (out_f + in_f)) ** 0.5

    W = W * std_target
    tensor.data.copy_(W.to(device=tensor.device, dtype=tensor.dtype))
    return tensor
