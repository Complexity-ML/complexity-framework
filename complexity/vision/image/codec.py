"""Frozen image codec used outside the trainable TR-Hash model."""

from __future__ import annotations

import torch
import torch.nn as nn


class FrozenAutoencoderKL(nn.Module):
    """Thin optional Diffusers adapter with the latent scaling contract enforced."""

    def __init__(self, model_id: str, *, local_files_only: bool = False):
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:
            raise ImportError(
                "image training requires the optional dependency: "
                "pip install -e '.[image]'"
            ) from exc
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        self.vae.requires_grad_(False).eval()
        self.scaling_factor = float(self.vae.config.scaling_factor)

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        posterior = self.vae.encode(pixels.float()).latent_dist
        return posterior.sample() * self.scaling_factor

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        pixels = self.vae.decode(latents.float() / self.scaling_factor).sample
        return pixels.clamp(-1.0, 1.0)

