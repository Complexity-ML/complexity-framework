"""Instruction-guided image editing with spatial TR-Hash conditioning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .config import TRHashImageConfig
from .model import TRHashTextToImage, _modulate, _timestep_embedding


def sample_edit_condition_dropout(
    batch: int,
    *,
    caption_dropout: float,
    source_dropout: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample image-only and unconditional CFG training branches.

    An edit instruction does not describe target content without its source
    image. Dropping the source must therefore also drop text; independently
    dropping text while retaining the source trains the image-only branch.
    """

    dropped_text = torch.rand(batch, device=device, generator=generator) < caption_dropout
    dropped_source = torch.rand(batch, device=device, generator=generator) < source_dropout
    return dropped_text | dropped_source, dropped_source


@dataclass(frozen=True)
class TRHashImageEditConfig(TRHashImageConfig):
    """Architecture contract for source-image + instruction editing."""

    source_dropout: float = 0.05

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.source_dropout < 1.0:
            raise ValueError("source_dropout must be in [0, 1)")

    def estimated_parameter_count(self) -> int:
        base = super().estimated_parameter_count()
        h = self.hidden_size
        source_projection = self.latent_patch_features * h + h
        source_norm = 2 * h
        source_gates = (self.num_layers + 2) * h
        return base + source_projection + source_norm + source_gates

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TRHashImageEditConfig":
        return cls(**values)


class TRHashImageEditor(TRHashTextToImage):
    """Generate a target image from a source image and a text instruction.

    The noisy target follows the same rectified-flow path as text-to-image.
    Aligned source-latent patches are injected before the transformer and at
    every TR-Hash block. Zero-initialized gates make a newly constructed editor
    exactly compatible with text-to-image weights before edit fine-tuning.
    """

    config: TRHashImageEditConfig

    def __init__(self, config: Optional[TRHashImageEditConfig] = None):
        super().__init__(config or TRHashImageEditConfig())
        config = self.config
        self.source_latent_in = nn.Linear(
            config.latent_patch_features,
            config.hidden_size,
        )
        self.source_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.source_input_gate = nn.Parameter(torch.zeros(config.hidden_size))
        self.source_condition_gate = nn.Parameter(torch.zeros(config.hidden_size))
        self.source_block_gates = nn.Parameter(torch.zeros(config.num_layers, config.hidden_size))

    def load_text_to_image_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> Sequence[str]:
        """Load a base text-to-image checkpoint while retaining edit modules."""

        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                "unexpected text-to-image checkpoint keys: "
                + ", ".join(incompatible.unexpected_keys)
            )
        allowed_prefixes = (
            "source_latent_in.",
            "source_norm.",
            "source_input_gate",
            "source_condition_gate",
            "source_block_gates",
        )
        invalid = [key for key in incompatible.missing_keys if not key.startswith(allowed_prefixes)]
        if invalid:
            raise ValueError("base checkpoint is missing shared keys: " + ", ".join(invalid))
        return tuple(incompatible.missing_keys)

    def _source_tokens(self, source_latents: torch.Tensor) -> torch.Tensor:
        tokens = self.source_latent_in(self._patchify(source_latents))
        return self.source_norm(tokens + self.position_embedding)

    def forward(
        self,
        latents: torch.Tensor,
        source_latents: torch.Tensor,
        timesteps: torch.Tensor,
        instruction_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        if source_latents.shape != latents.shape:
            raise ValueError("source and target latents must have identical shapes")
        if timesteps.shape != (latents.size(0),):
            raise ValueError("one timestep is required per latent sample")

        text, text_mask = self.text_encoder(instruction_ids, instruction_mask)
        masked = text * text_mask.unsqueeze(-1).to(text.dtype)
        pooled_text = masked.sum(dim=1) / text_mask.sum(dim=1, keepdim=True).clamp_min(1)
        time = self.time_mlp(
            _timestep_embedding(timesteps, self.config.hidden_size).to(latents.dtype)
        )
        source = self._source_tokens(source_latents)
        source_pooled = source.mean(dim=1)
        condition = pooled_text + time
        condition = condition + self.source_condition_gate.tanh() * source_pooled

        x = self.latent_in(self._patchify(latents)) + self.position_embedding
        x = x + self.source_input_gate.tanh()[None, None] * source
        route_ids = self.build_image_route_ids(timesteps)
        for index, block in enumerate(self.blocks):
            x = x + self.source_block_gates[index].tanh()[None, None] * source
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    condition,
                    text,
                    text_mask,
                    route_ids,
                    use_reentrant=False,
                )
            else:
                x = block(x, condition, text, text_mask, route_ids)
        shift, scale = self.final_condition(condition).chunk(2, dim=-1)
        x = _modulate(F.layer_norm(x, (x.size(-1),)), shift, scale)
        return self._unpatchify(self.latent_out(x))

    def flow_matching_loss(
        self,
        target_latents: torch.Tensor,
        source_latents: torch.Tensor,
        instruction_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if source_latents.shape != target_latents.shape:
            raise ValueError("source and target latents must have identical shapes")
        batch = target_latents.size(0)
        noise = torch.randn(
            target_latents.shape,
            dtype=target_latents.dtype,
            device=target_latents.device,
            generator=generator,
        )
        timesteps = torch.rand(
            batch,
            dtype=target_latents.dtype,
            device=target_latents.device,
            generator=generator,
        )
        training_ids = instruction_ids
        training_mask = instruction_mask.clone()
        training_source = source_latents
        if self.training:
            dropped_text, dropped_source = sample_edit_condition_dropout(
                batch,
                caption_dropout=self.config.caption_dropout,
                source_dropout=self.config.source_dropout,
                device=target_latents.device,
                generator=generator,
            )
            training_ids = instruction_ids.clone()
            training_ids[dropped_text] = 0
            training_mask[dropped_text] = False
            training_source = source_latents * (~dropped_source).to(source_latents.dtype).view(
                batch, 1, 1, 1
            )
        shape = (batch,) + (1,) * (target_latents.ndim - 1)
        interpolation = timesteps.view(shape)
        noisy = (1.0 - interpolation) * target_latents + interpolation * noise
        prediction = self(
            noisy,
            training_source,
            timesteps,
            training_ids,
            training_mask,
        )
        return F.mse_loss(prediction.float(), (noise - target_latents).float())

    @torch.no_grad()
    def edit(
        self,
        source_latents: torch.Tensor,
        instruction_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
        *,
        steps: int = 30,
        image_guidance_scale: float = 1.5,
        text_guidance_scale: float = 5.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Edit source latents with separate image and instruction guidance."""

        if steps <= 0:
            raise ValueError("steps must be positive")
        if source_latents.size(0) != instruction_ids.size(0):
            raise ValueError("source image and instruction batch sizes must match")
        batch = source_latents.size(0)
        x = torch.randn(
            source_latents.shape,
            dtype=source_latents.dtype,
            device=source_latents.device,
            generator=generator,
        )
        empty_ids = torch.zeros_like(instruction_ids)
        empty_mask = torch.zeros_like(instruction_mask, dtype=torch.bool)
        empty_mask[:, 0] = True
        empty_source = torch.zeros_like(source_latents)
        step_size = 1.0 / steps
        for index in range(steps):
            timestep = torch.full(
                (batch,),
                1.0 - index / steps,
                device=x.device,
                dtype=x.dtype,
            )
            both = self(
                x,
                source_latents,
                timestep,
                instruction_ids,
                instruction_mask,
            )
            image_only = self(x, source_latents, timestep, empty_ids, empty_mask)
            unconditional = self(x, empty_source, timestep, empty_ids, empty_mask)
            velocity = unconditional
            velocity = velocity + image_guidance_scale * (image_only - unconditional)
            velocity = velocity + text_guidance_scale * (both - image_only)
            x = x - step_size * velocity
        return x
