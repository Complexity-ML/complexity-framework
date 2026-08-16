"""Direct image-and-text to text model with a TR-Hash causal decoder."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from complexity.models import ComplexityModel

from .config import TRHashVisionLanguageConfig
from .vision_tower import TRHashVisionTower


class VisualTokenResampler(nn.Module):
    """Compress variable image patch features into a fixed visual prefix."""

    def __init__(self, hidden_size: int, num_heads: int, num_tokens: int):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_tokens, hidden_size))
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        queries = self.queries.expand(image_features.size(0), -1, -1)
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            self.context_norm(image_features),
            self.context_norm(image_features),
            need_weights=False,
        )
        hidden = queries + attended
        return hidden + self.mlp(self.output_norm(hidden))


class TRHashImageTextToText(nn.Module):
    """Use an image and a text prompt to predict a text response directly.

    The vision tower and resampler produce a fixed prefix in the decoder's
    embedding space. Prompt/response embeddings follow that prefix in the same
    causal sequence. There is no intermediate caption-generation stage.
    """

    def __init__(
        self,
        config: Optional[TRHashVisionLanguageConfig] = None,
        *,
        decoder: Optional[ComplexityModel] = None,
    ):
        super().__init__()
        self.config = config or TRHashVisionLanguageConfig()
        self.vision_tower = TRHashVisionTower(self.config.vision_tower_config())
        self.resampler = VisualTokenResampler(
            self.config.vision_hidden_size,
            self.config.vision_heads,
            self.config.num_visual_tokens,
        )
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(self.config.vision_hidden_size),
            nn.Linear(self.config.vision_hidden_size, self.config.hidden_size),
        )
        self.decoder = decoder or ComplexityModel(self.config.decoder_config())
        if self.decoder.config.hidden_size != self.config.hidden_size:
            raise ValueError("decoder hidden size does not match VLM configuration")
        if self.decoder.config.vocab_size != self.config.vocab_size:
            raise ValueError("decoder vocabulary does not match VLM configuration")

        positions = torch.arange(self.config.num_visual_tokens, dtype=torch.long)
        visual_route_ids = (
            positions * 2_654_435_761 + int(self.config.route_seed)
        ) % self.config.vocab_size
        self.register_buffer("visual_route_ids", visual_route_ids, persistent=True)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.vision_tower(pixel_values)
        return self.visual_projection(self.resampler(features))

    def prepare_multimodal_inputs(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, text_seq]")
        visual = self.encode_images(pixel_values)
        if visual.size(0) != input_ids.size(0):
            raise ValueError("image and text batch sizes must match")
        text = self.decoder.embed_tokens(input_ids)
        embeddings = torch.cat((visual, text), dim=1)
        image_routes = self.visual_route_ids.unsqueeze(0).expand(input_ids.size(0), -1)
        routing_ids = torch.cat((image_routes, input_ids), dim=1)
        return embeddings, routing_ids

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        embeddings, routing_ids = self.prepare_multimodal_inputs(pixel_values, input_ids)
        decoder_output = self.decoder(
            inputs_embeds=embeddings,
            routing_ids=routing_ids,
            return_hidden_states=return_hidden_states,
        )
        visual_tokens = self.config.num_visual_tokens
        text_logits = decoder_output["logits"][:, visual_tokens:, :]
        output: Dict[str, Any] = {
            "logits": text_logits,
            "past_key_values": decoder_output["past_key_values"],
            "visual_tokens": embeddings[:, :visual_tokens],
        }
        if return_hidden_states:
            output["hidden_states"] = decoder_output["hidden_states"]
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids")
            output["loss"] = F.cross_entropy(
                text_logits[:, :-1].reshape(-1, text_logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return output

    def num_parameters(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)
