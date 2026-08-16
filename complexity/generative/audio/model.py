"""TR-Hash speech-to-text (ASR) and text-to-speech (TTS) models."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from complexity.generative.image.model import (
    _TextEncoder,
    _TRHashDiTBlock,
    _modulate,
    _timestep_embedding,
)
from complexity.models import ComplexityModel

from .config import TRHashAudioConfig, TRHashSpeechToTextConfig
from .encoder import AudioEncoder


class TokenResampler(nn.Module):
    """Compress a variable-length feature sequence into a fixed-size prefix.

    Modality-agnostic Perceiver-style cross-attention pooler — used for both
    the image and audio prefixes.
    """

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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        queries = self.queries.expand(features.size(0), -1, -1)
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            self.context_norm(features),
            self.context_norm(features),
            need_weights=False,
        )
        hidden = queries + attended
        return hidden + self.mlp(self.output_norm(hidden))


class TRHashSpeechToText(nn.Module):
    """Transcribe a waveform (plus optional text prompt) to text directly.

    The audio encoder and resampler produce a fixed prefix in the decoder's
    embedding space, exactly as ``TRHashImageTextToText`` does for images —
    audio frames get deterministic synthetic route IDs so they dispatch
    through the same token-ID-routed TR-Hash decoder as real text tokens.
    """

    def __init__(
        self,
        config: Optional[TRHashSpeechToTextConfig] = None,
        *,
        decoder: Optional[ComplexityModel] = None,
    ):
        super().__init__()
        self.config = config or TRHashSpeechToTextConfig()
        self.audio_encoder = AudioEncoder(self.config.audio_encoder_config())
        self.resampler = TokenResampler(
            self.config.audio_hidden_size,
            self.config.audio_heads,
            self.config.num_audio_tokens,
        )
        self.audio_projection = nn.Sequential(
            nn.LayerNorm(self.config.audio_hidden_size),
            nn.Linear(self.config.audio_hidden_size, self.config.hidden_size),
        )
        self.decoder = decoder or ComplexityModel(self.config.decoder_config())
        if self.decoder.config.hidden_size != self.config.hidden_size:
            raise ValueError("decoder hidden size does not match ASR configuration")
        if self.decoder.config.vocab_size != self.config.vocab_size:
            raise ValueError("decoder vocabulary does not match ASR configuration")

        positions = torch.arange(self.config.num_audio_tokens, dtype=torch.long)
        audio_route_ids = (
            positions * 2_654_435_761 + int(self.config.route_seed)
        ) % self.config.vocab_size
        self.register_buffer("audio_route_ids", audio_route_ids, persistent=True)

    def encode_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.audio_encoder(waveform)
        return self.audio_projection(self.resampler(features))

    def prepare_multimodal_inputs(
        self,
        waveform: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, text_seq]")
        audio = self.encode_audio(waveform)
        if audio.size(0) != input_ids.size(0):
            raise ValueError("audio and text batch sizes must match")
        text = self.decoder.embed_tokens(input_ids)
        embeddings = torch.cat((audio, text), dim=1)
        audio_routes = self.audio_route_ids.unsqueeze(0).expand(input_ids.size(0), -1)
        routing_ids = torch.cat((audio_routes, input_ids), dim=1)
        return embeddings, routing_ids

    def forward(
        self,
        waveform: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        embeddings, routing_ids = self.prepare_multimodal_inputs(waveform, input_ids)
        decoder_output = self.decoder(
            inputs_embeds=embeddings,
            routing_ids=routing_ids,
            return_hidden_states=return_hidden_states,
        )
        audio_tokens = self.config.num_audio_tokens
        text_logits = decoder_output["logits"][:, audio_tokens:, :]
        output: Dict[str, Any] = {
            "logits": text_logits,
            "past_key_values": decoder_output["past_key_values"],
            "audio_tokens": embeddings[:, :audio_tokens],
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


class TRHashTextToSpeech(nn.Module):
    """Text-conditioned latent rectified-flow model over log-mel spectrogram frames.

    Same architecture family as ``TRHashTextToImage`` (same DiT block, same
    rectified-flow objective, same deterministic route-ID scheme keyed on
    timestep bucket + position) — 1D mel frames instead of a 2D patch grid,
    and no separate learned codec: the mel spectrogram itself is the latent.
    """

    def __init__(self, config: Optional[TRHashAudioConfig] = None):
        super().__init__()
        self.config = config or TRHashAudioConfig()
        self.gradient_checkpointing = False
        config = self.config
        self.text_encoder = _TextEncoder(config)
        self.latent_in = nn.Linear(config.latent_patch_features, config.hidden_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.audio_token_count, config.hidden_size)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
        )
        self.blocks = nn.ModuleList(
            _TRHashDiTBlock(config, layer_index=index) for index in range(config.num_layers)
        )
        self.final_condition = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.latent_out = nn.Linear(config.hidden_size, config.latent_patch_features)
        nn.init.normal_(self.position_embedding, std=0.02)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _patchify(self, mel: torch.Tensor) -> torch.Tensor:
        config = self.config
        expected = (config.n_mels, config.max_audio_frames)
        if tuple(mel.shape[1:]) != expected:
            raise ValueError(f"expected mel shape [batch, {expected}], got {tuple(mel.shape)}")
        patch = config.frame_patch_size
        return (
            mel.unfold(2, patch, patch)
            .permute(0, 2, 1, 3)
            .reshape(mel.size(0), config.audio_token_count, config.latent_patch_features)
        )

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        config = self.config
        batch = patches.size(0)
        return (
            patches.view(batch, config.audio_token_count, config.n_mels, config.frame_patch_size)
            .permute(0, 2, 1, 3)
            .reshape(batch, config.n_mels, config.max_audio_frames)
        )

    def build_audio_route_ids(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must be [batch]")
        buckets = (timesteps.float().clamp(0, 1) * self.config.time_buckets).long()
        buckets = buckets.clamp_max(self.config.time_buckets - 1)
        positions = torch.arange(self.config.audio_token_count, device=timesteps.device)
        return buckets[:, None] * self.config.audio_token_count + positions[None]

    def forward(
        self,
        mel: torch.Tensor,
        timesteps: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
    ) -> torch.Tensor:
        if timesteps.shape != (mel.size(0),):
            raise ValueError("one timestep is required per mel sample")
        text, text_mask = self.text_encoder(caption_ids, caption_mask)
        masked = text * text_mask.unsqueeze(-1).to(text.dtype)
        pooled_text = masked.sum(dim=1) / text_mask.sum(dim=1, keepdim=True).clamp_min(1)
        time = self.time_mlp(
            _timestep_embedding(timesteps, self.config.hidden_size).to(mel.dtype)
        )
        condition = pooled_text + time
        x = self.latent_in(self._patchify(mel)) + self.position_embedding
        route_ids = self.build_audio_route_ids(timesteps)
        for block in self.blocks:
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
        clean_mel: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        batch = clean_mel.size(0)
        noise = torch.randn(
            clean_mel.shape,
            dtype=clean_mel.dtype,
            device=clean_mel.device,
            generator=generator,
        )
        timesteps = torch.rand(
            batch, dtype=clean_mel.dtype, device=clean_mel.device, generator=generator
        )
        if self.training and self.config.caption_dropout:
            dropped = torch.rand(batch, device=clean_mel.device, generator=generator)
            caption_mask = caption_mask.clone()
            caption_mask[dropped < self.config.caption_dropout] = False
        shape = (batch,) + (1,) * (clean_mel.ndim - 1)
        t = timesteps.view(shape)
        noisy = (1.0 - t) * clean_mel + t * noise
        target_velocity = noise - clean_mel
        prediction = self(noisy, timesteps, caption_ids, caption_mask)
        return F.mse_loss(prediction.float(), target_velocity.float())

    @torch.no_grad()
    def sample(
        self,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
        *,
        steps: int = 30,
        guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")
        batch = caption_ids.size(0)
        config = self.config
        x = torch.randn(
            batch,
            config.n_mels,
            config.max_audio_frames,
            device=caption_ids.device,
            dtype=self.position_embedding.dtype,
            generator=generator,
        )
        empty_ids = torch.zeros_like(caption_ids)
        empty_mask = torch.zeros_like(caption_mask, dtype=torch.bool)
        empty_mask[:, 0] = True
        step_size = 1.0 / steps
        for index in range(steps):
            timestep = torch.full(
                (batch,), 1.0 - index / steps, device=x.device, dtype=x.dtype
            )
            conditional = self(x, timestep, caption_ids, caption_mask)
            if guidance_scale == 1.0:
                velocity = conditional
            else:
                unconditional = self(x, timestep, empty_ids, empty_mask)
                velocity = unconditional + guidance_scale * (conditional - unconditional)
            x = x - step_size * velocity
        return x
