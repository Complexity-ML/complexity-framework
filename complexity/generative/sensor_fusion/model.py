"""Eight-expert TR-Hash fusion model for multimodal robot perception."""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from complexity.generative.detection.config import TRHashDetectorConfig
from complexity.generative.detection.hierarchical_tower import (
    HierarchicalTRHashVisionTower,
)
from complexity.tr_hash import (
    TRHashBackend,
    TRHashEngine,
    TRHashEngineConfig,
    TRHashPrecision,
    TRHashStrategy,
)

from .config import SENSOR_MODALITIES, TRHashSensorFusionConfig
from .encoders import SequenceTokenizerV2
from .structured_encoders import IMUDeviceGraphTokenizer, SkeletonGraphTokenizer


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return values

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


def gradient_reverse(values: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Keep the forward value and reverse its encoder gradient."""

    return _GradientReverse.apply(values, scale)


class _FusionSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        batch, length, width = tokens.shape
        qkv = self.qkv(tokens).view(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=token_mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(output.transpose(1, 2).reshape(batch, length, width))


class TRHashSensorFusionBlock(nn.Module):
    def __init__(self, config: TRHashSensorFusionConfig, layer_index: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = _FusionSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_dropout,
        )
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = TRHashEngine(
            TRHashEngineConfig(
                hidden_size=config.hidden_size,
                vocab_size=config.route_vocab_size,
                num_experts=config.num_experts,
                top_k=config.top_k,
                shared_width=config.shared_width,
                expert_width=config.expert_width,
                routing_strategy=TRHashStrategy.BALANCED_HASH,
                layer_index=layer_index,
                route_seed=config.route_seed,
                precision=TRHashPrecision(config.precision),
                backend=TRHashBackend.AUTO,
            )
        )

    def forward(
        self,
        tokens: torch.Tensor,
        route_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = tokens + self.attention(self.norm1(tokens), token_mask)
        tokens = tokens + self.mlp(self.norm2(tokens), route_ids)
        return tokens


class _ClassModalityHashGate(nn.Module):
    """Score every class-modality pair through fixed TR-HASH routes."""

    def __init__(self, config: TRHashSensorFusionConfig):
        super().__init__()
        num_modalities = len(SENSOR_MODALITIES)
        self.num_classes = config.num_classes
        self.num_modalities = num_modalities
        self.class_embedding = nn.Parameter(
            torch.empty(config.num_classes, config.hidden_size)
        )
        self.modality_embedding = nn.Parameter(
            torch.empty(num_modalities, config.hidden_size)
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = TRHashEngine(
            TRHashEngineConfig(
                hidden_size=config.hidden_size,
                vocab_size=num_modalities * config.num_classes,
                num_experts=config.num_experts,
                top_k=config.top_k,
                shared_width=config.shared_width,
                expert_width=config.expert_width,
                routing_strategy=TRHashStrategy.BALANCED_HASH,
                layer_index=config.num_hidden_layers + config.vision_layers,
                route_seed=config.route_seed,
                precision=TRHashPrecision(config.precision),
                backend=TRHashBackend.AUTO,
            )
        )
        self.score = nn.Linear(config.hidden_size, 1)
        self.register_buffer(
            "route_ids",
            torch.arange(num_modalities * config.num_classes, dtype=torch.long),
            persistent=True,
        )
        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expected = (self.num_modalities, self.class_embedding.size(1))
        if features.ndim != 3 or tuple(features.shape[1:]) != expected:
            raise ValueError(
                "class-modality gate features must have shape "
                f"[batch, {self.num_modalities}, {self.class_embedding.size(1)}]"
            )
        batch = features.size(0)
        tokens = (
            features[:, :, None, :]
            + self.modality_embedding[None, :, None, :]
            + self.class_embedding[None, None, :, :]
        ).reshape(batch, self.num_modalities * self.num_classes, -1)
        route_ids = self.route_ids[None, :].expand(batch, -1)
        tokens = tokens + self.mlp(self.norm(tokens), route_ids)
        return self.score(tokens).reshape(batch, self.num_modalities, self.num_classes)


class _ClassHashResidualHead(nn.Module):
    """Add class-specialized capacity without replacing the shared head."""

    def __init__(self, config: TRHashSensorFusionConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.class_embedding = nn.Parameter(
            torch.empty(config.num_classes, config.hidden_size)
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = TRHashEngine(
            TRHashEngineConfig(
                hidden_size=config.hidden_size,
                vocab_size=config.num_classes,
                num_experts=config.num_experts,
                top_k=config.top_k,
                shared_width=config.class_hash_shared_width,
                expert_width=config.class_hash_expert_width,
                routing_strategy=TRHashStrategy.BALANCED_HASH,
                layer_index=config.num_hidden_layers + config.vision_layers + 1,
                route_seed=config.route_seed ^ 0xC1A55,
                precision=TRHashPrecision(config.precision),
                backend=TRHashBackend.AUTO,
            )
        )
        self.score = nn.Linear(config.hidden_size, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(config.class_hash_initial_scale))
        self.register_buffer(
            "route_ids",
            torch.arange(config.num_classes, dtype=torch.long),
            persistent=True,
        )
        nn.init.normal_(self.class_embedding, std=0.02)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.ndim != 2:
            raise ValueError("class-hash head expects [batch, hidden] features")
        batch = pooled.size(0)
        tokens = pooled[:, None, :] + self.class_embedding[None, :, :]
        route_ids = self.route_ids[None, :].expand(batch, -1)
        tokens = tokens + self.mlp(self.norm(tokens), route_ids)
        return self.scale * self.score(tokens).squeeze(-1)


class _ResidualModalityAdapter(nn.Module):
    """Give each visual sensor a cheap specialization path."""

    def __init__(self, hidden_size: int):
        super().__init__()
        inner = max(32, hidden_size // 4)
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, inner, bias=False)
        self.up = nn.Linear(inner, hidden_size, bias=False)
        self.scale = nn.Parameter(torch.full((hidden_size,), 1e-3))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        adapted = self.up(F.gelu(self.down(self.norm(values))))
        return values + adapted * self.scale


class _VisualMotionTokenizer(nn.Module):
    """Encode frame differences into the same token lattice as the tower."""

    def __init__(
        self,
        hidden_size: int,
        token_grid: tuple[int, int, int],
    ):
        super().__init__()
        width = max(32, hidden_size // 8)
        self.token_grid = token_grid
        self.stem = nn.Sequential(
            nn.Conv3d(
                3,
                width,
                kernel_size=(3, 5, 5),
                stride=(1, 4, 4),
                padding=(1, 2, 2),
                bias=False,
            ),
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv3d(
                width,
                width,
                kernel_size=3,
                padding=1,
                groups=width,
                bias=False,
            ),
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv3d(width, hidden_size, kernel_size=1, bias=False),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5 or values.size(1) != 3:
            raise ValueError("visual motion input must have shape [batch, 3, time, H, W]")
        differences = torch.zeros_like(values)
        differences[:, :, 1:] = values[:, :, 1:] - values[:, :, :-1]
        features = self.stem(differences)
        features = F.adaptive_avg_pool3d(features, self.token_grid)
        return features.flatten(2).transpose(1, 2)


class TRHashSensorFusionClassifier(nn.Module):
    """Fuse six sensor modalities and predict one of ``num_classes`` states.

    Routing IDs encode ``(modality, local token position)``. The assignment to
    experts is hash-based and fixed per layer; input content never selects the
    expert. A sample-level modality mask allows clips with absent sensors.

    This is the perception front-end of a robot stack: sensors -> encoders ->
    TR-Hash fusion -> fused state -> classification head. Downstream policy,
    trajectory, and value heads can consume ``pooled_features`` without
    retraining this path.
    """

    def __init__(self, config: Optional[TRHashSensorFusionConfig] = None):
        super().__init__()
        self.config = config or TRHashSensorFusionConfig()
        config = self.config
        depth_channels, ir_channels, thermal_channels = config.visual_channels
        del depth_channels, ir_channels, thermal_channels

        # The three visual streams (depth, IR, thermal) share one compact
        # TR-Hash MoE vision tower; modality embeddings keep their identities
        # distinct after the shared visual representation is formed.
        tower_config = TRHashDetectorConfig(
            image_size=config.vision_image_size,
            patch_size=config.vision_patch_size,
            vision_hidden_size=config.vision_hidden_size,
            vision_layers=config.vision_layers,
            vision_heads=config.vision_heads,
            vision_num_experts=config.vision_num_experts,
            vision_top_k=config.vision_top_k,
            vision_shared_width=config.vision_shared_width,
            vision_expert_width=config.vision_expert_width,
            vision_stage_depths=config.vision_stage_depths,
            vision_window_size=config.vision_window_size,
            vision_precision=config.precision,
            route_seed=config.route_seed,
            num_classes=config.num_classes,
            scale_factors=tuple(2**index for index in range(len(config.vision_stage_depths))),
        )
        self.vision_tower = HierarchicalTRHashVisionTower(tower_config)
        self.visual_projection = nn.Linear(
            config.vision_hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.visual_motion_encoder = _VisualMotionTokenizer(
            config.hidden_size,
            config.visual_token_grid,
        )
        self.visual_motion_scale = nn.Parameter(torch.tensor(0.1))

        def sequence_encoder(features: int) -> nn.Module:
            if features == config.imu_features:
                return IMUDeviceGraphTokenizer(
                    features,
                    config.hidden_size,
                    config.sequence_tokens,
                )
            return SequenceTokenizerV2(
                features,
                config.hidden_size,
                config.sequence_tokens,
            )

        skeleton_encoder = SkeletonGraphTokenizer(
            config.skeleton_joints,
            config.skeleton_coordinates,
            config.hidden_size,
            config.sequence_tokens,
        )
        self.encoders = nn.ModuleDict(
            {
                "depth": nn.Identity(),
                "ir": nn.Identity(),
                "thermal": nn.Identity(),
                "imu": sequence_encoder(config.imu_features),
                "radar": sequence_encoder(config.radar_features),
                "skeleton": skeleton_encoder,
            }
        )
        self.encoder_norms = nn.ModuleDict(
            {name: nn.LayerNorm(config.hidden_size) for name in SENSOR_MODALITIES}
        )
        self.visual_adapters = nn.ModuleDict(
            {
                name: _ResidualModalityAdapter(config.hidden_size)
                for name in ("depth", "ir", "thermal")
            }
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.modality_embedding = nn.Parameter(
            torch.zeros(len(SENSOR_MODALITIES), config.hidden_size)
        )
        self.position_embeddings = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(1, count, config.hidden_size))
                for name, count in zip(SENSOR_MODALITIES, config.modality_token_counts)
            }
        )
        self.missing_tokens = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(1, count, config.hidden_size))
                for name, count in zip(SENSOR_MODALITIES, config.modality_token_counts)
            }
        )
        self.blocks = nn.ModuleList(
            TRHashSensorFusionBlock(config, layer_index=index)
            for index in range(config.num_hidden_layers)
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size, config.num_classes),
        )
        self.class_hash_head = _ClassHashResidualHead(config)
        self.subject_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 18),
        )
        self.modality_heads = nn.ModuleDict(
            {
                name: nn.Linear(config.hidden_size, config.num_classes)
                for name in SENSOR_MODALITIES
            }
        )
        self.class_modality_gate = _ClassModalityHashGate(config)
        self.register_buffer(
            "route_ids",
            torch.arange(config.route_vocab_size, dtype=torch.long),
            persistent=True,
        )
        nn.init.normal_(self.modality_embedding, std=0.02)
        for parameter in (*self.position_embeddings.values(), *self.missing_tokens.values()):
            nn.init.normal_(parameter, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def _batch_size(self, inputs: Mapping[str, torch.Tensor]) -> int:
        unknown = sorted(set(inputs) - set(SENSOR_MODALITIES))
        if unknown:
            raise ValueError(f"unknown sensor modalities: {unknown}")
        if not inputs:
            raise ValueError("at least one sensor modality is required")
        batch_sizes = {int(values.size(0)) for values in inputs.values()}
        if len(batch_sizes) != 1:
            raise ValueError("all sensor modalities must use the same batch size")
        return batch_sizes.pop()

    def _presence(
        self,
        name: str,
        batch_size: int,
        device: torch.device,
        modality_mask: Optional[Mapping[str, torch.Tensor]],
        provided: bool,
    ) -> torch.Tensor:
        if modality_mask is None or name not in modality_mask:
            return torch.full((batch_size,), provided, dtype=torch.bool, device=device)
        mask = modality_mask[name].to(device=device, dtype=torch.bool)
        if mask.shape != (batch_size,):
            raise ValueError(f"modality_mask[{name!r}] must have shape [batch]")
        if not provided and mask.any():
            raise ValueError(f"modality_mask marks absent modality {name!r} as present")
        return mask

    def _encode_visual(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5 or values.size(1) not in (1, 3):
            raise ValueError("visual clips must have shape [batch, 1|3, time, height, width]")
        batch, channels, frames, height, width = values.shape
        if channels == 1:
            values = values.expand(-1, 3, -1, -1, -1)
        frame_batch = values.permute(0, 2, 1, 3, 4).reshape(
            batch * frames,
            3,
            height,
            width,
        )
        features = self.vision_tower(frame_batch)[-1]
        _, feature_width, feature_height, feature_columns = features.shape
        features = features.reshape(
            batch,
            frames,
            feature_width,
            feature_height,
            feature_columns,
        ).permute(0, 2, 1, 3, 4)
        features = F.adaptive_avg_pool3d(features, self.config.visual_token_grid)
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.visual_projection(tokens)
        tokens = tokens + self.visual_motion_scale * self.visual_motion_encoder(values)
        return tokens

    def tokenize(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = self._batch_size(inputs)
        first = next(iter(inputs.values()))
        device = first.device
        chunks = [self.cls_token.expand(batch_size, -1, -1)]
        masks = [torch.ones(batch_size, 1, dtype=torch.bool, device=device)]
        presence_by_name = {
            name: self._presence(
                name,
                batch_size,
                device,
                modality_mask,
                name in inputs,
            )
            for name in SENSOR_MODALITIES
        }
        encoded_by_name: dict[str, torch.Tensor] = {}
        for name in SENSOR_MODALITIES:
            provided = name in inputs
            presence = presence_by_name[name]
            missing = self.missing_tokens[name].expand(batch_size, -1, -1)
            if provided:
                values = inputs[name]
                if values.device != device:
                    raise ValueError("all sensor modalities must be on the same device")
                if name in {"depth", "ir", "thermal"}:
                    encoded = self._encode_visual(values)
                else:
                    encoded = self.encoders[name](values)
                encoded = self.encoder_norms[name](encoded)
                if name in self.visual_adapters:
                    encoded = self.visual_adapters[name](encoded)
                encoded = torch.where(presence[:, None, None], encoded, missing)
            else:
                encoded = missing
            encoded_by_name[name] = encoded
        for modality_index, name in enumerate(SENSOR_MODALITIES):
            presence = presence_by_name[name]
            encoded = encoded_by_name[name]
            encoded = (
                encoded
                + self.position_embeddings[name]
                + self.modality_embedding[modality_index][None, None, :]
            )
            chunks.append(encoded)
            masks.append(presence[:, None].expand(-1, encoded.size(1)))
        token_mask = torch.cat(masks, dim=1)
        if not token_mask.any(dim=1).all():
            raise ValueError("every sample must contain at least one sensor modality")
        return torch.cat(chunks, dim=1), token_mask

    def _late_fusion(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        modality_logits = []
        modality_features = []
        presence = []
        offset = 1  # skip the fusion CLS token
        for name, count in zip(SENSOR_MODALITIES, self.config.modality_token_counts):
            end = offset + count
            mask = token_mask[:, offset:end]
            weights = mask.to(tokens.dtype).unsqueeze(-1)
            features = (tokens[:, offset:end] * weights).sum(dim=1)
            features = features / weights.sum(dim=1).clamp_min(1.0)
            modality_features.append(features)
            modality_logits.append(self.modality_heads[name](features))
            presence.append(mask.any(dim=1))
            offset = end
        logits = torch.stack(modality_logits, dim=1)
        available = torch.stack(presence, dim=1)
        scores = self.class_modality_gate(torch.stack(modality_features, dim=1))
        score_mask = available if scores.ndim == 2 else available.unsqueeze(-1)
        scores = scores.masked_fill(~score_mask, -1.0e4)
        gates = scores.softmax(dim=1)
        weighted_logits = logits * (gates.unsqueeze(-1) if gates.ndim == 2 else gates)
        return weighted_logits.sum(dim=1), logits, gates

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        *,
        modality_mask: Optional[Mapping[str, torch.Tensor]] = None,
        subject_adversarial_scale: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        tokens, token_mask = self.tokenize(inputs, modality_mask)
        route_ids = self.route_ids[None, :].expand(tokens.size(0), -1)
        for block in self.blocks:
            tokens = block(tokens, route_ids, token_mask)
        tokens = self.norm(tokens)
        pooled = tokens[:, 0]
        fused_logits = self.head(pooled)
        class_hash_logits = self.class_hash_head(pooled)
        fused_logits = fused_logits + class_hash_logits
        output = {
            "fused_logits": fused_logits,
            "pooled_features": pooled,
            "token_mask": token_mask,
            "class_hash_logits": class_hash_logits,
            "subject_logits": self.subject_head(
                gradient_reverse(pooled, subject_adversarial_scale)
            ),
        }
        late_logits, modality_logits, modality_weights = self._late_fusion(tokens, token_mask)
        weight = self.config.late_fusion_weight
        logits = (fused_logits + weight * late_logits) / (1.0 + weight)
        output.update(
            {
                "logits": logits,
                "late_fusion_logits": late_logits,
                "modality_logits": modality_logits,
                "modality_weights": modality_weights,
            }
        )
        if labels is not None:
            if labels.shape != (tokens.size(0),):
                raise ValueError("labels must have shape [batch]")
            output["loss"] = F.cross_entropy(logits, labels.long())
        return output

    def num_parameters(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (value for value in parameters if value.requires_grad)
        return sum(value.numel() for value in parameters)

    @property
    def tr_hash_blocks(self) -> list[nn.Module]:
        blocks: list[nn.Module] = list(self.blocks)
        blocks.extend(self.vision_tower.blocks)
        blocks.append(self.class_modality_gate)
        blocks.append(self.class_hash_head)
        return blocks

    @property
    def fp32_size_bytes(self) -> int:
        return self.num_parameters() * 4
