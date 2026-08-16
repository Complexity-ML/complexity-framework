"""Hierarchical TR-Hash vision tower for detector architecture v6.

The detector needs genuine features at progressively coarser strides rather
than resampling one final ViT map.  This tower keeps deterministic TR-Hash
experts in every block while adding patch merging, local shifted windows on
the high-resolution stages, a cheap global final stage, and interpolated 2D
position grids for variable-resolution training and inference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from complexity.tr_hash import (
    TRHashBackend,
    TRHashEngine,
    TRHashEngineConfig,
    TRHashPrecision,
    TRHashStrategy,
)


def _build_relative_position_index(window_size: int) -> torch.Tensor:
    """Return the ``[N, N]`` lookup into a ``(2W-1)^2``-row relative-bias table."""

    coords = torch.stack(
        torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")
    )  # 2, W, W
    coords_flatten = coords.flatten(1)  # 2, N
    relative = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
    relative = relative.permute(1, 2, 0).contiguous()  # N, N, 2
    relative[:, :, 0] += window_size - 1
    relative[:, :, 1] += window_size - 1
    relative[:, :, 0] *= 2 * window_size - 1
    return relative.sum(-1)  # N, N


class _SpatialSelfAttention(nn.Module):
    """Windowed attention with a learned relative position bias (Swin-style).

    Absolute additive position embeddings tell a token roughly where it is on
    the feature map but not how it relates to its neighbors inside a local
    window; a per-offset bias table gives windowed attention that structure
    directly, which absolute embeddings alone approximate poorly. Only
    defined for a fixed ``window_size > 0`` -- the un-windowed global stage
    has no fixed window shape to key a relative table on, so it falls back to
    plain attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        window_size: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.window_size = int(window_size)
        if self.window_size > 0:
            table_size = (2 * self.window_size - 1) ** 2
            self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
            self.register_buffer(
                "relative_position_index",
                _build_relative_position_index(self.window_size),
                persistent=False,
            )
        else:
            self.relative_position_bias_table = None
            self.relative_position_index = None

    def _relative_position_bias(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        area = self.window_size**2
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(area, area, -1).permute(2, 0, 1).contiguous()
        return bias.to(dtype=dtype, device=device).unsqueeze(0)

    def forward(
        self,
        tokens: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, width = tokens.shape
        qkv = self.qkv(tokens).view(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        attention_mask = None
        if self.relative_position_bias_table is not None and length == self.window_size**2:
            attention_mask = self._relative_position_bias(query.dtype, query.device).expand(
                batch, -1, -1, -1
            )
            if key_mask is not None:
                additive = torch.zeros(
                    key_mask.shape, dtype=attention_mask.dtype, device=key_mask.device
                ).masked_fill(~key_mask, float("-inf"))
                attention_mask = attention_mask + additive[:, None, None, :]
        elif key_mask is not None:
            attention_mask = key_mask[:, None, None, :]
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(output.transpose(1, 2).reshape(batch, length, width))


def _window_partition(
    values: torch.Tensor,
    window_size: int,
    shift: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Partition ``BHWC`` values into non-wrapping shifted local windows."""

    batch, height, width, channels = values.shape
    top = left = shift
    padded_height = math.ceil((height + top) / window_size) * window_size
    padded_width = math.ceil((width + left) / window_size) * window_size
    bottom = padded_height - height - top
    right = padded_width - width - left
    padded = F.pad(
        values.permute(0, 3, 1, 2),
        (left, right, top, bottom),
    ).permute(0, 2, 3, 1)
    valid = F.pad(
        torch.ones(1, 1, height, width, dtype=torch.bool, device=values.device),
        (left, right, top, bottom),
        value=False,
    ).permute(0, 2, 3, 1)
    rows = padded_height // window_size
    cols = padded_width // window_size
    windows = (
        padded.view(batch, rows, window_size, cols, window_size, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch * rows * cols, window_size**2, channels)
    )
    masks = (
        valid.view(1, rows, window_size, cols, window_size, 1)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(rows * cols, window_size**2)
        .repeat(batch, 1)
    )
    metadata = (batch, height, width, channels, rows, cols, top, left)
    return windows, masks, metadata


def _window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    metadata: tuple[int, ...],
) -> torch.Tensor:
    batch, height, width, channels, rows, cols, top, left = metadata
    padded = (
        windows.view(batch, rows, cols, window_size, window_size, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, rows * window_size, cols * window_size, channels)
    )
    return padded[:, top : top + height, left : left + width]


def _build_patch_stem(in_channels: int, hidden_size: int, patch_size: int) -> nn.Sequential:
    """Progressive stride-2 conv stem, replacing one non-overlapping kernel=patch_size conv.

    A single large-kernel, large-stride conv patchifies the image in one step
    and never lets nearby pixels influence more than one output token. Splitting
    the same total stride into ``log2(patch_size)`` stride-2 stages lets each
    stage mix a small overlapping neighborhood before the next downsample,
    preserving more fine spatial detail -- useful for small-object boxes that a
    single coarse patch can otherwise wash out. The first stage is a regular
    conv (cheap already, since it only reads 3 input channels); every later
    stage is depthwise-separable (depthwise stride-2 + pointwise channel
    projection) so the extra stages cost a fraction of a same-shaped dense
    conv instead of multiplying the stem's parameter count.
    """

    if patch_size <= 0 or (patch_size & (patch_size - 1)):
        raise ValueError("progressive patch stem requires a power-of-two patch_size")
    num_stages = patch_size.bit_length() - 1
    layers: list[nn.Module] = []
    channels = in_channels
    for stage in range(num_stages):
        out_channels = max(hidden_size // (2 ** (num_stages - 1 - stage)), 8)
        if stage == 0:
            layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1))
        else:
            layers.append(
                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=channels)
            )
            layers.append(nn.Conv2d(channels, out_channels, kernel_size=1))
        if stage < num_stages - 1:
            layers.append(nn.GroupNorm(1, out_channels))
            layers.append(nn.GELU())
        channels = out_channels
    return nn.Sequential(*layers)


class HierarchicalTRHashBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
        shared_width: int,
        expert_width: int,
        route_vocab_size: int,
        route_seed: int,
        layer_index: int,
        precision: TRHashPrecision,
        dropout: float,
        layer_norm_eps: float,
        window_size: int,
        shifted: bool,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.shift = self.window_size // 2 if shifted and self.window_size > 1 else 0
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = _SpatialSelfAttention(
            hidden_size, num_heads, dropout, window_size=self.window_size
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = TRHashEngine(
            TRHashEngineConfig(
                hidden_size=hidden_size,
                vocab_size=route_vocab_size,
                num_experts=num_experts,
                top_k=top_k,
                shared_width=shared_width,
                expert_width=expert_width,
                routing_strategy=TRHashStrategy.BALANCED_HASH,
                layer_index=layer_index,
                route_seed=route_seed,
                precision=precision,
                backend=TRHashBackend.AUTO,
            )
        )

    def _attention(self, values: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = values.shape
        normalized = self.norm1(values)
        if self.window_size <= 0 or max(height, width) <= self.window_size:
            tokens = normalized.reshape(batch, height * width, channels)
            return self.attention(tokens).reshape(batch, height, width, channels)
        windows, masks, metadata = _window_partition(
            normalized,
            self.window_size,
            self.shift,
        )
        attended = self.attention(windows, masks)
        return _window_unpartition(attended, self.window_size, metadata)

    def forward(self, values: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        values = values + self._attention(values)
        batch, height, width, channels = values.shape
        tokens = values.reshape(batch, height * width, channels)
        tokens = tokens + self.mlp(self.norm2(tokens), route_ids)
        return tokens.reshape(batch, height, width, channels)


class HierarchicalTRHashVisionTower(nn.Module):
    """Return real P3/P4/P5-like feature maps from successive TR-Hash stages."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_base_grid = config.image_size // config.patch_size
        self.patch_embed = _build_patch_stem(3, config.vision_hidden_size, config.patch_size)
        stage_count = len(config.vision_stage_depths)
        max_grids = [math.ceil(self.max_base_grid / (2**stage)) for stage in range(stage_count)]
        self.position_rows = nn.ParameterList(
            nn.Parameter(
                torch.zeros(
                    1,
                    config.vision_hidden_size,
                    grid,
                )
            )
            for grid in max_grids
        )
        self.position_cols = nn.ParameterList(
            nn.Parameter(
                torch.zeros(
                    1,
                    config.vision_hidden_size,
                    grid,
                )
            )
            for grid in max_grids
        )
        self.downsamples = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    config.vision_hidden_size,
                    config.vision_hidden_size,
                    3,
                    stride=2,
                    padding=1,
                    groups=config.vision_hidden_size,
                ),
                nn.Conv2d(
                    config.vision_hidden_size,
                    config.vision_hidden_size,
                    1,
                ),
            )
            for _ in range(stage_count - 1)
        )
        precision = TRHashPrecision(config.vision_precision)
        stages = []
        layer_index = 0
        for stage_index, (depth, max_grid) in enumerate(zip(config.vision_stage_depths, max_grids)):
            window_size = 0 if stage_index == stage_count - 1 else config.vision_window_size
            blocks = []
            for block_index in range(depth):
                blocks.append(
                    HierarchicalTRHashBlock(
                        hidden_size=config.vision_hidden_size,
                        num_heads=config.vision_heads,
                        num_experts=config.vision_num_experts,
                        top_k=config.vision_top_k,
                        shared_width=config.vision_shared_width,
                        expert_width=config.vision_expert_width,
                        route_vocab_size=max_grid**2,
                        route_seed=config.route_seed,
                        layer_index=layer_index,
                        precision=precision,
                        dropout=config.dropout,
                        layer_norm_eps=config.layer_norm_eps,
                        window_size=window_size,
                        shifted=block_index % 2 == 1,
                    )
                )
                layer_index += 1
            stages.append(nn.ModuleList(blocks))
        self.stages = nn.ModuleList(stages)
        self.stage_norms = nn.ModuleList(
            nn.LayerNorm(config.vision_hidden_size, eps=config.layer_norm_eps)
            for _ in range(stage_count)
        )
        for position in (*self.position_rows, *self.position_cols):
            nn.init.normal_(position, std=0.02)

    @property
    def blocks(self) -> list[HierarchicalTRHashBlock]:
        """Flattened block view used by backend diagnostics and LR grouping."""

        return [block for stage in self.stages for block in stage]

    def _route_ids(
        self,
        stage_index: int,
        height: int,
        width: int,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        max_grid = self.position_rows[stage_index].shape[-1]
        if height > max_grid or width > max_grid:
            raise ValueError(
                "input resolution exceeds the configured maximum vision grid "
                f"at stage {stage_index}: {(height, width)} > {(max_grid, max_grid)}"
            )
        rows = torch.arange(height, device=device)[:, None]
        cols = torch.arange(width, device=device)[None, :]
        route_ids = (rows * max_grid + cols).reshape(1, height * width)
        return route_ids.expand(batch, -1)

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        if pixel_values.ndim != 4 or pixel_values.size(1) != 3:
            raise ValueError("pixel_values must have shape [batch, 3, height, width]")
        if (
            pixel_values.size(-2) % self.config.patch_size
            or pixel_values.size(-1) % self.config.patch_size
        ):
            raise ValueError("input height and width must be divisible by patch_size")
        if max(pixel_values.shape[-2:]) > self.config.image_size:
            raise ValueError("input resolution exceeds configured image_size maximum")

        feature_map = self.patch_embed(pixel_values)
        outputs = []
        for stage_index, (blocks, norm) in enumerate(zip(self.stages, self.stage_norms)):
            if stage_index:
                feature_map = self.downsamples[stage_index - 1](feature_map)
            row_position = F.interpolate(
                self.position_rows[stage_index],
                size=feature_map.size(-2),
                mode="linear",
                align_corners=False,
            )[..., :, None]
            col_position = F.interpolate(
                self.position_cols[stage_index],
                size=feature_map.size(-1),
                mode="linear",
                align_corners=False,
            )[..., None, :]
            position = (row_position + col_position).to(feature_map.dtype)
            values = (feature_map + position).permute(0, 2, 3, 1)
            route_ids = self._route_ids(
                stage_index,
                values.size(1),
                values.size(2),
                values.size(0),
                values.device,
            )
            for block in blocks:
                values = block(values, route_ids)
            values = norm(values)
            feature_map = values.permute(0, 3, 1, 2).contiguous()
            outputs.append(feature_map)
        return outputs


class HierarchicalTRHashVisionClassifier(nn.Module):
    """Classification pretraining wrapper for the detector's exact v6 tower."""

    def __init__(self, config, num_classes: int):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.tower = HierarchicalTRHashVisionTower(config)
        self.head_norm = nn.LayerNorm(
            config.vision_hidden_size,
            eps=config.layer_norm_eps,
        )
        self.head = nn.Linear(config.vision_hidden_size, num_classes)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.tower(pixel_values)[-1]
        pooled = self.head_norm(features.mean(dim=(-2, -1)))
        logits = self.head(pooled)
        output = {"logits": logits, "pooled_features": pooled}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits, labels)
        return output
