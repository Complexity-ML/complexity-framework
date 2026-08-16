"""Structure-aware temporal encoders for articulated pose and wearable IMU."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalized_adjacency(size: int, edges: Iterable[tuple[int, int]]) -> torch.Tensor:
    adjacency = torch.eye(size, dtype=torch.float32)
    for first, second in edges:
        adjacency[first, second] = 1.0
        adjacency[second, first] = 1.0
    inverse_sqrt = adjacency.sum(dim=1).clamp_min(1.0).rsqrt()
    return inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


class _StructuredTemporalBlock(nn.Module):
    """Mix a fixed sensor graph, then model motion along time."""

    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.graph_norm = nn.LayerNorm(width)
        self.graph_projection = nn.Linear(width, width, bias=False)
        self.graph_scale = nn.Parameter(torch.tensor(1e-3))
        self.temporal = nn.Conv2d(
            width,
            width,
            kernel_size=(5, 1),
            padding=(2 * dilation, 0),
            dilation=(dilation, 1),
            groups=width,
            bias=False,
        )
        self.temporal_norm = nn.GroupNorm(1, width)
        self.temporal_projection = nn.Sequential(
            nn.Conv2d(width, 2 * width, 1),
            nn.GELU(),
            nn.Conv2d(2 * width, width, 1),
        )
        self.temporal_scale = nn.Parameter(torch.tensor(1e-3))

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        graph = torch.einsum("ij,btjc->btic", adjacency, values)
        values = values + self.graph_scale * self.graph_projection(
            self.graph_norm(graph)
        )
        temporal = values.permute(0, 3, 1, 2)
        temporal = self.temporal_projection(
            self.temporal_norm(self.temporal(temporal))
        )
        return values + self.temporal_scale * temporal.permute(0, 2, 3, 1)


class _StructuredSequenceTokenizer(nn.Module):
    def __init__(
        self,
        *,
        nodes: int,
        input_features: int,
        hidden_size: int,
        token_count: int,
        edges: Iterable[tuple[int, int]],
    ):
        super().__init__()
        width = max(64, hidden_size // 2)
        self.nodes = nodes
        self.input_features = input_features
        self.token_count = token_count
        self.input_projection = nn.Linear(input_features, width)
        self.node_embedding = nn.Parameter(torch.empty(nodes, width))
        self.blocks = nn.ModuleList(
            _StructuredTemporalBlock(width, dilation)
            for dilation in (1, 2, 4, 8)
        )
        self.node_score = nn.Linear(width, 1)
        self.output_projection = nn.Linear(width, hidden_size)
        self.register_buffer(
            "adjacency",
            _normalized_adjacency(nodes, edges),
            persistent=True,
        )
        nn.init.normal_(self.node_embedding, std=0.02)

    def forward_features(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[-2:] != (
            self.nodes,
            self.input_features,
        ):
            raise ValueError(
                "structured sequence must have shape "
                f"[batch, time, {self.nodes}, {self.input_features}]"
            )
        features = self.input_projection(values)
        features = features + self.node_embedding[None, None, :, :]
        for block in self.blocks:
            features = block(features, self.adjacency)
        weights = self.node_score(features).softmax(dim=2)
        return (features * weights).sum(dim=2)

    def pool_tokens(self, features: torch.Tensor) -> torch.Tensor:
        features = F.adaptive_avg_pool1d(
            features.transpose(1, 2),
            self.token_count,
        ).transpose(1, 2)
        return self.output_projection(features)


class SkeletonGraphTokenizer(_StructuredSequenceTokenizer):
    """Use the 17-joint Human3.6M graph and temporal joint velocities."""

    _EDGES = (
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 4),
        (4, 5),
        (5, 6),
        (0, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (8, 11),
        (11, 12),
        (12, 13),
        (8, 14),
        (14, 15),
        (15, 16),
    )

    def __init__(
        self,
        num_joints: int,
        coordinates: int,
        hidden_size: int,
        token_count: int,
    ):
        if num_joints != 17 or coordinates != 3:
            raise ValueError("structured skeleton encoder requires 17 three-dimensional joints")
        super().__init__(
            nodes=num_joints,
            input_features=2 * coordinates,
            hidden_size=hidden_size,
            token_count=token_count,
            edges=self._EDGES,
        )
        self.coordinates = coordinates

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[-2:] != (self.nodes, self.coordinates):
            raise ValueError(
                f"skeleton must have shape [batch, time, {self.nodes}, {self.coordinates}]"
            )
        velocity = torch.diff(values, dim=1, prepend=values[:, :1])
        features = torch.cat((values, velocity), dim=-1)
        return self.pool_tokens(self.forward_features(features))


class IMUDeviceGraphTokenizer(_StructuredSequenceTokenizer):
    """Preserve the five wearable devices instead of flattening 45 channels."""

    DEVICES = 5
    FEATURES_PER_DEVICE = 9
    _EDGES = ((2, 0), (2, 1), (2, 3), (2, 4))

    def __init__(self, input_features: int, hidden_size: int, token_count: int):
        if input_features != self.DEVICES * self.FEATURES_PER_DEVICE:
            raise ValueError("structured IMU encoder requires five devices x nine features")
        super().__init__(
            nodes=self.DEVICES,
            input_features=3 * self.FEATURES_PER_DEVICE,
            hidden_size=hidden_size,
            token_count=token_count,
            edges=self._EDGES,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.size(-1) != 45:
            raise ValueError("IMU must have shape [batch, time, 45]")
        devices = values.reshape(values.size(0), values.size(1), self.DEVICES, -1)
        centered = devices - devices.mean(dim=1, keepdim=True)
        velocity = torch.diff(devices, dim=1, prepend=devices[:, :1])
        features = torch.cat((devices, centered, velocity), dim=-1)
        return self.pool_tokens(self.forward_features(features))
