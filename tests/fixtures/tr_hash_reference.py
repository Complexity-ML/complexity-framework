"""Independent, fixed-size PyTorch oracle for the canonical TR-HASH equations."""

import torch
from torch import nn
import torch.nn.functional as F


def linear(input_width, output_width, residual=False):
    layer = nn.Linear(input_width, output_width, bias=False)
    nn.init.normal_(layer.weight, std=0.01 if residual else 0.02)
    return layer


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate = linear(64, width)
        self.up = linear(64, width)
        self.down = linear(width, 64, True)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def mlp_routes(ids, layer):
    zeros = torch.zeros(ids.shape[0], 2, dtype=torch.long, device=ids.device)
    history = torch.cat((zeros, ids), 1)
    u = (history[:, 1:-1] + 5 * history[:, :-2]) % 9
    if layer == 1:
        a = ((ids + 1) % 4) ^ (2 * ((u % 3) % 2))
        stride = 1 + (ids // 4 + 2 * (u // 3) + 1) % 3
    else:
        a = ids % 4
        stride = 1 + (ids // 4 + u % 3) % 3
    return torch.stack((a, (a + stride) % 4), -1)


class RoutedMLP(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.shared = SwiGLU(128)
        self.experts = nn.ModuleList(SwiGLU(16) for _ in range(4))
        self.layer = layer

    def forward(self, x, ids):
        routes = mlp_routes(ids, self.layer)
        routed = torch.zeros_like(x)
        for index, expert in enumerate(self.experts):
            primary = (routes[..., :1] == index).to(x.dtype)
            secondary = (routes[..., 1:] == index).to(x.dtype)
            routed = routed + (0.5 * primary + 0.5 * secondary) * expert(x)
        return self.shared(x) + 2 * routed


def rope(x):
    length, width = x.shape[-2:]
    position = torch.arange(length, device=x.device)[:, None]
    frequency = 10000.0 ** (-torch.arange(0, width, 2, device=x.device) / width)
    angle = torch.cat((position * frequency, position * frequency), -1)
    left, right = x.chunk(2, -1)
    return x * angle.cos() + torch.cat((-right, left), -1) * angle.sin()


class Attention(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.q = linear(64, 64)
        self.k = linear(64, 16)
        self.v = linear(64, 16)
        self.o = linear(64, 64, True)
        self.norm_q = nn.RMSNorm(16, eps=1e-6)
        self.norm_k = nn.RMSNorm(16, eps=1e-6)

    def forward(self, x, ids):
        batch, length, _ = x.shape
        value_input = x
        if self.layer == 0:
            channels = (torch.arange(64, device=x.device) + 16 * (ids % 4).unsqueeze(-1)) % 64
            value_input = x.gather(-1, channels)
        q = self.q(x).view(batch, length, 4, 16).transpose(1, 2)
        k = self.k(x).view(batch, length, 1, 16).transpose(1, 2)
        v = self.v(value_input).view(batch, length, 1, 16).transpose(1, 2)
        q, k = rope(self.norm_q(q)), rope(self.norm_k(k))
        y = F.scaled_dot_product_attention(
            q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), is_causal=True, dropout_p=0.0
        )
        if self.layer == 1:
            a = (ids // 12) % 4
            b = (a + 1) % 4
        else:
            a = ids % 4
            b = (a + 1 + (ids // 4) % 3) % 4
        selection = F.one_hot(torch.stack((a, b), -1), 4).sum(-2)
        y = y * selection.permute(0, 2, 1).unsqueeze(-1).to(y.dtype) * (2**0.5)
        return self.o(y.transpose(1, 2).reshape(batch, length, 64))


class Block(nn.Module):
    def __init__(self, mlp, layer):
        super().__init__()
        self.attention = Attention(layer)
        self.norm1 = nn.RMSNorm(64, eps=1e-6)
        self.norm2 = nn.RMSNorm(64, eps=1e-6)
        self.mlp = mlp

    def forward(self, x, ids):
        x = x + self.attention(self.norm1(x), ids)
        return x + self.mlp(self.norm2(x), ids)


class TRHashReference(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 64)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.layers = nn.ModuleList(Block(RoutedMLP(i), i) for i in range(2))
        self.norm = nn.RMSNorm(64, eps=1e-6)

    def forward(self, ids):
        x = self.embedding(ids)
        for layer in self.layers:
            x = layer(x, ids)
        return F.linear(self.norm(x), self.embedding.weight)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
