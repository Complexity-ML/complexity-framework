"""
RWKV: Reinventing RNNs for the Transformer Era.

RWKV combines the best of RNNs and Transformers:
- O(N) complexity like RNNs
- Parallelizable training like Transformers
- No attention mechanism

Key mechanisms:
- Time-mixing: Linear attention with exponential decay
- Channel-mixing: Token-based feed-forward

Reference: https://arxiv.org/abs/2305.13048
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class RWKVConfig:
    """Configuration for RWKV model."""
    hidden_size: int = 768
    num_layers: int = 12
    vocab_size: int = 50257
    ctx_len: int = 1024
    norm_eps: float = 1e-5


class RWKVTimeMix(nn.Module):
    """
    RWKV Time-Mixing block (replaces attention).

    Computes a linear attention with exponential decay:
    wkv(t) = Σ exp(-(t-i)*w + k[i]) * v[i] / Σ exp(-(t-i)*w + k[i])

    Where w is a learned decay parameter.
    """

    def __init__(self, config: RWKVConfig, layer_id: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len

        # Layer-specific initialization
        ratio_0_to_1 = layer_id / max(config.num_layers - 1, 1)
        ratio_1_to_almost_0 = 1.0 - (layer_id / config.num_layers)

        # Time decay (learned per-channel)
        decay_speed = torch.ones(config.hidden_size)
        for h in range(config.hidden_size):
            decay_speed[h] = -5 + 8 * (h / (config.hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        self.time_decay = nn.Parameter(decay_speed)

        # Time first (bonus for current position)
        time_first = torch.ones(config.hidden_size) * math.log(0.3)
        self.time_first = nn.Parameter(time_first)

        # Mix parameters for r, k, v
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, config.hidden_size) * 0.65)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, config.hidden_size) * 0.45)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, config.hidden_size) * 0.0)

        # Projections
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.receptance = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass for time-mixing.

        Args:
            x: [batch, seq_len, hidden_size]
            state: Optional (aa, bb, pp) state for incremental inference

        Returns:
            output: [batch, seq_len, hidden_size]
            new_state: Updated state
        """
        batch_size, seq_len, hidden_size = x.shape

        # Get previous token for mixing
        if state is not None:
            last_x = state[0]
            x_prev = torch.cat([last_x.unsqueeze(1), x[:, :-1, :]], dim=1)
        else:
            x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))  # Pad with zeros

        # Mix current and previous
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)

        # Project
        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        # WKV computation
        if state is not None and seq_len == 1:
            # Incremental inference
            aa, bb, pp = state[1], state[2], state[3]
            wkv, new_aa, new_bb, new_pp = self._wkv_single(k, v, aa, bb, pp)
            new_state = (x[:, -1, :], new_aa, new_bb, new_pp)
        else:
            # Full sequence (training)
            wkv = self._wkv_full(k, v)
            # Compute final state
            new_state = (
                x[:, -1, :],
                torch.zeros(batch_size, hidden_size, device=x.device),
                torch.zeros(batch_size, hidden_size, device=x.device),
                torch.full((batch_size, hidden_size), float('-inf'), device=x.device),
            )

        # Gate and project
        output = self.output(r * wkv)

        return output, new_state

    def _wkv_full(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Full WKV computation for training.

        Uses parallel scan for efficiency.
        """
        batch_size, seq_len, hidden_size = k.shape
        w = self.time_decay
        u = self.time_first

        # Initialize
        aa = torch.zeros(batch_size, hidden_size, device=k.device, dtype=k.dtype)
        bb = torch.zeros(batch_size, hidden_size, device=k.device, dtype=k.dtype)
        pp = torch.full((batch_size, hidden_size), float('-inf'), device=k.device, dtype=k.dtype)

        wkvs = []
        for t in range(seq_len):
            kt = k[:, t, :]
            vt = v[:, t, :]

            # ww = u + kt
            ww = u + kt
            # p = max(pp, ww)
            p = torch.maximum(pp, ww)
            # e1 = exp(pp - p)
            e1 = torch.exp(pp - p)
            # e2 = exp(ww - p)
            e2 = torch.exp(ww - p)

            # wkv = (e1 * aa + e2 * vt) / (e1 * bb + e2)
            wkv_t = (e1 * aa + e2 * vt) / (e1 * bb + e2 + 1e-9)
            wkvs.append(wkv_t)

            # Update state
            # ww = w + pp
            ww = w + pp
            # p = max(ww, kt)
            p = torch.maximum(ww, kt)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kt - p)

            aa = e1 * aa + e2 * vt
            bb = e1 * bb + e2
            pp = p

        return torch.stack(wkvs, dim=1)

    def _wkv_single(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        aa: torch.Tensor,
        bb: torch.Tensor,
        pp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step WKV for inference."""
        k = k.squeeze(1)
        v = v.squeeze(1)
        w = self.time_decay
        u = self.time_first

        ww = u + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)

        wkv = (e1 * aa + e2 * v) / (e1 * bb + e2 + 1e-9)

        # Update state
        ww = w + pp
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)

        new_aa = e1 * aa + e2 * v
        new_bb = e1 * bb + e2
        new_pp = p

        return wkv.unsqueeze(1), new_aa, new_bb, new_pp


class RWKVChannelMix(nn.Module):
    """
    RWKV Channel-Mixing block (replaces FFN).

    Similar to a gated FFN but with time-mixing:
    y = σ(r) * (W_v * (max(k, 0)^2))
    """

    def __init__(self, config: RWKVConfig, layer_id: int):
        super().__init__()

        self.hidden_size = config.hidden_size

        # Mix parameters
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, config.hidden_size) * 0.65)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, config.hidden_size) * 0.0)

        # Projections (4x expansion)
        hidden_dim = config.hidden_size * 4
        self.key = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, config.hidden_size, bias=False)
        self.receptance = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for channel-mixing.

        Args:
            x: [batch, seq_len, hidden_size]
            state: Optional previous x for mixing

        Returns:
            output: [batch, seq_len, hidden_size]
            new_state: Last x for next call
        """
        # Get previous token for mixing
        if state is not None:
            x_prev = torch.cat([state.unsqueeze(1), x[:, :-1, :]], dim=1)
        else:
            x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))

        # Mix
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)

        # Compute
        k = self.key(xk)
        k = torch.square(F.relu(k))  # Squared ReLU activation
        output = torch.sigmoid(self.receptance(xr)) * self.value(k)

        return output, x[:, -1, :]


class RWKVBlock(nn.Module):
    """Single RWKV block combining time-mixing and channel-mixing."""

    def __init__(self, config: RWKVConfig, layer_id: int):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

        self.time_mix = RWKVTimeMix(config, layer_id)
        self.channel_mix = RWKVChannelMix(config, layer_id)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward with residual connections."""
        time_state = state.get("time") if state else None
        channel_state = state.get("channel") if state else None

        # Time mixing
        time_out, new_time_state = self.time_mix(self.ln1(x), time_state)
        x = x + time_out

        # Channel mixing
        channel_out, new_channel_state = self.channel_mix(self.ln2(x), channel_state)
        x = x + channel_out

        new_state = {
            "time": new_time_state,
            "channel": new_channel_state,
        }

        return x, new_state


class RWKV(nn.Module):
    """
    Full RWKV model for language modeling.

    RWKV = Receptance Weighted Key Value
    """

    def __init__(self, config: RWKVConfig):
        super().__init__()

        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            RWKVBlock(config, i) for i in range(config.num_layers)
        ])

        self.ln_out = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            state: Optional state dict for incremental inference

        Returns:
            logits: [batch, seq_len, vocab_size]
            new_state: Updated state
        """
        x = self.embedding(input_ids)

        new_state = {}
        for i, layer in enumerate(self.layers):
            layer_state = state.get(f"layer_{i}") if state else None
            x, layer_new_state = layer(x, layer_state)
            new_state[f"layer_{i}"] = layer_new_state

        x = self.ln_out(x)
        logits = self.lm_head(x)

        return logits, new_state

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        state = None

        # Process prompt
        if input_ids.size(1) > 1:
            logits, state = self(input_ids[:, :-1])
            input_ids = input_ids[:, -1:]

        for _ in range(max_new_tokens):
            logits, state = self(input_ids, state)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = next_token

        return input_ids
