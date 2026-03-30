"""
RetNet: Retentive Network.

RetNet introduces "retention" mechanism as an alternative to attention:
- O(N) complexity for inference (recurrent mode)
- O(N²) but parallelizable for training
- Supports chunk-wise parallel + recurrent hybrid

Key insight: Retention is linear attention with exponential decay,
but designed to be more stable and effective.

Reference: https://arxiv.org/abs/2307.08621
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class RetNetConfig:
    """Configuration for RetNet model."""
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
    ffn_dim: int = 3072
    vocab_size: int = 50257
    max_seq_len: int = 2048
    dropout: float = 0.0
    norm_eps: float = 1e-6


def get_decay_gammas(num_heads: int) -> torch.Tensor:
    """
    Get decay rates (gamma) for each head.

    Different heads have different decay rates to capture
    patterns at different time scales.
    """
    # Decay rates range from 1 - 2^(-5) to 1 - 2^(-12)
    gammas = 1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), num_heads))
    return gammas


class MultiScaleRetention(nn.Module):
    """
    Multi-Scale Retention (MSR) layer.

    Each head has a different decay rate, allowing the model
    to capture patterns at different time scales.

    Retention formula:
    Retention(X) = (QK^T ⊙ D) V

    Where D is a causal decay mask:
    D[n,m] = γ^(n-m) if n >= m else 0
    """

    def __init__(self, config: RetNetConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        assert self.hidden_size % self.num_heads == 0

        # Projections with swish activation for Q, K
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.g_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Output projection
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Group normalization per head
        self.group_norm = nn.GroupNorm(config.num_heads, config.hidden_size)

        # Decay rates (gamma) for each head
        self.register_buffer(
            "gammas",
            get_decay_gammas(config.num_heads),
        )

        # Scaling factor
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        mode: str = "parallel",  # "parallel", "recurrent", or "chunkwise"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with three modes:
        - parallel: O(N²) but parallelizable (training)
        - recurrent: O(1) per step (inference)
        - chunkwise: Hybrid for long sequences

        Args:
            x: [batch, seq_len, hidden_size]
            state: Optional recurrent state [batch, heads, head_dim, head_dim]
            mode: Computation mode

        Returns:
            output: [batch, seq_len, hidden_size]
            new_state: Updated state for recurrent mode
        """
        batch_size, seq_len, _ = x.shape

        # Projections
        q = self.q_proj(x)  # [batch, seq, hidden]
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = F.silu(self.g_proj(x))  # Gate with swish

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mode == "parallel":
            output, new_state = self._parallel_retention(q, k, v)
        elif mode == "recurrent":
            output, new_state = self._recurrent_retention(q, k, v, state)
        else:  # chunkwise
            output, new_state = self._chunkwise_retention(q, k, v, state)

        # Reshape: [batch, seq, heads, head_dim] -> [batch, seq, hidden]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_size)

        # Group norm
        output = self.group_norm(output.transpose(1, 2)).transpose(1, 2)

        # Gate and project
        output = (g * output)
        output = self.out_proj(output)

        return output, new_state

    def _parallel_retention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """
        Parallel retention for training.

        Computes full retention matrix with decay mask.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device

        # Create decay mask D[n,m] = gamma^(n-m) for n >= m, else 0
        # Position indices
        n = torch.arange(seq_len, device=device).unsqueeze(1)
        m = torch.arange(seq_len, device=device).unsqueeze(0)

        # Decay for each head: gamma^(n-m)
        # gammas: [num_heads]
        decay = self.gammas.view(1, num_heads, 1, 1) ** (n - m).unsqueeze(0).unsqueeze(0)

        # Causal mask
        causal_mask = (n >= m).unsqueeze(0).unsqueeze(0).float()
        decay = decay * causal_mask  # [1, heads, seq, seq]

        # Retention: (Q @ K^T) * D @ V
        # Q @ K^T: [batch, heads, seq, seq]
        retention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        retention = retention * decay

        # @ V: [batch, heads, seq, head_dim]
        output = torch.matmul(retention, v)

        return output, None

    def _recurrent_retention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent retention for inference.

        State: S[n] = gamma * S[n-1] + k[n]^T @ v[n]
        Output: o[n] = q[n] @ S[n]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Initialize state if needed
        if state is None:
            state = torch.zeros(
                batch_size, num_heads, head_dim, head_dim,
                device=q.device, dtype=q.dtype
            )

        outputs = []
        for t in range(seq_len):
            # Get current timestep
            q_t = q[:, :, t:t+1, :]  # [batch, heads, 1, head_dim]
            k_t = k[:, :, t:t+1, :]
            v_t = v[:, :, t:t+1, :]

            # Update state: S = gamma * S + k^T @ v
            # k^T @ v: [batch, heads, head_dim, head_dim]
            kv = torch.matmul(k_t.transpose(-2, -1), v_t)

            # Decay per head
            gamma = self.gammas.view(1, num_heads, 1, 1)
            state = gamma * state + kv

            # Output: q @ S
            # [batch, heads, 1, head_dim] @ [batch, heads, head_dim, head_dim]
            o_t = torch.matmul(q_t, state) * self.scale
            outputs.append(o_t)

        output = torch.cat(outputs, dim=2)  # [batch, heads, seq, head_dim]

        return output, state

    def _chunkwise_retention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: Optional[torch.Tensor],
        chunk_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chunk-wise retention for long sequences.

        Combines parallel (within chunk) and recurrent (across chunks).
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Initialize state
        if state is None:
            state = torch.zeros(
                batch_size, num_heads, head_dim, head_dim,
                device=q.device, dtype=q.dtype
            )

        # Pad sequence to chunk boundary
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        pad_len = num_chunks * chunk_size - seq_len

        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        # Reshape to chunks
        q = q.view(batch_size, num_heads, num_chunks, chunk_size, head_dim)
        k = k.view(batch_size, num_heads, num_chunks, chunk_size, head_dim)
        v = v.view(batch_size, num_heads, num_chunks, chunk_size, head_dim)

        outputs = []
        for c in range(num_chunks):
            q_c = q[:, :, c]  # [batch, heads, chunk_size, head_dim]
            k_c = k[:, :, c]
            v_c = v[:, :, c]

            # Intra-chunk: parallel retention
            intra_out, _ = self._parallel_retention(q_c, k_c, v_c)

            # Inter-chunk: contribution from state
            # [batch, heads, chunk_size, head_dim] @ [batch, heads, head_dim, head_dim]
            decay_factor = self.gammas.view(1, num_heads, 1, 1) ** chunk_size
            inter_out = torch.matmul(q_c, state) * self.scale

            # Combine with appropriate decay
            # Position-dependent decay for inter-chunk contribution
            pos = torch.arange(chunk_size, device=q.device)
            pos_decay = self.gammas.view(1, num_heads, 1, 1) ** pos.view(1, 1, chunk_size, 1)
            inter_out = inter_out * pos_decay

            output_c = intra_out + inter_out
            outputs.append(output_c)

            # Update state with this chunk
            # Decay existing state
            state = decay_factor * state
            # Add contribution from this chunk
            for t in range(chunk_size):
                k_t = k_c[:, :, t:t+1, :]
                v_t = v_c[:, :, t:t+1, :]
                kv = torch.matmul(k_t.transpose(-2, -1), v_t)
                decay_t = self.gammas.view(1, num_heads, 1, 1) ** (chunk_size - t - 1)
                state = state + kv * decay_t

        output = torch.cat(outputs, dim=2)  # [batch, heads, padded_seq, head_dim]

        # Remove padding
        if pad_len > 0:
            output = output[:, :, :seq_len, :]

        return output, state


class RetNetFFN(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, config: RetNetConfig):
        super().__init__()

        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RetNetBlock(nn.Module):
    """Single RetNet block."""

    def __init__(self, config: RetNetConfig):
        super().__init__()

        self.retention = MultiScaleRetention(config)
        self.ffn = RetNetFFN(config)

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        mode: str = "parallel",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Retention with residual
        residual = x
        x = self.norm1(x)
        x, new_state = self.retention(x, state, mode)
        x = self.dropout(x)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x, new_state


class RetNet(nn.Module):
    """
    Full RetNet model for language modeling.

    Retentive Network offers:
    - Training: Parallel mode with O(N²) but parallelizable
    - Inference: Recurrent mode with O(1) per token
    - Long sequences: Chunkwise mode combining both
    """

    def __init__(self, config: RetNetConfig):
        super().__init__()

        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            RetNetBlock(config) for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
        mode: str = "parallel",
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            states: Optional list of states per layer
            mode: "parallel", "recurrent", or "chunkwise"

        Returns:
            logits: [batch, seq_len, vocab_size]
            new_states: Updated states
        """
        x = self.embedding(input_ids)

        new_states = []
        for i, layer in enumerate(self.layers):
            state = states[i] if states else None
            x, new_state = layer(x, state, mode)
            new_states.append(new_state)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, new_states

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text using recurrent mode for efficiency."""
        # Process prompt in parallel mode
        if input_ids.size(1) > 1:
            logits, states = self(input_ids[:, :-1], mode="parallel")
            input_ids = input_ids[:, -1:]
        else:
            states = None

        generated = [input_ids]

        for _ in range(max_new_tokens):
            logits, states = self(input_ids, states, mode="recurrent")
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)
            input_ids = next_token

        return torch.cat(generated, dim=1)
