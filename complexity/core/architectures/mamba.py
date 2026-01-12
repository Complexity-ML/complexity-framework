"""
Mamba: Linear-Time Sequence Modeling with Selective State Spaces.

Mamba is a state space model (SSM) that achieves O(N) complexity
by using a selective scan mechanism instead of attention.

Key innovations:
- Selective state spaces (input-dependent dynamics)
- Hardware-efficient parallel scan
- No attention mechanism needed

Reference: https://arxiv.org/abs/2312.00752
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class MambaConfig:
    """Configuration for Mamba model."""
    hidden_size: int = 768
    state_size: int = 16  # SSM state dimension (N)
    conv_kernel_size: int = 4  # Convolution kernel size
    expand_factor: int = 2  # Inner dimension = hidden_size * expand_factor
    dt_rank: str = "auto"  # Rank of dt projection
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    num_layers: int = 12
    vocab_size: int = 50257
    norm_eps: float = 1e-5
    bias: bool = False
    conv_bias: bool = True


class MambaBlock(nn.Module):
    """
    Single Mamba block with selective state space.

    Architecture:
    1. Linear projection to expand dimension
    2. Depthwise convolution
    3. SSM (Selective State Space)
    4. Linear projection back

    The SSM computes:
    h(t) = Ah(t-1) + Bx(t)
    y(t) = Ch(t)

    Where A, B, C are input-dependent (selective).
    """

    def __init__(self, config: MambaConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel_size
        self.expand_factor = config.expand_factor
        self.inner_size = config.hidden_size * config.expand_factor

        # Rank for dt projection
        if config.dt_rank == "auto":
            self.dt_rank = math.ceil(config.hidden_size / 16)
        else:
            self.dt_rank = int(config.dt_rank)

        # Input projection (to 2x inner size for gating)
        self.in_proj = nn.Linear(
            config.hidden_size,
            self.inner_size * 2,
            bias=config.bias,
        )

        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_size,
            out_channels=self.inner_size,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size - 1,
            groups=self.inner_size,  # Depthwise
            bias=config.conv_bias,
        )

        # SSM parameters projection
        # Projects to: dt, B, C
        self.x_proj = nn.Linear(
            self.inner_size,
            self.dt_rank + config.state_size * 2,
            bias=False,
        )

        # dt projection
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_size, bias=True)

        # Initialize dt bias to be between dt_min and dt_max
        dt_init_std = self.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # dt bias initialization
        dt = torch.exp(
            torch.rand(self.inner_size) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A parameter (diagonal, log-parameterized for stability)
        A = torch.arange(1, config.state_size + 1, dtype=torch.float32).repeat(self.inner_size, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.inner_size))

        # Output projection
        self.out_proj = nn.Linear(self.inner_size, config.hidden_size, bias=config.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        ssm_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through Mamba block.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            conv_state: Optional conv cache for inference
            ssm_state: Optional SSM state for inference

        Returns:
            output: [batch, seq_len, hidden_size]
            new_conv_state: Updated conv state
            new_ssm_state: Updated SSM state
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Input projection with gating
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)  # [batch, seq, inner_size]

        # Convolution
        x = x.transpose(1, 2)  # [batch, inner_size, seq]

        if conv_state is not None:
            # Incremental inference
            x = torch.cat([conv_state, x], dim=2)
            new_conv_state = x[:, :, -(self.conv_kernel_size - 1):]
        else:
            new_conv_state = None

        x = self.conv1d(x)
        x = x[:, :, :seq_len]  # Remove padding
        x = x.transpose(1, 2)  # [batch, seq, inner_size]

        # Activation
        x = F.silu(x)

        # SSM
        y, new_ssm_state = self._ssm_forward(x, ssm_state)

        # Gating
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        return output, new_conv_state, new_ssm_state

    def _ssm_forward(
        self,
        x: torch.Tensor,
        ssm_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Selective State Space forward pass.

        Computes:
        h(t) = exp(dt * A) * h(t-1) + dt * B * x(t)
        y(t) = C * h(t) + D * x(t)
        """
        batch_size, seq_len, _ = x.shape

        # Get A (negative for stability)
        A = -torch.exp(self.A_log)  # [inner_size, state_size]

        # Project x to get dt, B, C
        x_proj = self.x_proj(x)  # [batch, seq, dt_rank + 2*state_size]

        dt, B, C = torch.split(
            x_proj,
            [self.dt_rank, self.state_size, self.state_size],
            dim=-1,
        )

        # dt projection and softplus
        dt = self.dt_proj(dt)  # [batch, seq, inner_size]
        dt = F.softplus(dt)

        # Discretize A and B
        # A_discrete = exp(dt * A)
        # B_discrete = dt * B

        # For training: use parallel scan
        # For inference: use recurrent form
        if ssm_state is not None:
            # Recurrent inference
            y, new_state = self._ssm_step(x, dt, A, B, C, ssm_state)
        else:
            # Parallel scan for training
            y = self._ssm_scan(x, dt, A, B, C)
            new_state = None

        # Skip connection
        y = y + x * self.D

        return y, new_state

    def _ssm_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parallel selective scan.

        This is the training-time computation using associative scan.
        """
        batch_size, seq_len, inner_size = x.shape
        state_size = self.state_size

        # Compute deltaA = exp(dt * A)
        # dt: [batch, seq, inner_size]
        # A: [inner_size, state_size]
        deltaA = torch.exp(dt.unsqueeze(-1) * A)  # [batch, seq, inner_size, state_size]

        # Compute deltaB = dt * B
        # B: [batch, seq, state_size]
        deltaB_x = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # [batch, seq, inner_size, state_size]

        # Simple sequential scan (not optimal but correct)
        # For production, use CUDA parallel scan
        h = torch.zeros(batch_size, inner_size, state_size, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB_x[:, t]
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # [batch, inner_size]
            ys.append(y)

        return torch.stack(ys, dim=1)  # [batch, seq, inner_size]

    def _ssm_step(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single step SSM for inference."""
        # x: [batch, 1, inner_size]
        batch_size = x.shape[0]

        x = x.squeeze(1)  # [batch, inner_size]
        dt = dt.squeeze(1)
        B = B.squeeze(1)
        C = C.squeeze(1)

        # Discretize
        deltaA = torch.exp(dt.unsqueeze(-1) * A)  # [batch, inner_size, state_size]
        deltaB = dt.unsqueeze(-1) * B.unsqueeze(1)  # [batch, inner_size, state_size]

        # Update state
        new_state = deltaA * ssm_state + deltaB * x.unsqueeze(-1)

        # Output
        y = (new_state * C.unsqueeze(1)).sum(dim=-1)  # [batch, inner_size]

        return y.unsqueeze(1), new_state


class Mamba(nn.Module):
    """
    Full Mamba model for language modeling.

    Architecture:
    - Token embedding
    - N x MambaBlock
    - RMSNorm
    - LM head
    """

    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            cache: Optional cache for incremental inference

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        hidden_states = self.embedding(input_ids)

        for i, layer in enumerate(self.layers):
            conv_state = cache[f"conv_{i}"] if cache else None
            ssm_state = cache[f"ssm_{i}"] if cache else None

            hidden_states, new_conv, new_ssm = layer(hidden_states, conv_state, ssm_state)

            if cache is not None:
                cache[f"conv_{i}"] = new_conv
                cache[f"ssm_{i}"] = new_ssm

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        cache = {}

        for _ in range(max_new_tokens):
            logits = self(input_ids, cache)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
