"""
Mixture of Depths (MoD): Dynamically allocating compute to tokens.

Instead of applying all layers to all tokens, MoD learns to:
- Route "easy" tokens through fewer layers
- Route "hard" tokens through more layers

This reduces overall compute while maintaining quality.

Key insight: Not all tokens require equal computation.
Some tokens (like punctuation, common words) are "easy",
while others (factual recall, reasoning) are "hard".

Reference: https://arxiv.org/abs/2404.02258
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class MoDConfig:
    """Configuration for Mixture of Depths."""
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 50257
    max_position_embeddings: int = 2048
    capacity_factor: float = 0.5  # Fraction of tokens to route through each layer
    norm_eps: float = 1e-6
    use_aux_loss: bool = True
    aux_loss_weight: float = 0.01


class TokenRouter(nn.Module):
    """
    Router that decides which tokens go through a layer.

    Learns to predict token difficulty and routes accordingly.
    """

    def __init__(self, hidden_size: int, capacity_factor: float = 0.5):
        super().__init__()

        self.capacity_factor = capacity_factor

        # Simple linear router
        self.router = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens based on learned difficulty.

        Args:
            x: [batch, seq_len, hidden_size]
            training: Whether in training mode

        Returns:
            routing_weights: [batch, seq_len, 1] - soft weights for training
            routing_mask: [batch, seq_len] - hard mask (top-k)
            router_logits: [batch, seq_len] - raw logits for aux loss
        """
        batch_size, seq_len, _ = x.shape

        # Compute routing scores
        router_logits = self.router(x).squeeze(-1)  # [batch, seq_len]

        # Number of tokens to route
        capacity = int(seq_len * self.capacity_factor)

        if training:
            # Soft routing with straight-through estimator
            routing_probs = torch.sigmoid(router_logits)

            # Top-k selection
            _, top_indices = torch.topk(router_logits, capacity, dim=-1)

            # Create hard mask
            routing_mask = torch.zeros_like(router_logits)
            routing_mask.scatter_(1, top_indices, 1.0)

            # Straight-through: use hard mask in forward, soft probs in backward
            routing_weights = routing_mask + routing_probs - routing_probs.detach()
            routing_weights = routing_weights.unsqueeze(-1)
        else:
            # Hard routing for inference
            _, top_indices = torch.topk(router_logits, capacity, dim=-1)
            routing_mask = torch.zeros_like(router_logits)
            routing_mask.scatter_(1, top_indices, 1.0)
            routing_weights = routing_mask.unsqueeze(-1)

        return routing_weights, routing_mask, router_logits


class MoDAttention(nn.Module):
    """Standard multi-head attention for MoD."""

    def __init__(self, config: MoDConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            scores = scores + attention_mask
        else:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=x.device),
                diagonal=1
            )
            scores = scores + causal_mask

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output)


class MoDFFN(nn.Module):
    """Feed-forward network with SwiGLU."""

    def __init__(self, config: MoDConfig):
        super().__init__()

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoDBlock(nn.Module):
    """
    Mixture of Depths block.

    Routes a subset of tokens through attention and FFN,
    while others skip via residual connection.
    """

    def __init__(self, config: MoDConfig):
        super().__init__()

        self.capacity_factor = config.capacity_factor

        # Token router
        self.router = TokenRouter(config.hidden_size, config.capacity_factor)

        # Sub-layers
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.attention = MoDAttention(config)

        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn = MoDFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with selective routing.

        Args:
            x: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            training: Training mode flag

        Returns:
            output: [batch, seq_len, hidden_size]
            router_logits: For auxiliary loss
        """
        batch_size, seq_len, hidden_size = x.shape

        # Route tokens
        routing_weights, routing_mask, router_logits = self.router(x, training)

        # Attention (only for routed tokens)
        residual = x
        x_norm = self.attn_norm(x)

        if training:
            # Full computation with weighted residual
            attn_out = self.attention(x_norm, attention_mask)
            x = residual + routing_weights * attn_out
        else:
            # Sparse computation for efficiency
            # Get indices of routed tokens
            routed_indices = routing_mask.nonzero(as_tuple=True)

            if len(routed_indices[0]) > 0:
                # Gather routed tokens
                # This is simplified - full implementation would batch properly
                attn_out = self.attention(x_norm, attention_mask)
                x = residual + routing_weights * attn_out
            else:
                x = residual

        # FFN (same routing)
        residual = x
        x_norm = self.ffn_norm(x)

        if training:
            ffn_out = self.ffn(x_norm)
            x = residual + routing_weights * ffn_out
        else:
            if len(routed_indices[0]) > 0:
                ffn_out = self.ffn(x_norm)
                x = residual + routing_weights * ffn_out
            else:
                x = residual

        return x, router_logits


class MixtureOfDepths(nn.Module):
    """
    Full Mixture of Depths model.

    Key features:
    - Each layer has a router that selects top-k tokens
    - Unselected tokens skip the layer (residual only)
    - Reduces FLOPs proportional to (1 - capacity_factor)
    """

    def __init__(self, config: MoDConfig):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.layers = nn.ModuleList([
            MoDBlock(config) for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Auxiliary loss for load balancing
        self.aux_loss_weight = config.aux_loss_weight
        self.use_aux_loss = config.use_aux_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional mask
            labels: Optional labels for loss computation

        Returns:
            Dictionary with logits, loss, aux_loss
        """
        batch_size, seq_len = input_ids.shape
        training = self.training

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.position_embedding(positions)

        # Collect router logits for aux loss
        all_router_logits = []

        # Transformer layers
        for layer in self.layers:
            x, router_logits = layer(x, attention_mask, training)
            all_router_logits.append(router_logits)

        x = self.norm(x)
        logits = self.lm_head(x)

        # Compute losses
        loss = None
        aux_loss = None

        if labels is not None:
            # LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if self.use_aux_loss and training:
            # Auxiliary loss to encourage balanced routing
            aux_loss = self._compute_aux_loss(all_router_logits)

            if loss is not None:
                loss = loss + self.aux_loss_weight * aux_loss

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss,
            "router_logits": all_router_logits,
        }

    def _compute_aux_loss(self, router_logits: list) -> torch.Tensor:
        """
        Compute auxiliary loss for load balancing.

        Encourages router to distribute tokens evenly across
        "route" and "skip" decisions.
        """
        aux_loss = 0.0

        for logits in router_logits:
            # logits: [batch, seq_len]
            probs = torch.sigmoid(logits)

            # Mean routing probability
            mean_prob = probs.mean()

            # Encourage mean to be close to capacity_factor
            target = self.config.capacity_factor
            aux_loss = aux_loss + (mean_prob - target) ** 2

        return aux_loss / len(router_logits)

    def get_routing_stats(self, router_logits: list) -> dict:
        """Get statistics about routing decisions."""
        stats = {
            "mean_routing_prob": [],
            "routing_entropy": [],
        }

        for logits in router_logits:
            probs = torch.sigmoid(logits)
            stats["mean_routing_prob"].append(probs.mean().item())

            # Entropy of routing decisions
            entropy = -probs * torch.log(probs + 1e-9) - (1 - probs) * torch.log(1 - probs + 1e-9)
            stats["routing_entropy"].append(entropy.mean().item())

        return stats

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        self.eval()

        for _ in range(max_new_tokens):
            outputs = self(input_ids)
            logits = outputs["logits"][:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


class AdaptiveMoD(MixtureOfDepths):
    """
    Adaptive Mixture of Depths.

    Extends MoD with:
    - Per-layer adaptive capacity
    - Token difficulty estimation
    - Early exit for very easy tokens
    """

    def __init__(self, config: MoDConfig):
        super().__init__(config)

        # Per-layer learnable capacity
        self.layer_capacities = nn.Parameter(
            torch.ones(config.num_layers) * config.capacity_factor
        )

        # Early exit classifier
        self.early_exit = nn.Linear(config.hidden_size, 1, bias=False)
        self.early_exit_threshold = 0.9

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward with adaptive capacity per layer."""
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.position_embedding(positions)

        all_router_logits = []
        early_exit_losses = []

        for i, layer in enumerate(self.layers):
            # Update router capacity for this layer
            capacity = torch.sigmoid(self.layer_capacities[i]).item()
            layer.router.capacity_factor = capacity

            x, router_logits = layer(x, attention_mask, self.training)
            all_router_logits.append(router_logits)

            # Early exit check (during inference)
            if not self.training and i < len(self.layers) - 1:
                exit_prob = torch.sigmoid(self.early_exit(x)).squeeze(-1)
                if (exit_prob > self.early_exit_threshold).all():
                    break

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            if self.use_aux_loss:
                aux_loss = self._compute_aux_loss(all_router_logits)
                loss = loss + self.aux_loss_weight * aux_loss

        return {
            "logits": logits,
            "loss": loss,
            "router_logits": all_router_logits,
            "layer_capacities": torch.sigmoid(self.layer_capacities),
        }
