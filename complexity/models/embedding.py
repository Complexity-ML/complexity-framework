"""Bidirectional text-embedding model: TR-Hash architecture, mean-pooled.

Wraps ComplexityModel(config.is_causal=False) -- the hash-routed MoE MLP and
GQA attention are reused unmodified (see complexity/core/attention/gqa.py's
is_causal flag), just run non-causally, like a BERT-style encoder instead of
a decoder. Sentence embeddings are produced by mean-pooling last_hidden_state
over real (non-padding) tokens and L2-normalizing -- the standard approach
used by E5/GTE/BGE-style embedding models, not a learned pooling head, so
there are no parameters here beyond the backbone itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..core.registry import register_model
from .builder import ComplexityModel


def build_extended_attention_mask(padding_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(batch, seq_len) 1=real/0=pad -> additive (batch, 1, 1, seq_len) mask.

    0.0 at attendable key positions, a large negative value (not literal
    -inf, to avoid NaN from -inf * 0 in bf16/fp16 softmax) at padding.
    Broadcasts against (batch, heads, seq_q, seq_k) attention scores.
    """
    keep = padding_mask[:, None, None, :].to(dtype=dtype)
    return (1.0 - keep) * torch.finfo(dtype).min


def mean_pool(last_hidden_state: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings over real (non-pad) tokens only."""
    mask = padding_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return summed / count


@register_model("bidirectional_encoder")
class BidirectionalEmbeddingModel(nn.Module):
    """Sentence-embedding model: bidirectional TR-Hash backbone + mean pooling.

    forward() returns L2-normalized (batch, hidden_size) sentence embeddings,
    ready for cosine-similarity comparison (retrieval, InfoNCE training).
    """

    def __init__(self, config: ModelConfig, backbone: Optional[ComplexityModel] = None):
        super().__init__()
        if getattr(config, "is_causal", True):
            raise ValueError(
                "BidirectionalEmbeddingModel requires config.is_causal=False; "
                "a causal config would silently train an encoder with "
                "future-token masking, which sentence embeddings must not have."
            )
        self.config = config
        self.backbone = backbone if backbone is not None else ComplexityModel(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """input_ids, attention_mask: (batch, seq_len); attention_mask is
        1 for real tokens, 0 for padding. Returns (batch, hidden_size)."""
        extended_mask = build_extended_attention_mask(
            attention_mask, dtype=self.backbone.embed_tokens.weight.dtype
        )
        outputs = self.backbone(input_ids, attention_mask=extended_mask, return_logits=False)
        pooled = mean_pool(outputs["last_hidden_state"], attention_mask)
        return F.normalize(pooled, p=2, dim=-1)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return self.backbone.num_parameters(trainable_only=trainable_only)

    def save_pretrained(self, save_directory: Union[str, Path], safe_serialization: bool = True) -> None:
        # No parameters live outside the backbone (mean pooling is
        # parameter-free), so this delegates directly.
        self.backbone.save_pretrained(save_directory, safe_serialization=safe_serialization)

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Union[str, Path], **kwargs) -> "BidirectionalEmbeddingModel":
        backbone = ComplexityModel.from_pretrained(pretrained_model_path, **kwargs)
        return cls(backbone.config, backbone=backbone)
