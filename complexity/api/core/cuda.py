"""
CUDA / Triton Optimizations API - Attentions optimisées pour GPU.
"""

from __future__ import annotations

import torch.nn as nn

from complexity.core.attention.flash_attention import (
    FlashAttention,
    SlidingWindowAttention,
    SparseAttention,
    LinearAttention,
    MultiScaleAttention,
)
from complexity.core.attention import AttentionConfig


class CUDA:
    """
    Factory pour créer des attention layers optimisées CUDA/Triton.

    Ces implémentations utilisent des optimisations GPU avancées:
    - Flash Attention: O(N) mémoire au lieu de O(N²)
    - Sliding Window: Attention locale efficace
    - Sparse: Patterns block-sparse (BigBird/Longformer)
    - Linear: O(N) complexity totale

    Usage:
        # Flash Attention (2-4x plus rapide, moins de mémoire)
        attn = CUDA.flash(hidden_size=4096, num_heads=32)

        # Sliding Window (Mistral-style, window de 4096 tokens)
        attn = CUDA.sliding_window(hidden_size=4096, num_heads=32, window_size=4096)

        # Sparse Attention (BigBird/Longformer)
        attn = CUDA.sparse(hidden_size=4096, num_heads=32, block_size=64)

        # Linear Attention (O(N) au lieu de O(N²))
        attn = CUDA.linear(hidden_size=4096, num_heads=32)

        # Multi-Scale (mix local + global heads)
        attn = CUDA.multiscale(hidden_size=4096, num_heads=32)
    """

    TYPES = {
        "flash": FlashAttention,
        "flash_attention": FlashAttention,
        "sliding_window": SlidingWindowAttention,
        "local": SlidingWindowAttention,
        "sparse": SparseAttention,
        "block_sparse": SparseAttention,
        "linear": LinearAttention,
        "linear_attention": LinearAttention,
        "multiscale": MultiScaleAttention,
    }

    @classmethod
    def create(
        cls,
        attn_type: str = "flash",
        hidden_size: int = 768,
        num_heads: int = 12,
        num_kv_heads: int = None,
        head_dim: int = None,
        **kwargs
    ) -> nn.Module:
        """
        Crée une attention layer optimisée.

        Args:
            attn_type: "flash", "sliding_window", "sparse", "linear", "multiscale"
            hidden_size: Dimension cachée
            num_heads: Nombre de têtes d'attention
            num_kv_heads: Nombre de têtes KV (GQA), défaut = num_heads
            head_dim: Dimension par tête, défaut = hidden_size // num_heads
            **kwargs: Arguments spécifiques au type d'attention
        """
        if attn_type not in cls.TYPES:
            raise ValueError(f"Unknown attention type: {attn_type}. Use: {list(set(cls.TYPES.values()))}")

        attn_cls = cls.TYPES[attn_type]

        config = AttentionConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads or num_heads,
            head_dim=head_dim or hidden_size // num_heads,
        )

        return attn_cls(config, **kwargs)

    # ==================== Flash Attention ====================

    @classmethod
    def flash(
        cls,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_kv_heads: int = None,
        **kwargs
    ) -> nn.Module:
        """
        Flash Attention - Attention memory-efficient.

        Avantages:
        - O(N) mémoire au lieu de O(N²)
        - 2-4x plus rapide que l'attention standard
        - Support natif du masking causal

        Nécessite PyTorch 2.0+ pour le backend flash attention.
        """
        return cls.create("flash", hidden_size, num_heads, num_kv_heads, **kwargs)

    # ==================== Sliding Window ====================

    @classmethod
    def sliding_window(
        cls,
        hidden_size: int = 768,
        num_heads: int = 12,
        window_size: int = 4096,
        num_kv_heads: int = None,
        **kwargs
    ) -> nn.Module:
        """
        Sliding Window Attention - Attention locale.

        Chaque token n'attend que les `window_size` tokens précédents.
        Réduit la complexité de O(N²) à O(N * W).

        Utilisé par:
        - Mistral (window 4096)
        - Longformer
        - BigBird

        Args:
            window_size: Taille de la fenêtre d'attention (défaut 4096)
        """
        return cls.create(
            "sliding_window", hidden_size, num_heads, num_kv_heads,
            window_size=window_size, **kwargs
        )

    @classmethod
    def local(
        cls,
        hidden_size: int = 768,
        num_heads: int = 12,
        window_size: int = 512,
        **kwargs
    ) -> nn.Module:
        """Alias pour sliding_window avec fenêtre plus petite."""
        return cls.sliding_window(hidden_size, num_heads, window_size, **kwargs)

    # ==================== Sparse Attention ====================

    @classmethod
    def sparse(
        cls,
        hidden_size: int = 768,
        num_heads: int = 12,
        block_size: int = 64,
        num_global_tokens: int = 1,
        num_random_blocks: int = 3,
        **kwargs
    ) -> nn.Module:
        """
        Sparse Attention - Block-sparse patterns.

        Combine attention locale + globale + aléatoire.
        Pattern inspiré de BigBird et Longformer.

        Args:
            block_size: Taille des blocs d'attention locale
            num_global_tokens: Tokens avec attention globale (ex: [CLS])
            num_random_blocks: Nombre de blocs aléatoires
        """
        return cls.create(
            "sparse", hidden_size, num_heads,
            block_size=block_size,
            num_global_tokens=num_global_tokens,
            num_random_blocks=num_random_blocks,
            **kwargs
        )

    # ==================== Linear Attention ====================

    @classmethod
    def linear(
        cls,
        hidden_size: int = 768,
        num_heads: int = 12,
        feature_map: str = "elu",
        **kwargs
    ) -> nn.Module:
        """
        Linear Attention - O(N) complexity.

        Remplace softmax(QK^T)V par φ(Q)(φ(K)^T V).
        Permet attention en O(N) au lieu de O(N²).

        Reference: "Transformers are RNNs" (Katharopoulos et al., 2020)

        Args:
            feature_map: "elu" (défaut), "relu", ou "softmax"
        """
        return cls.create(
            "linear", hidden_size, num_heads,
            feature_map=feature_map, **kwargs
        )

    # ==================== Multi-Scale Attention ====================

    @classmethod
    def multiscale(
        cls,
        hidden_size: int = 768,
        num_heads: int = 12,
        local_heads: int = None,
        window_sizes: tuple = (256, 512, 1024),
        **kwargs
    ) -> nn.Module:
        """
        Multi-Scale Attention - Mix local + global.

        Différentes têtes utilisent différentes échelles:
        - Certaines têtes font attention locale (sliding window)
        - Certaines têtes font attention globale
        - Différentes tailles de fenêtre

        Args:
            local_heads: Nombre de têtes locales (défaut: num_heads // 2)
            window_sizes: Tailles de fenêtre pour les têtes locales
        """
        return cls.create(
            "multiscale", hidden_size, num_heads,
            local_heads=local_heads,
            window_sizes=window_sizes,
            **kwargs
        )


# Alias pour Triton (même implémentations, PyTorch utilise Triton en backend)
Triton = CUDA


__all__ = [
    # Factory
    "CUDA",
    "Triton",
    # Classes directes
    "FlashAttention",
    "SlidingWindowAttention",
    "SparseAttention",
    "LinearAttention",
    "MultiScaleAttention",
]
