"""
Helpers API - Utilitaires pour les builders de modèles.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List, Dict, Any
from dataclasses import dataclass


# =============================================================================
# Weight Initialization
# =============================================================================

class Init:
    """
    Factory pour initialiser les poids des modèles.

    Usage:
        Init.xavier(model)
        Init.kaiming(model, mode="fan_out")
        Init.orthogonal(model, gain=1.0)
        Init.normal(model, std=0.02)
    """

    @staticmethod
    def apply(
        module: nn.Module,
        init_type: str = "normal",
        std: float = 0.02,
        **kwargs
    ) -> nn.Module:
        """Applique l'initialisation sur tout le module."""
        init_fn = {
            "normal": Init.normal,
            "xavier": Init.xavier,
            "kaiming": Init.kaiming,
            "orthogonal": Init.orthogonal,
            "zeros": Init.zeros,
        }.get(init_type, Init.normal)

        init_fn(module, std=std, **kwargs)
        return module

    @staticmethod
    def normal(module: nn.Module, std: float = 0.02, **kwargs):
        """Initialisation normale (GPT-style)."""
        for name, param in module.named_parameters():
            if "weight" in name:
                if param.dim() >= 2:
                    nn.init.normal_(param, mean=0.0, std=std)
                else:
                    nn.init.zeros_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    @staticmethod
    def xavier(module: nn.Module, gain: float = 1.0, **kwargs):
        """Xavier/Glorot initialization."""
        for name, param in module.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=gain)
            elif "bias" in name:
                nn.init.zeros_(param)

    @staticmethod
    def kaiming(module: nn.Module, mode: str = "fan_in", nonlinearity: str = "relu", **kwargs):
        """Kaiming/He initialization."""
        for name, param in module.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.kaiming_uniform_(param, mode=mode, nonlinearity=nonlinearity)
            elif "bias" in name:
                nn.init.zeros_(param)

    @staticmethod
    def orthogonal(module: nn.Module, gain: float = 1.0, **kwargs):
        """Orthogonal initialization."""
        for name, param in module.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.orthogonal_(param, gain=gain)
            elif "bias" in name:
                nn.init.zeros_(param)

    @staticmethod
    def zeros(module: nn.Module, **kwargs):
        """Zero initialization (pour output projections)."""
        for param in module.parameters():
            nn.init.zeros_(param)

    @staticmethod
    def scaled_init(module: nn.Module, num_layers: int, std: float = 0.02):
        """
        Scaled initialization (GPT-2 style).

        Scale output projections by 1/sqrt(2*num_layers).
        """
        scale = 1 / math.sqrt(2 * num_layers)
        for name, param in module.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=std * scale)
            elif "bias" in name:
                nn.init.zeros_(param)


# =============================================================================
# Masking Utilities
# =============================================================================

class Mask:
    """
    Factory pour créer des masks d'attention.

    Usage:
        mask = Mask.causal(seq_len=2048, device="cuda")
        mask = Mask.padding(lengths=[10, 15, 8], max_len=20)
        mask = Mask.sliding_window(seq_len=2048, window_size=512)
    """

    @staticmethod
    def causal(
        seq_len: int,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Crée un mask causal (triangulaire inférieur).

        Returns: [seq_len, seq_len] avec -inf au-dessus de la diagonale.
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
            diagonal=1
        )
        return mask

    @staticmethod
    def padding(
        lengths: Union[List[int], torch.Tensor],
        max_len: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Crée un mask de padding.

        Args:
            lengths: Longueurs réelles de chaque séquence [batch_size]
            max_len: Longueur max (défaut: max(lengths))

        Returns: [batch_size, max_len] avec -inf pour les positions paddées.
        """
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, device=device)

        batch_size = lengths.size(0)
        max_len = max_len or int(lengths.max().item())

        # [batch_size, max_len]
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = positions >= lengths.unsqueeze(1)

        return mask.to(dtype) * float("-inf")

    @staticmethod
    def sliding_window(
        seq_len: int,
        window_size: int,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Crée un mask sliding window + causal.

        Chaque token n'attend que les `window_size` tokens précédents.

        Returns: [seq_len, seq_len]
        """
        # Distance matrix
        positions = torch.arange(seq_len, device=device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)

        # Causal + window
        mask = torch.where(
            (distance >= 0) & (distance < window_size),
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(float("-inf"), device=device, dtype=dtype),
        )
        return mask

    @staticmethod
    def block_sparse(
        seq_len: int,
        block_size: int = 64,
        num_global: int = 1,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Crée un mask block-sparse (BigBird style).

        Args:
            block_size: Taille des blocs locaux
            num_global: Nombre de tokens globaux (début de séquence)
        """
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)

        # Global tokens
        mask[:num_global, :] = 0
        mask[:, :num_global] = 0

        # Local blocks
        for i in range(seq_len):
            start = max(0, i - block_size // 2)
            end = min(seq_len, i + block_size // 2 + 1)
            mask[i, start:end] = 0

        # Causal
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        mask = mask.masked_fill(causal, float("-inf"))

        return mask

    @staticmethod
    def combine(
        *masks: torch.Tensor,
        mode: str = "add"
    ) -> torch.Tensor:
        """Combine plusieurs masks."""
        if mode == "add":
            result = masks[0].clone()
            for m in masks[1:]:
                result = result + m
            return result
        elif mode == "min":
            return torch.stack(masks).min(dim=0).values
        else:
            raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# KV Cache
# =============================================================================

@dataclass
class KVCache:
    """
    Cache pour les clés et valeurs d'attention.

    Usage:
        cache = KVCache.create(batch_size=1, num_layers=32, num_heads=32, head_dim=128)

        # Dans la boucle de génération
        k, v = cache.update(layer_idx, new_k, new_v)
    """
    keys: List[Optional[torch.Tensor]]
    values: List[Optional[torch.Tensor]]
    num_layers: int
    max_length: Optional[int] = None

    @classmethod
    def create(
        cls,
        num_layers: int,
        batch_size: int = 1,
        num_heads: int = 32,
        head_dim: int = 128,
        max_length: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> "KVCache":
        """Crée un cache vide."""
        if max_length is not None:
            # Pre-allocate
            keys = [
                torch.zeros(batch_size, max_length, num_heads, head_dim, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
            values = [
                torch.zeros(batch_size, max_length, num_heads, head_dim, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        else:
            keys = [None] * num_layers
            values = [None] * num_layers

        return cls(keys=keys, values=values, num_layers=num_layers, max_length=max_length)

    def update(
        self,
        layer_idx: int,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Met à jour le cache et retourne les KV complets.

        Args:
            layer_idx: Index de la layer
            new_key: [batch, seq, heads, head_dim]
            new_value: [batch, seq, heads, head_dim]

        Returns:
            (full_key, full_value) avec tout l'historique
        """
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = new_key
            self.values[layer_idx] = new_value
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], new_key], dim=1)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], new_value], dim=1)

        return self.keys[layer_idx], self.values[layer_idx]

    def get(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Récupère le cache d'une layer."""
        return self.keys[layer_idx], self.values[layer_idx]

    def clear(self):
        """Vide le cache."""
        self.keys = [None] * self.num_layers
        self.values = [None] * self.num_layers

    @property
    def seq_length(self) -> int:
        """Longueur actuelle du cache."""
        if self.keys[0] is None:
            return 0
        return self.keys[0].size(1)


class SlidingWindowCache(KVCache):
    """Cache avec fenêtre glissante (Mistral-style)."""

    def __init__(self, window_size: int, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    @classmethod
    def create(
        cls,
        num_layers: int,
        window_size: int = 4096,
        **kwargs
    ) -> "SlidingWindowCache":
        cache = super().create(num_layers, **kwargs)
        cache.__class__ = cls
        cache.window_size = window_size
        return cache

    def update(
        self,
        layer_idx: int,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Met à jour avec troncature à window_size."""
        k, v = super().update(layer_idx, new_key, new_value)

        # Truncate to window
        if k.size(1) > self.window_size:
            self.keys[layer_idx] = k[:, -self.window_size:]
            self.values[layer_idx] = v[:, -self.window_size:]

        return self.keys[layer_idx], self.values[layer_idx]


# =============================================================================
# Sampling Utilities
# =============================================================================

class Sampling:
    """
    Utilitaires de sampling pour la génération.

    Usage:
        token = Sampling.top_k(logits, k=50)
        token = Sampling.top_p(logits, p=0.9)
        token = Sampling.temperature(logits, temp=0.7)
    """

    @staticmethod
    def temperature(
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Applique temperature scaling."""
        if temperature == 0:
            return logits  # Will use argmax
        return logits / temperature

    @staticmethod
    def top_k(
        logits: torch.Tensor,
        k: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Top-K sampling.

        Garde seulement les k tokens les plus probables.
        """
        if temperature != 1.0:
            logits = Sampling.temperature(logits, temperature)

        if k > 0:
            # Keep top k
            values, _ = torch.topk(logits, min(k, logits.size(-1)))
            min_value = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_value, float("-inf"), logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @staticmethod
    def top_p(
        logits: torch.Tensor,
        p: float = 0.9,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Top-P (nucleus) sampling.

        Garde les tokens jusqu'à ce que leur prob cumulative atteigne p.
        """
        if temperature != 1.0:
            logits = Sampling.temperature(logits, temperature)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative prob > p
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Scatter back
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @staticmethod
    def greedy(logits: torch.Tensor) -> torch.Tensor:
        """Greedy decoding (argmax)."""
        return logits.argmax(dim=-1, keepdim=True)

    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Sampling combiné.

        Args:
            temperature: Temperature scaling (0 = greedy)
            top_k: Nombre de tokens à garder (0 = désactivé)
            top_p: Nucleus probability (1.0 = désactivé)
        """
        if temperature == 0:
            return Sampling.greedy(logits)

        logits = Sampling.temperature(logits, temperature)

        # Apply top_k
        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            min_value = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_value, float("-inf"), logits)

        # Apply top_p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


# =============================================================================
# Tensor Operations
# =============================================================================

class Tensors:
    """
    Opérations tensorielles communes.

    Usage:
        k_expanded = Tensors.repeat_kv(k, n_rep=4)
        x_rotated = Tensors.rotate_half(x)
    """

    @staticmethod
    def repeat_kv(
        hidden_states: torch.Tensor,
        n_rep: int,
    ) -> torch.Tensor:
        """
        Répète les KV heads pour GQA.

        [batch, seq, kv_heads, head_dim] -> [batch, seq, num_heads, head_dim]
        """
        if n_rep == 1:
            return hidden_states
        batch, seq_len, num_kv_heads, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, :, None, :].expand(
            batch, seq_len, num_kv_heads, n_rep, head_dim
        )
        return hidden_states.reshape(batch, seq_len, num_kv_heads * n_rep, head_dim)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotation pour RoPE."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applique RoPE sur Q et K."""
        q_embed = (q * cos) + (Tensors.rotate_half(q) * sin)
        k_embed = (k * cos) + (Tensors.rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def chunk(x: torch.Tensor, num_chunks: int, dim: int = 0) -> List[torch.Tensor]:
        """Divise un tensor en chunks égaux."""
        return torch.chunk(x, num_chunks, dim=dim)

    @staticmethod
    def interleave(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """Entrelace plusieurs tensors."""
        stacked = torch.stack(tensors, dim=dim + 1)
        shape = list(stacked.shape)
        shape[dim] = -1
        del shape[dim + 1]
        return stacked.reshape(shape)


# =============================================================================
# Debug & Profiling
# =============================================================================

class Debug:
    """
    Utilitaires de debug pour les modèles.

    Usage:
        print(Debug.count_params(model))
        Debug.print_summary(model)
        mem = Debug.memory_usage()
    """

    @staticmethod
    def count_params(
        model: nn.Module,
        trainable_only: bool = False,
        human_readable: bool = True,
    ) -> Union[int, str]:
        """
        Compte les paramètres du modèle.

        Returns: "7.2B" ou 7234567890
        """
        if trainable_only:
            total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            total = sum(p.numel() for p in model.parameters())

        if not human_readable:
            return total

        if total >= 1e12:
            return f"{total / 1e12:.1f}T"
        elif total >= 1e9:
            return f"{total / 1e9:.1f}B"
        elif total >= 1e6:
            return f"{total / 1e6:.1f}M"
        elif total >= 1e3:
            return f"{total / 1e3:.1f}K"
        return str(total)

    @staticmethod
    def model_summary(model: nn.Module, max_depth: int = 2) -> str:
        """Génère un résumé du modèle."""
        lines = []
        lines.append(f"Model: {model.__class__.__name__}")
        lines.append(f"Total params: {Debug.count_params(model)}")
        lines.append(f"Trainable params: {Debug.count_params(model, trainable_only=True)}")
        lines.append("")

        def _summarize(module, prefix="", depth=0):
            if depth > max_depth:
                return
            for name, child in module.named_children():
                params = Debug.count_params(child, human_readable=True)
                lines.append(f"{prefix}{name}: {child.__class__.__name__} ({params})")
                _summarize(child, prefix + "  ", depth + 1)

        _summarize(model)
        return "\n".join(lines)

    @staticmethod
    def memory_usage(device: Union[str, torch.device] = "cuda") -> Dict[str, str]:
        """Retourne l'utilisation mémoire GPU."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        if isinstance(device, str):
            device = torch.device(device)

        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)

        def _format(bytes):
            if bytes >= 1e9:
                return f"{bytes / 1e9:.2f} GB"
            elif bytes >= 1e6:
                return f"{bytes / 1e6:.2f} MB"
            return f"{bytes / 1e3:.2f} KB"

        return {
            "allocated": _format(allocated),
            "reserved": _format(reserved),
            "max_allocated": _format(max_allocated),
        }

    @staticmethod
    def print_summary(model: nn.Module, max_depth: int = 2):
        """Affiche le résumé du modèle."""
        print(Debug.model_summary(model, max_depth))

    @staticmethod
    def check_gradients(model: nn.Module) -> Dict[str, Any]:
        """Vérifie l'état des gradients."""
        stats = {
            "has_nan": False,
            "has_inf": False,
            "max_grad": 0.0,
            "min_grad": float("inf"),
            "layers_with_grad": 0,
            "layers_without_grad": 0,
        }

        for name, param in model.named_parameters():
            if param.grad is not None:
                stats["layers_with_grad"] += 1
                grad = param.grad.data

                if torch.isnan(grad).any():
                    stats["has_nan"] = True
                if torch.isinf(grad).any():
                    stats["has_inf"] = True

                max_val = grad.abs().max().item()
                min_val = grad.abs().min().item()

                stats["max_grad"] = max(stats["max_grad"], max_val)
                stats["min_grad"] = min(stats["min_grad"], min_val)
            else:
                stats["layers_without_grad"] += 1

        return stats


# =============================================================================
# Gradient Checkpointing
# =============================================================================

class Checkpointing:
    """
    Utilitaires pour gradient checkpointing.

    Usage:
        model = Checkpointing.enable(model)
        Checkpointing.disable(model)
    """

    @staticmethod
    def enable(model: nn.Module) -> nn.Module:
        """Active le gradient checkpointing."""
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            # Manual enable for custom models
            model._gradient_checkpointing = True
        return model

    @staticmethod
    def disable(model: nn.Module) -> nn.Module:
        """Désactive le gradient checkpointing."""
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        else:
            model._gradient_checkpointing = False
        return model

    @staticmethod
    def checkpoint(
        function,
        *args,
        use_reentrant: bool = False,
        **kwargs
    ):
        """
        Wrapper pour torch.utils.checkpoint.

        Usage:
            output = Checkpointing.checkpoint(layer.forward, hidden_states)
        """
        from torch.utils.checkpoint import checkpoint
        return checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)


# =============================================================================
# Unified Helpers Factory
# =============================================================================

class Helpers:
    """
    Point d'entrée unifié pour tous les helpers.

    Usage:
        from complexity.api import Helpers

        # Initialisation
        Helpers.init.xavier(model)

        # Masking
        mask = Helpers.mask.causal(seq_len=2048)

        # Cache
        cache = Helpers.cache.create(num_layers=32)

        # Sampling
        token = Helpers.sampling.top_p(logits, p=0.9)

        # Debug
        print(Helpers.debug.count_params(model))
    """

    init = Init
    mask = Mask
    cache = KVCache
    sliding_cache = SlidingWindowCache
    sampling = Sampling
    tensors = Tensors
    debug = Debug
    checkpointing = Checkpointing

    # Shortcuts
    @staticmethod
    def count_params(model: nn.Module, **kwargs) -> str:
        return Debug.count_params(model, **kwargs)

    @staticmethod
    def causal_mask(seq_len: int, **kwargs) -> torch.Tensor:
        return Mask.causal(seq_len, **kwargs)

    @staticmethod
    def kv_cache(num_layers: int, **kwargs) -> KVCache:
        return KVCache.create(num_layers, **kwargs)

    @staticmethod
    def init_weights(model: nn.Module, init_type: str = "normal", **kwargs) -> nn.Module:
        return Init.apply(model, init_type, **kwargs)


__all__ = [
    # Main factory
    "Helpers",
    # Sub-factories
    "Init",
    "Mask",
    "KVCache",
    "SlidingWindowCache",
    "Sampling",
    "Tensors",
    "Debug",
    "Checkpointing",
]
