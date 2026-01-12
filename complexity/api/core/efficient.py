"""
Efficient API - Pour les small budgets et hardware limité.
==========================================================

Optimisations INL pour builder des modèles avec ressources limitées:
- Quantization (INT8, INT4, FP16, BF16)
- Memory-efficient training (gradient checkpointing, mixed precision)
- Small model architectures optimisées
- CPU-friendly et single GPU setups

Usage:
    from complexity.api import Efficient

    # Quantization
    model = Efficient.quantize(model, bits=4)

    # Mixed precision training
    model, optimizer = Efficient.mixed_precision(model, optimizer)

    # Small efficient architectures
    model = Efficient.tiny_llm(hidden_size=512, num_layers=6)
    model = Efficient.micro_llm(hidden_size=256, num_layers=4)

    # Memory estimation
    Efficient.estimate_memory(model, batch_size=4, seq_len=2048)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, Dict, Any, List
from dataclasses import dataclass
from contextlib import contextmanager


# =============================================================================
# Quantization
# =============================================================================

@dataclass
class QuantConfig:
    """Configuration de quantization."""
    bits: int = 8  # 4, 8
    group_size: int = 128
    symmetric: bool = True
    compute_dtype: torch.dtype = torch.float16


class Quantize:
    """
    Quantization pour réduire la mémoire.

    Usage:
        # Quantize un modèle
        model = Quantize.dynamic(model)  # INT8 dynamic
        model = Quantize.static(model, bits=4)  # INT4 static

        # Quantize un layer spécifique
        linear = Quantize.linear(linear_layer, bits=8)
    """

    @staticmethod
    def dynamic(
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
    ) -> nn.Module:
        """
        Dynamic quantization (INT8).

        Quantize les poids statiquement, activations dynamiquement.
        Bon pour inference CPU.
        """
        return torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=dtype,
        )

    @staticmethod
    def static(
        model: nn.Module,
        bits: int = 8,
        group_size: int = 128,
    ) -> nn.Module:
        """
        Static quantization.

        Pour INT4/INT8 avec groupes.
        """
        config = QuantConfig(bits=bits, group_size=group_size)

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                parent = model
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)

                quantized = QuantizedLinear.from_linear(module, config)
                setattr(parent, parts[-1], quantized)

        return model

    @staticmethod
    def linear(
        layer: nn.Linear,
        bits: int = 8,
        group_size: int = 128,
    ) -> nn.Module:
        """Quantize un seul Linear layer."""
        config = QuantConfig(bits=bits, group_size=group_size)
        return QuantizedLinear.from_linear(layer, config)

    @staticmethod
    def estimate_savings(model: nn.Module, bits: int = 4) -> Dict[str, str]:
        """Estime les économies de mémoire."""
        original_params = sum(p.numel() for p in model.parameters())
        original_bytes = original_params * 4  # FP32

        # Quantized size
        quantized_bytes = original_params * (bits / 8)

        # Scales et zeros (per group)
        num_groups = original_params // 128  # Assuming group_size=128
        overhead_bytes = num_groups * 4 * 2  # scale + zero in FP32

        total_quantized = quantized_bytes + overhead_bytes

        def _format(b):
            if b >= 1e9:
                return f"{b / 1e9:.2f} GB"
            elif b >= 1e6:
                return f"{b / 1e6:.2f} MB"
            return f"{b / 1e3:.2f} KB"

        return {
            "original": _format(original_bytes),
            "quantized": _format(total_quantized),
            "savings": f"{(1 - total_quantized / original_bytes) * 100:.1f}%",
            "compression_ratio": f"{original_bytes / total_quantized:.1f}x",
        }


class QuantizedLinear(nn.Module):
    """Linear layer quantifié (INT4/INT8)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QuantConfig,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Quantized weights (packed)
        self.register_buffer(
            "weight",
            torch.zeros(out_features, in_features // (8 // config.bits), dtype=torch.uint8)
        )

        # Scales and zeros per group
        num_groups = (in_features + config.group_size - 1) // config.group_size
        self.register_buffer("scales", torch.ones(out_features, num_groups))
        self.register_buffer("zeros", torch.zeros(out_features, num_groups))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: QuantConfig) -> "QuantizedLinear":
        """Crée depuis un Linear existant."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            config,
            bias=linear.bias is not None,
        )

        # Quantize weights
        weight = linear.weight.data
        layer._quantize_weights(weight)

        if linear.bias is not None:
            layer.bias.data = linear.bias.data.clone()

        return layer

    def _quantize_weights(self, weight: torch.Tensor):
        """Quantize les poids avec grouping."""
        bits = self.config.bits
        group_size = self.config.group_size
        max_val = 2 ** bits - 1

        out_features, in_features = weight.shape
        num_groups = (in_features + group_size - 1) // group_size

        # Pad if needed
        if in_features % group_size != 0:
            pad_size = group_size - (in_features % group_size)
            weight = F.pad(weight, (0, pad_size))

        weight = weight.view(out_features, num_groups, group_size)

        # Compute scales and zeros per group
        w_min = weight.min(dim=-1).values
        w_max = weight.max(dim=-1).values

        scales = (w_max - w_min) / max_val
        scales = scales.clamp(min=1e-8)
        zeros = w_min

        self.scales.copy_(scales)
        self.zeros.copy_(zeros)

        # Quantize
        weight_q = ((weight - zeros.unsqueeze(-1)) / scales.unsqueeze(-1)).round().clamp(0, max_val)
        weight_q = weight_q.to(torch.uint8)

        # Pack bits
        weight_q = weight_q.view(out_features, -1)
        if bits == 4:
            # Pack 2 values per byte
            weight_q = weight_q[:, ::2] | (weight_q[:, 1::2] << 4)

        self.weight.copy_(weight_q[:, :self.weight.shape[1]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward avec dequantization à la volée."""
        # Dequantize
        weight = self._dequantize()

        # Compute
        output = F.linear(x.to(weight.dtype), weight, self.bias)

        return output

    def _dequantize(self) -> torch.Tensor:
        """Dequantize les poids."""
        bits = self.config.bits
        group_size = self.config.group_size

        weight_q = self.weight

        # Unpack if INT4
        if bits == 4:
            low = weight_q & 0x0F
            high = (weight_q >> 4) & 0x0F
            weight_q = torch.stack([low, high], dim=-1).view(self.out_features, -1)

        weight_q = weight_q.to(self.scales.dtype)

        # Reshape to groups
        num_groups = self.scales.shape[1]
        weight_q = weight_q[:, :num_groups * group_size].view(self.out_features, num_groups, group_size)

        # Dequantize
        weight = weight_q * self.scales.unsqueeze(-1) + self.zeros.unsqueeze(-1)
        weight = weight.view(self.out_features, -1)[:, :self.in_features]

        return weight


# =============================================================================
# Mixed Precision Training
# =============================================================================

class MixedPrecision:
    """
    Mixed precision pour training économique.

    Usage:
        # Setup
        model, optimizer, scaler = MixedPrecision.setup(model, optimizer)

        # Training loop
        with MixedPrecision.autocast():
            loss = model(inputs)

        MixedPrecision.backward(loss, optimizer, scaler)
    """

    @staticmethod
    def setup(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, torch.cuda.amp.GradScaler]:
        """Setup mixed precision training."""
        scaler = torch.cuda.amp.GradScaler()
        return model, optimizer, scaler

    @staticmethod
    @contextmanager
    def autocast(
        device_type: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enabled: bool = True,
    ):
        """Context manager pour autocast."""
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled):
            yield

    @staticmethod
    def backward(
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        clip_grad: Optional[float] = 1.0,
        model: Optional[nn.Module] = None,
    ):
        """Backward pass avec gradient scaling."""
        scaler.scale(loss).backward()

        if clip_grad is not None and model is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    @staticmethod
    def convert_to_fp16(model: nn.Module) -> nn.Module:
        """Convertit le modèle en FP16."""
        return model.half()

    @staticmethod
    def convert_to_bf16(model: nn.Module) -> nn.Module:
        """Convertit le modèle en BF16 (meilleur pour training)."""
        return model.to(torch.bfloat16)


# =============================================================================
# Memory Efficient Training
# =============================================================================

class MemoryEfficient:
    """
    Techniques pour réduire l'usage mémoire.

    Usage:
        # Gradient checkpointing
        model = MemoryEfficient.enable_checkpointing(model)

        # Gradient accumulation
        for i, batch in enumerate(dataloader):
            loss = model(batch) / accum_steps
            MemoryEfficient.accumulate(loss, optimizer, i, accum_steps)

        # Estimate memory
        MemoryEfficient.estimate(model, batch_size=4, seq_len=2048)
    """

    @staticmethod
    def enable_checkpointing(model: nn.Module) -> nn.Module:
        """Active gradient checkpointing."""
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            model._gradient_checkpointing = True

            # Patch forward of transformer blocks
            for module in model.modules():
                if hasattr(module, "forward") and "block" in module.__class__.__name__.lower():
                    module._original_forward = module.forward
                    module.forward = lambda *args, _m=module, **kwargs: \
                        torch.utils.checkpoint.checkpoint(_m._original_forward, *args, use_reentrant=False, **kwargs)

        return model

    @staticmethod
    def disable_checkpointing(model: nn.Module) -> nn.Module:
        """Désactive gradient checkpointing."""
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        else:
            model._gradient_checkpointing = False
        return model

    @staticmethod
    def accumulate(
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        step: int,
        accumulation_steps: int,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> bool:
        """
        Gradient accumulation.

        Returns True quand optimizer.step() a été appelé.
        """
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            return True

        return False

    @staticmethod
    def estimate(
        model: nn.Module,
        batch_size: int = 1,
        seq_len: int = 2048,
        dtype: torch.dtype = torch.float16,
        training: bool = True,
    ) -> Dict[str, str]:
        """
        Estime l'utilisation mémoire.

        Returns dict avec estimations pour model, activations, gradients, optimizer.
        """
        bytes_per_elem = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int8: 1,
        }.get(dtype, 2)

        # Model parameters
        num_params = sum(p.numel() for p in model.parameters())
        model_bytes = num_params * bytes_per_elem

        # Activations (rough estimate)
        # Assuming transformer: ~12 * hidden_size * seq_len * batch_size * num_layers
        hidden_size = 768  # Default
        num_layers = 12
        for module in model.modules():
            if hasattr(module, "hidden_size"):
                hidden_size = module.hidden_size
            if hasattr(module, "num_hidden_layers"):
                num_layers = module.num_hidden_layers

        activation_bytes = 12 * hidden_size * seq_len * batch_size * num_layers * bytes_per_elem

        # Gradients (same as params if training)
        gradient_bytes = model_bytes if training else 0

        # Optimizer states (Adam: 2x params for momentum + variance)
        optimizer_bytes = num_params * 4 * 2 if training else 0  # FP32 optimizer states

        # Total
        total_bytes = model_bytes + activation_bytes + gradient_bytes + optimizer_bytes

        def _format(b):
            if b >= 1e9:
                return f"{b / 1e9:.2f} GB"
            elif b >= 1e6:
                return f"{b / 1e6:.2f} MB"
            return f"{b / 1e3:.2f} KB"

        return {
            "model": _format(model_bytes),
            "activations": _format(activation_bytes),
            "gradients": _format(gradient_bytes),
            "optimizer": _format(optimizer_bytes),
            "total": _format(total_bytes),
            "params": f"{num_params / 1e6:.1f}M",
        }

    @staticmethod
    def offload_to_cpu(model: nn.Module) -> nn.Module:
        """Offload le modèle sur CPU (économise VRAM)."""
        return model.cpu()

    @staticmethod
    def clear_cache():
        """Vide le cache CUDA."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# =============================================================================
# Small Efficient Architectures
# =============================================================================

class SmallModels:
    """
    Architectures pré-configurées pour small budgets.

    Usage:
        # Tiny LLM (~125M params)
        model = SmallModels.tiny(vocab_size=32000)

        # Micro LLM (~30M params)
        model = SmallModels.micro(vocab_size=32000)

        # Nano LLM (~10M params)
        model = SmallModels.nano(vocab_size=32000)

        # Custom efficient
        model = SmallModels.efficient(
            vocab_size=32000,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
        )
    """

    # Pre-defined configs
    CONFIGS = {
        "nano": {
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 512,
            "max_position_embeddings": 1024,
        },
        "micro": {
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 6,
            "intermediate_size": 1024,
            "max_position_embeddings": 2048,
        },
        "tiny": {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "intermediate_size": 1408,
            "max_position_embeddings": 2048,
        },
        "small": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 2048,
            "max_position_embeddings": 4096,
        },
        "medium": {
            "hidden_size": 1024,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "intermediate_size": 2816,
            "max_position_embeddings": 4096,
        },
    }

    @classmethod
    def create(
        cls,
        size: str = "tiny",
        vocab_size: int = 32000,
        use_flash: bool = True,
        use_gqa: bool = True,
        **kwargs
    ) -> nn.Module:
        """Crée un modèle de taille pré-définie."""
        if size not in cls.CONFIGS:
            raise ValueError(f"Unknown size: {size}. Use: {list(cls.CONFIGS.keys())}")

        config = cls.CONFIGS[size].copy()
        config["vocab_size"] = vocab_size
        config.update(kwargs)

        # Optimizations for efficiency
        if use_gqa:
            config["num_key_value_heads"] = max(1, config["num_attention_heads"] // 4)
        if use_flash:
            config["attention_type"] = "flash"

        # Use efficient defaults
        config.setdefault("mlp_type", "swiglu")
        config.setdefault("norm_type", "rmsnorm")
        config.setdefault("position_type", "rope")
        config.setdefault("tie_word_embeddings", True)

        # Import here to avoid circular deps
        from complexity.config import ModelConfig
        from complexity.models import ComplexityModel

        model_config = ModelConfig(**config)
        return ComplexityModel(model_config)

    @classmethod
    def nano(cls, vocab_size: int = 32000, **kwargs) -> nn.Module:
        """
        Nano LLM (~10M params).

        Idéal pour: Tests, prototypage, CPU training.
        """
        return cls.create("nano", vocab_size, **kwargs)

    @classmethod
    def micro(cls, vocab_size: int = 32000, **kwargs) -> nn.Module:
        """
        Micro LLM (~30M params).

        Idéal pour: Single GPU, fine-tuning rapide.
        """
        return cls.create("micro", vocab_size, **kwargs)

    @classmethod
    def tiny(cls, vocab_size: int = 32000, **kwargs) -> nn.Module:
        """
        Tiny LLM (~125M params).

        Idéal pour: Training sur consumer GPU (RTX 3060+).
        """
        return cls.create("tiny", vocab_size, **kwargs)

    @classmethod
    def small(cls, vocab_size: int = 32000, **kwargs) -> nn.Module:
        """
        Small LLM (~350M params).

        Idéal pour: RTX 3080/4070+, A10.
        """
        return cls.create("small", vocab_size, **kwargs)

    @classmethod
    def medium(cls, vocab_size: int = 32000, **kwargs) -> nn.Module:
        """
        Medium LLM (~760M params).

        Idéal pour: RTX 3090/4080+, A100.
        """
        return cls.create("medium", vocab_size, **kwargs)

    @classmethod
    def estimate_params(cls, size: str) -> str:
        """Estime le nombre de paramètres pour une taille."""
        if size not in cls.CONFIGS:
            return "Unknown"

        config = cls.CONFIGS[size]
        h = config["hidden_size"]
        l = config["num_hidden_layers"]
        v = 32000  # Assume vocab

        # Rough estimate: embed + layers + head
        embed = v * h * 2  # Embed + LM head (tied = /2)
        per_layer = 4 * h * h + 3 * h * config["intermediate_size"]  # Attn + MLP
        total = embed / 2 + l * per_layer

        if total >= 1e9:
            return f"~{total / 1e9:.1f}B"
        return f"~{total / 1e6:.0f}M"


# =============================================================================
# Unified Efficient Factory
# =============================================================================

class Efficient:
    """
    Point d'entrée unifié pour les optimisations INL small budget.

    Usage:
        from complexity.api import Efficient

        # Quantization
        model = Efficient.quantize(model, bits=4)

        # Mixed precision
        model, opt, scaler = Efficient.mixed_precision(model, optimizer)

        # Small models
        model = Efficient.tiny_llm(vocab_size=32000)

        # Memory estimation
        print(Efficient.estimate_memory(model, batch_size=4))

        # Training économique
        Efficient.enable_checkpointing(model)
    """

    # Sub-modules
    quantize = Quantize
    mixed = MixedPrecision
    memory = MemoryEfficient
    models = SmallModels

    # ==================== Shortcuts ====================

    @staticmethod
    def quantize_model(model: nn.Module, bits: int = 4) -> nn.Module:
        """Quantize le modèle en INT4/INT8."""
        return Quantize.static(model, bits=bits)

    @staticmethod
    def mixed_precision(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, torch.cuda.amp.GradScaler]:
        """Setup mixed precision training."""
        return MixedPrecision.setup(model, optimizer)

    @staticmethod
    def enable_checkpointing(model: nn.Module) -> nn.Module:
        """Active gradient checkpointing."""
        return MemoryEfficient.enable_checkpointing(model)

    @staticmethod
    def estimate_memory(
        model: nn.Module,
        batch_size: int = 1,
        seq_len: int = 2048,
    ) -> Dict[str, str]:
        """Estime l'usage mémoire."""
        return MemoryEfficient.estimate(model, batch_size, seq_len)

    @staticmethod
    def estimate_savings(model: nn.Module, bits: int = 4) -> Dict[str, str]:
        """Estime les économies de quantization."""
        return Quantize.estimate_savings(model, bits)

    # ==================== Small Models ====================

    @staticmethod
    def nano_llm(vocab_size: int = 32000, **kwargs) -> nn.Module:
        """Nano LLM (~10M params) - CPU friendly."""
        return SmallModels.nano(vocab_size, **kwargs)

    @staticmethod
    def micro_llm(vocab_size: int = 32000, **kwargs) -> nn.Module:
        """Micro LLM (~30M params) - Single GPU."""
        return SmallModels.micro(vocab_size, **kwargs)

    @staticmethod
    def tiny_llm(vocab_size: int = 32000, **kwargs) -> nn.Module:
        """Tiny LLM (~125M params) - Consumer GPU."""
        return SmallModels.tiny(vocab_size, **kwargs)

    @staticmethod
    def small_llm(vocab_size: int = 32000, **kwargs) -> nn.Module:
        """Small LLM (~350M params) - RTX 3080+."""
        return SmallModels.small(vocab_size, **kwargs)

    # ==================== Hardware Recommendations ====================

    @staticmethod
    def recommend_config(
        vram_gb: float,
        training: bool = True,
    ) -> Dict[str, Any]:
        """
        Recommande une config basée sur le VRAM disponible.

        Args:
            vram_gb: VRAM disponible en GB
            training: True pour training, False pour inference

        Returns:
            Dict avec model_size, batch_size, seq_len, optimizations
        """
        if training:
            if vram_gb >= 80:  # A100 80GB
                return {
                    "model_size": "medium",
                    "batch_size": 32,
                    "seq_len": 4096,
                    "optimizations": [],
                }
            elif vram_gb >= 40:  # A100 40GB
                return {
                    "model_size": "small",
                    "batch_size": 16,
                    "seq_len": 4096,
                    "optimizations": ["flash_attention"],
                }
            elif vram_gb >= 24:  # RTX 3090/4090
                return {
                    "model_size": "tiny",
                    "batch_size": 8,
                    "seq_len": 2048,
                    "optimizations": ["flash_attention", "gradient_checkpointing"],
                }
            elif vram_gb >= 12:  # RTX 3060/4070
                return {
                    "model_size": "micro",
                    "batch_size": 4,
                    "seq_len": 1024,
                    "optimizations": ["flash_attention", "gradient_checkpointing", "mixed_precision"],
                }
            elif vram_gb >= 8:  # RTX 3050
                return {
                    "model_size": "nano",
                    "batch_size": 2,
                    "seq_len": 512,
                    "optimizations": ["gradient_checkpointing", "mixed_precision", "gradient_accumulation"],
                }
            else:  # CPU or very limited
                return {
                    "model_size": "nano",
                    "batch_size": 1,
                    "seq_len": 256,
                    "optimizations": ["cpu", "gradient_accumulation"],
                }
        else:  # Inference
            if vram_gb >= 24:
                return {
                    "model_size": "small",
                    "batch_size": 32,
                    "seq_len": 4096,
                    "optimizations": ["flash_attention"],
                }
            elif vram_gb >= 12:
                return {
                    "model_size": "tiny",
                    "batch_size": 16,
                    "seq_len": 2048,
                    "optimizations": ["flash_attention", "int8_quantization"],
                }
            elif vram_gb >= 8:
                return {
                    "model_size": "micro",
                    "batch_size": 8,
                    "seq_len": 1024,
                    "optimizations": ["int4_quantization"],
                }
            else:
                return {
                    "model_size": "nano",
                    "batch_size": 1,
                    "seq_len": 512,
                    "optimizations": ["int4_quantization", "cpu"],
                }

    @staticmethod
    def get_vram() -> float:
        """Retourne le VRAM disponible en GB."""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / 1e9
        return 0.0


__all__ = [
    # Main factory
    "Efficient",
    # Quantization
    "Quantize",
    "QuantConfig",
    "QuantizedLinear",
    # Mixed Precision
    "MixedPrecision",
    # Memory
    "MemoryEfficient",
    # Small Models
    "SmallModels",
]
