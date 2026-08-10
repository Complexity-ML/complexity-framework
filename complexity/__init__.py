"""
Framework-Complexity
====================

A modular research framework for building, training, and evaluating Transformer
language models, with first-class support for Token-Routed MLP experiments.

Supports:
- **Architectures**: Llama/GPT-style decoders, Token-Routed MLPs, dense SwiGLU baselines
- **Attention**: Multi-Head, GQA, MQA, SDPA/Flash-compatible attention
- **Training**: DDP/FSDP utilities, mixed precision, checkpointed long runs
- **Inference**: checkpoint export + vLLM/SGLang serving integration
- **Quantization**: INT8, INT4, GPTQ, AWQ, GGUF export
- **Tokenization**: local BPE tokenizers and tiktoken/o200k-compatible tokenizers

=== EASY API (Recommended for beginners) ===

    from complexity import ComplexityModel, ModelConfig

    # Build/train/export PyTorch models here; serve generated text via vLLM/SGLang.
    model = ComplexityModel(ModelConfig())

=== CLI Usage ===

    # Generate through a running vLLM/SGLang OpenAI-compatible server
    complexity inference generate my-model --backend vllm --base-url http://localhost:8000 --prompt "Hello"

=== Advanced API ===

Quick Start:
    from complexity import ComplexityModel, ModelConfig

    # Create custom model
    config = ModelConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_type="tr_hash_engine",
        num_experts=4,
    )
    model = ComplexityModel(config)

    # Or use a preset
    model = ComplexityModel.from_preset("complexity-7b")

    # Forward pass
    outputs = model(input_ids)
    logits = outputs["logits"]

    # Generation is intentionally not a native model method; use vLLM/SGLang.

Training:
    from complexity.training import Trainer, TrainingConfig
    from complexity.parallel import wrap_model_fsdp

    model = wrap_model_fsdp(model)
    trainer = Trainer(model, config, train_loader)
    trainer.train()

Inference:
    from complexity.inference import create_external_backend, ExternalGenerationConfig

    backend = create_external_backend("vllm", base_url="http://localhost:8000", model="my-model")
    output = backend.complete("Hello", ExternalGenerationConfig(max_tokens=100))

Tokenization:
    from complexity.data import ComplexityTokenizer, ComplexityTokens

    tokenizer = ComplexityTokenizer(base_tokenizer)

    # Encode with reasoning
    tokens = tokenizer.encode_chat(
        messages=[{"role": "user", "content": "Hello"}],
        enable_reasoning=True,
    )

Registry System:
    from complexity.core.registry import register_attention, register_mlp

    @register_attention("my_attention")
    class MyAttention(AttentionBase):
        ...

    config = ModelConfig(attention_type="my_attention")
"""

__version__ = "1.0.0"
__author__ = "Complexity-ML"

# Config
from complexity.config import ModelConfig, get_preset, PRESET_CONFIGS

# Models
from complexity.models import ComplexityModel, TransformerBlock

# Core components (for extension)
from complexity.core.registry import (
    ATTENTION_REGISTRY,
    MLP_REGISTRY,
    NORMALIZATION_REGISTRY,
    POSITION_REGISTRY,
    MODEL_REGISTRY,
    register_attention,
    register_mlp,
    register_normalization,
    register_position,
    register_model,
)

# Attention
from complexity.core.attention import (
    AttentionBase,
    AttentionConfig,
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
)

# MLP
from complexity.core.mlp import (
    MLPBase,
    MLPConfig,
    TRHashEngineMLP,
)

# Normalization
from complexity.core.normalization import RMSNorm, LayerNorm

# Position
from complexity.core.position import (
    RotaryEmbedding,
    StandardRoPE,
    YaRNRoPE,
    DynamicNTKRoPE,
    apply_rotary_pos_emb,
)

# General deterministic token-ID routed execution engine
from complexity.tr_hash import (
    TRHashEngine,
    TRHashEngineConfig,
)

# Parallel training (submodule - import as needed)
# from complexity.parallel import wrap_model_fsdp, ShardingMode, PrecisionMode

# Training (submodule - import as needed)
# from complexity.training import Trainer, TrainingConfig

# Utilities (submodule - import as needed)
# from complexity.utils import CheckpointManager, safe_torch_load

__all__ = [
    # Version
    "__version__",
    # Config
    "ModelConfig",
    "get_preset",
    "PRESET_CONFIGS",
    # Models
    "ComplexityModel",
    "TransformerBlock",
    # Registries
    "ATTENTION_REGISTRY",
    "MLP_REGISTRY",
    "NORMALIZATION_REGISTRY",
    "POSITION_REGISTRY",
    "MODEL_REGISTRY",
    # Registration decorators
    "register_attention",
    "register_mlp",
    "register_normalization",
    "register_position",
    "register_model",
    # Attention
    "AttentionBase",
    "AttentionConfig",
    "GroupedQueryAttention",
    "MultiHeadAttention",
    "MultiQueryAttention",
    # MLP
    "MLPBase",
    "MLPConfig",
    "TRHashEngineMLP",
    "TRHashEngine",
    "TRHashEngineConfig",
    # Normalization
    "RMSNorm",
    "LayerNorm",
    # Position
    "RotaryEmbedding",
    "StandardRoPE",
    "YaRNRoPE",
    "DynamicNTKRoPE",
    "apply_rotary_pos_emb",
]
