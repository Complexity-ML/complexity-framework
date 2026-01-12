"""
Framework-Complexity
====================

A comprehensive, modular framework for building, training, and deploying
transformer and state-space models. Developed by INL to provide a production-ready
ML infrastructure comparable to Nanotron and NeMo.

Supports:
- **Architectures**: Llama, Mistral, GPT, Mamba, RWKV, RetNet, Mixture of Depths
- **Attention**: Multi-Head, GQA, MQA, Flash, Sliding Window, Sparse, Linear
- **Training**: FSDP, Tensor Parallel, Pipeline Parallel, Mixed Precision
- **Inference**: KV Cache, Speculative Decoding, Continuous Batching
- **Quantization**: INT8, INT4, GPTQ, AWQ, GGUF export
- **Multi-modal**: Vision (ViT, CLIP), Audio (Whisper), Fusion
- **Robotics**: RT-1, RT-2, Octo, DreamerV3, Behavior Cloning

=== EASY API (Recommended for beginners) ===

    from complexity.easy import load_model, chat, generate

    # Load model (one line!)
    model = load_model("complexity-7b")

    # Chat
    response = chat(model, "What is machine learning?")
    print(response.content)

    # Chat with reasoning (step-by-step)
    response = chat(model, "Solve: 2x + 5 = 15", reasoning=True)
    for step in response.steps:
        print(f"- {step}")
    print(f"Answer: {response.conclusion}")

    # Generate text
    text = generate(model, "The future of AI is", max_tokens=100)

    # Robotics
    from complexity.easy import predict_action
    action = predict_action(model, state={"proprio": [...]}, goal="Pick cup")

=== CLI Usage ===

    # Chat
    complexity chat "Hello, how are you?"

    # Chat with reasoning
    complexity chat "Solve 15*23" --reasoning

    # Interactive mode
    complexity interactive

    # Run demo
    complexity demo

=== Advanced API ===

Quick Start:
    from complexity import ComplexityModel, ModelConfig

    # Create custom model
    config = ModelConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_type="swiglu",
    )
    model = ComplexityModel(config)

    # Or use a preset
    model = ComplexityModel.from_preset("llama-7b")
    model = ComplexityModel.from_preset("complexity-7b")

    # Forward pass
    outputs = model(input_ids)
    logits = outputs["logits"]

    # Generation
    output_ids = model.generate(input_ids, max_new_tokens=100)

Training:
    from complexity.training import Trainer, TrainingConfig
    from complexity.parallel import wrap_model_fsdp

    model = wrap_model_fsdp(model)
    trainer = Trainer(model, config, train_loader)
    trainer.train()

Inference:
    from complexity.inference import InferenceEngine, GenerationConfig

    engine = InferenceEngine(model)
    output = engine.generate(input_ids, GenerationConfig(max_new_tokens=100))

Tokenization (INL Complexity Format):
    from complexity.data import ComplexityTokenizer, ComplexityTokens

    tokenizer = ComplexityTokenizer(base_tokenizer)

    # Encode with reasoning
    tokens = tokenizer.encode_chat(
        messages=[{"role": "user", "content": "Hello"}],
        enable_reasoning=True,
    )

    # Robotics encoding
    tokens = tokenizer.encode_robotics(
        state={"proprio": [...]},
        goal="Pick up the cup",
    )

Registry System:
    from complexity.core.registry import register_attention, register_mlp

    @register_attention("my_attention")
    class MyAttention(AttentionBase):
        ...

    config = ModelConfig(attention_type="my_attention")
"""

__version__ = "1.0.0"
__author__ = "INL (Idaho National Laboratory)"

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
    StandardMLP,
    SwiGLUMLP,
    TokenRoutedMLP,
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
    "StandardMLP",
    "SwiGLUMLP",
    "TokenRoutedMLP",
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
