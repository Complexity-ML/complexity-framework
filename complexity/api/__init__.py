"""
INL Complexity Framework - API Python Complète
==============================================

API flexible pour:
- Utilisation simple (notebook, scripts)
- Construction de modèles maison (architectures custom)
- Training et inference

Usage Basique:
    from complexity.api import Tokenizer, Model, Dataset, Trainer

    tokenizer = Tokenizer.load("llama-7b")
    model = Model.load("llama-7b", device="cuda")
    dataset = Dataset.load("./train.jsonl", tokenizer=tokenizer)

    trainer = Trainer(model, dataset)
    trainer.train()

Usage Avancé (modèles maison):
    from complexity.api import (
        # Building blocks
        Attention, MLP, Position, Norm, Block,
        GQA, SwiGLU, RoPE, RMSNorm, TokenRoutedMLP,
        # CUDA / Triton optimizations
        CUDA, Triton, FlashAttention, SlidingWindowAttention,
        # Inference
        Generate, GenerationConfig,
        # Multimodal
        Vision, Audio, Fusion,
        # Architectures INL (O(N))
        Architecture, Mamba, RWKV, RetNet,
        # Registry pour custom
        register,
    )

    # Construire son propre modèle
    class MonModele(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.attn = GQA(config.hidden_size, config.num_heads, config.kv_heads)
            self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
            self.norm = RMSNorm(config.hidden_size)
            self.rope = RoPE(config.head_dim, config.max_seq_len)

    # Ou via factories
    attn = Attention.gqa(hidden_size=4096, num_heads=32, kv_heads=8)
    mlp = MLP.swiglu(hidden_size=4096, intermediate_size=11008)
    mlp_moe = MLP.moe(hidden_size=4096, num_experts=8, top_k=2)

    # CUDA / Triton optimizations
    flash_attn = CUDA.flash(hidden_size=4096, num_heads=32)
    sliding_attn = CUDA.sliding_window(hidden_size=4096, num_heads=32, window_size=4096)
    sparse_attn = CUDA.sparse(hidden_size=4096, num_heads=32)
    linear_attn = CUDA.linear(hidden_size=4096, num_heads=32)  # O(N)!

    # Enregistrer un composant custom
    @register("attention", "my_attention")
    class MyAttention(AttentionBase):
        ...
"""

# =============================================================================
# Base API (simple usage)
# =============================================================================

from .tokenizer import Tokenizer, TokenizerConfig
from .model import Model
from .trainer import Trainer, TrainerConfig
from .dataset import Dataset, DataConfig, StreamingDataset, DataPipeline

# =============================================================================
# Building Blocks (modèles maison) - depuis core/
# =============================================================================

from .core import (
    # Factories
    Attention,
    MLP,
    Position,
    Norm,
    Block,
    Architecture,
    # Attention
    GQA,
    MHA,
    MQA,
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
    AttentionBase,
    AttentionConfig,
    # MLP
    SwiGLU,
    GeGLU,
    SwiGLUMLP,
    GeGLUMLP,
    StandardMLP,
    TokenRoutedMLP,
    TokenRoutedMLPParallel,
    SparseMoE,
    SparseMoEParallel,
    SparseMoEConfig,
    MLPBase,
    MLPConfig,
    # Position
    RoPE,
    YaRN,
    ALiBi,
    StandardRoPE,
    YaRNRoPE,
    DynamicNTKRoPE,
    ALiBiPositionBias,
    LearnedPositionEmbedding,
    RotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
    # Normalization
    RMSNorm,
    LayerNorm,
    IdentityNorm,
    build_norm,
    # Block & Model
    TransformerBlock,
    ComplexityModel,
    ModelConfig,
    # INL Architectures (O(N))
    Mamba,
    MambaBlock,
    MambaConfig,
    RWKV,
    RWKVBlock,
    RWKVConfig,
    RetNet,
    RetNetBlock,
    RetNetConfig,
    MoDBlock,
    MoDConfig,
    # Registry
    Registry,
    register,
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
    # CUDA / Triton Optimizations
    CUDA,
    Triton,
    FlashAttention,
    SlidingWindowAttention,
    SparseAttention,
    LinearAttention,
    MultiScaleAttention,
    # Helpers
    Helpers,
    Init,
    Mask,
    KVCache,
    SlidingWindowCache,
    Sampling,
    Tensors,
    Debug,
    Checkpointing,
    # Efficient (small budget)
    Efficient,
    Quantize,
    QuantConfig,
    QuantizedLinear,
    MixedPrecision,
    MemoryEfficient,
    SmallModels,
    # INL Dynamics (velocity tracking)
    INLDynamics,
    INLDynamicsLite,
    DynamicsConfig,
)

# =============================================================================
# Inference & Generation
# =============================================================================

from .inference import (
    Generate,
    GenerationConfig,
    InferenceEngine,
    InferenceConfig,
    DecodingStrategy,
    create_engine,
)

# =============================================================================
# Multimodal (Vision, Audio)
# =============================================================================

from .multimodal import (
    # Factories
    Vision,
    Audio,
    Fusion,
    # Vision
    VisionEncoder,
    VisionConfig,
    PatchEmbedding,
    VisionTransformer,
    CLIPVisionEncoder,
    SigLIPEncoder,
    # Audio
    AudioEncoder,
    AudioConfig,
    MelSpectrogramEncoder,
    WhisperEncoder,
    AudioConvStack,
    # Fusion
    MultimodalFusion,
    FusionConfig,
    CrossAttentionFusion,
    GatedFusion,
    ConcatProjection,
    PerceiverResampler,
)

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # ========== Base API ==========
    "Tokenizer",
    "TokenizerConfig",
    "Model",
    "Trainer",
    "TrainerConfig",
    "Dataset",
    "DataConfig",
    "StreamingDataset",
    "DataPipeline",

    # ========== Building Blocks - Factories ==========
    "Attention",
    "MLP",
    "Position",
    "Norm",
    "Block",
    "Architecture",

    # ========== Attention ==========
    "GQA",
    "MHA",
    "MQA",
    "GroupedQueryAttention",
    "MultiHeadAttention",
    "MultiQueryAttention",
    "AttentionBase",
    "AttentionConfig",

    # ========== MLP ==========
    "SwiGLU",
    "GeGLU",
    "SwiGLUMLP",
    "GeGLUMLP",
    "StandardMLP",
    "TokenRoutedMLP",
    "TokenRoutedMLPParallel",
    "SparseMoE",
    "SparseMoEParallel",
    "SparseMoEConfig",
    "MLPBase",
    "MLPConfig",

    # ========== Position ==========
    "RoPE",
    "YaRN",
    "ALiBi",
    "StandardRoPE",
    "YaRNRoPE",
    "DynamicNTKRoPE",
    "ALiBiPositionBias",
    "LearnedPositionEmbedding",
    "RotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",

    # ========== Normalization ==========
    "RMSNorm",
    "LayerNorm",
    "IdentityNorm",
    "build_norm",

    # ========== Block & Model ==========
    "TransformerBlock",
    "ComplexityModel",
    "ModelConfig",

    # ========== INL Architectures (O(N)) ==========
    "Mamba",
    "MambaBlock",
    "MambaConfig",
    "RWKV",
    "RWKVBlock",
    "RWKVConfig",
    "RetNet",
    "RetNetBlock",
    "RetNetConfig",
    "MoDBlock",
    "MoDConfig",

    # ========== Registry ==========
    "Registry",
    "register",
    "ATTENTION_REGISTRY",
    "MLP_REGISTRY",
    "NORMALIZATION_REGISTRY",
    "POSITION_REGISTRY",
    "MODEL_REGISTRY",
    "register_attention",
    "register_mlp",
    "register_normalization",
    "register_position",
    "register_model",

    # ========== CUDA / Triton ==========
    "CUDA",
    "Triton",
    "FlashAttention",
    "SlidingWindowAttention",
    "SparseAttention",
    "LinearAttention",
    "MultiScaleAttention",

    # ========== Helpers ==========
    "Helpers",
    "Init",
    "Mask",
    "KVCache",
    "SlidingWindowCache",
    "Sampling",
    "Tensors",
    "Debug",
    "Checkpointing",

    # ========== Efficient (Small Budget) ==========
    "Efficient",
    "Quantize",
    "QuantConfig",
    "QuantizedLinear",
    "MixedPrecision",
    "MemoryEfficient",
    "SmallModels",

    # ========== INL Dynamics ==========
    "INLDynamics",
    "INLDynamicsLite",
    "DynamicsConfig",

    # ========== Inference ==========
    "Generate",
    "GenerationConfig",
    "InferenceEngine",
    "InferenceConfig",
    "DecodingStrategy",
    "create_engine",

    # ========== Multimodal - Factories ==========
    "Vision",
    "Audio",
    "Fusion",

    # ========== Vision ==========
    "VisionEncoder",
    "VisionConfig",
    "PatchEmbedding",
    "VisionTransformer",
    "CLIPVisionEncoder",
    "SigLIPEncoder",

    # ========== Audio ==========
    "AudioEncoder",
    "AudioConfig",
    "MelSpectrogramEncoder",
    "WhisperEncoder",
    "AudioConvStack",

    # ========== Fusion ==========
    "MultimodalFusion",
    "FusionConfig",
    "CrossAttentionFusion",
    "GatedFusion",
    "ConcatProjection",
    "PerceiverResampler",
]
