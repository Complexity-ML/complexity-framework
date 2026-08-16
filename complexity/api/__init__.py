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
        GQA, RoPE, RMSNorm, TokenRoutedMLP,
        # CUDA / Triton optimizations
        CUDA, Triton, FlashAttention, SlidingWindowAttention,
        # Inference
        Generate, GenerationConfig,
        # Multimodal
        Vision, Audio, Video, Fusion, Robot, Omni,
        # Registry pour custom
        register,
    )

    # Construire son propre modèle
    class MonModele(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.attn = GQA(config.hidden_size, config.num_heads, config.kv_heads)
            self.mlp = TokenRoutedMLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, num_experts=8)
            self.norm = RMSNorm(config.hidden_size)
            self.rope = RoPE(config.head_dim, config.max_seq_len)

    # Ou via factories
    attn = Attention.gqa(hidden_size=4096, num_heads=32, kv_heads=8)
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

# =============================================================================
# Building Blocks (modèles maison) - depuis core/
# =============================================================================
from .core import (
    ATTENTION_REGISTRY,
    # CUDA / Triton Optimizations
    CUDA,
    # Attention
    GQA,
    MHA,
    MLP,
    MLP_REGISTRY,
    MODEL_REGISTRY,
    MQA,
    NORMALIZATION_REGISTRY,
    POSITION_REGISTRY,
    ALiBi,
    ALiBiPositionBias,
    # Factories
    Attention,
    AttentionBase,
    AttentionConfig,
    Block,
    Checkpointing,
    ComplexityModel,
    Debug,
    DynamicNTKRoPE,
    # Efficient (small budget)
    Efficient,
    FlashAttention,
    GroupedQueryAttention,
    # Helpers
    Helpers,
    IdentityNorm,
    Init,
    KVCache,
    LayerNorm,
    LearnedPositionEmbedding,
    LinearAttention,
    Mask,
    MemoryEfficient,
    MixedPrecision,
    MLPBase,
    MLPConfig,
    ModelConfig,
    MultiHeadAttention,
    MultiQueryAttention,
    MultiScaleAttention,
    Norm,
    Position,
    QuantConfig,
    Quantize,
    QuantizedLinear,
    # Registry
    Registry,
    # Normalization
    RMSNorm,
    # Position
    RoPE,
    RotaryEmbedding,
    Sampling,
    SlidingWindowAttention,
    SlidingWindowCache,
    SmallModels,
    SparseAttention,
    StandardRoPE,
    Tensors,
    # MLP
    TokenRoutedMLP,
    # Block & Model
    TransformerBlock,
    Triton,
    YaRN,
    YaRNRoPE,
    apply_rotary_pos_emb,
    build_norm,
    register,
    register_attention,
    register_mlp,
    register_model,
    register_normalization,
    register_position,
    rotate_half,
)
from .dataset import DataConfig, DataPipeline, Dataset, StreamingDataset

# =============================================================================
# Inference & Generation
# =============================================================================
from .inference import (
    DecodingStrategy,
    Generate,
    GenerationConfig,
    InferenceConfig,
    InferenceEngine,
    create_engine,
)
from .model import Model

# =============================================================================
# Multimodal (Vision, Audio)
# =============================================================================
from .multimodal import (
    Audio,
    AudioConfig,
    AudioConvStack,
    # Audio
    AudioEncoder,
    CLIPVisionEncoder,
    ConcatProjection,
    CrossAttentionFusion,
    Fusion,
    FusionConfig,
    GatedFusion,
    MelSpectrogramEncoder,
    # Fusion
    MultimodalFusion,
    Omni,
    PatchEmbedding,
    PerceiverResampler,
    Robot,
    SigLIPEncoder,
    # Robotics
    TRHashSensorFusionClassifier,
    TRHashSensorFusionConfig,
    Video,
    # Factories
    Vision,
    VisionConfig,
    # Vision
    VisionEncoder,
    VisionTransformer,
    WhisperEncoder,
)
from .tokenizer import Tokenizer, TokenizerConfig
from .trainer import Trainer, TrainerConfig

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
    "TokenRoutedMLP",
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
    "Video",
    "Fusion",
    "Robot",
    "Omni",
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
    # ========== TR-Hash Robotics ==========
    "TRHashSensorFusionClassifier",
    "TRHashSensorFusionConfig",
]
