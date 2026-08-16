"""Global regression contract for the dense shared TR-Hash SwiGLU path."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from complexity.config import ModelConfig
from complexity.core.mlp.base import MLPConfig
from complexity.core.mlp.tr_hash_engine import TRHashEngineMLP
from complexity.generative.audio.config import (
    TRHashAudioConfig,
    TRHashSpeechToTextConfig,
)
from complexity.generative.audio.encoder import AudioEncoderConfig
from complexity.generative.detection import TRHashDetectorConfig
from complexity.generative.image.config import TRHashImageConfig
from complexity.generative.sensor_fusion.config import TRHashSensorFusionConfig
from complexity.generative.video.config import TRHashVideoTowerConfig
from complexity.generative.vision_language.config import TRHashVisionLanguageConfig
from complexity.generative.vision_language.vision_tower import TRHashVisionTowerConfig
from complexity.multimodal.audio import AudioConfig, AudioTokenRoutedMLP
from complexity.multimodal.fusion import FusionConfig, FusionTokenRoutedMLP
from complexity.multimodal.video import VideoConfig, VideoTokenRoutedMLP
from complexity.multimodal.vision import VisionConfig, VisionTokenRoutedMLP
from complexity.tr_hash import TRHashEngine, TRHashEngineConfig

PRODUCTION_CONFIGS = (
    ("engine", TRHashEngineConfig(hidden_size=32, vocab_size=64)),
    ("vision tower", TRHashVisionTowerConfig()),
    ("detector", TRHashDetectorConfig()),
    ("audio encoder", AudioEncoderConfig()),
    ("speech to text", TRHashSpeechToTextConfig()),
    ("text to audio", TRHashAudioConfig()),
    ("video tower", TRHashVideoTowerConfig()),
    ("vision language", TRHashVisionLanguageConfig()),
    ("text to image", TRHashImageConfig()),
    ("sensor fusion", TRHashSensorFusionConfig()),
)

TEXT_DECODER_CONFIG = ModelConfig(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    mlp_type="tr_hash_engine",
    num_experts=4,
    top_k=2,
)


def _shared_width_cases():
    for architecture, config in PRODUCTION_CONFIGS:
        for field in fields(config):
            if field.name.endswith("shared_width"):
                yield pytest.param(
                    architecture,
                    field.name,
                    getattr(config, field.name),
                    id=f"{architecture}-{field.name}",
                )


@pytest.mark.parametrize("architecture,field_name,width", tuple(_shared_width_cases()))
def test_every_production_tr_hash_path_has_dense_shared_width(
    architecture: str,
    field_name: str,
    width: int,
) -> None:
    """No production TR-Hash path may silently degrade to routed-only MoE."""

    assert width > 0, f"{architecture}.{field_name}={width} disables the dense shared SwiGLU path"


@pytest.mark.parametrize(
    "architecture,config",
    (
        (
            "text decoder",
            TEXT_DECODER_CONFIG,
        ),
        ("vision-language decoder", TRHashVisionLanguageConfig().decoder_config()),
        ("speech-to-text decoder", TRHashSpeechToTextConfig().decoder_config()),
    ),
)
def test_every_production_text_decoder_enables_shared_expert(
    architecture: str,
    config: ModelConfig,
) -> None:
    assert config.shared_expert, f"{architecture} disables the dense shared SwiGLU path"
    resolved_width = config.shared_intermediate_size or config.intermediate_size
    assert resolved_width > 0, f"{architecture} has no dense shared SwiGLU capacity"


def test_tr_hash_engine_materializes_all_shared_swiglu_projections() -> None:
    engine = TRHashEngine(
        TRHashEngineConfig(
            hidden_size=32,
            vocab_size=64,
            num_experts=4,
            top_k=2,
            shared_width=16,
            expert_width=8,
        )
    )

    assert engine.shared_gate is not None
    assert engine.shared_up is not None
    assert engine.shared_down is not None


def test_text_runtime_resolves_default_shared_width_instead_of_disabling_it() -> None:
    runtime_config = MLPConfig(
        hidden_size=32,
        intermediate_size=64,
        vocab_size=128,
        num_experts=4,
        top_k=2,
    )
    mlp = TRHashEngineMLP(runtime_config)

    assert mlp.engine.config.shared_width == runtime_config.intermediate_size
    assert mlp.engine.config.expert_width == (
        runtime_config.intermediate_size // runtime_config.num_experts
    )
    assert mlp.engine.shared_gate is not None
    assert mlp.engine.shared_up is not None
    assert mlp.engine.shared_down is not None


@pytest.mark.parametrize(
    "architecture,shared_width,expert_width,top_k",
    (
        (
            "text decoder",
            TEXT_DECODER_CONFIG.intermediate_size,
            TEXT_DECODER_CONFIG.intermediate_size // TEXT_DECODER_CONFIG.num_experts,
            TEXT_DECODER_CONFIG.top_k,
        ),
        (
            "vision-language decoder",
            TRHashVisionLanguageConfig().shared_width,
            TRHashVisionLanguageConfig().routed_width // TRHashVisionLanguageConfig().num_experts,
            TRHashVisionLanguageConfig().top_k,
        ),
        (
            "speech-to-text decoder",
            TRHashSpeechToTextConfig().shared_width,
            TRHashSpeechToTextConfig().routed_width // TRHashSpeechToTextConfig().num_experts,
            TRHashSpeechToTextConfig().top_k,
        ),
        (
            "engine",
            TRHashEngineConfig(hidden_size=32, vocab_size=64).shared_width,
            TRHashEngineConfig(hidden_size=32, vocab_size=64).expert_width,
            TRHashEngineConfig(hidden_size=32, vocab_size=64).top_k,
        ),
        (
            "vision tower",
            TRHashVisionTowerConfig().shared_width,
            TRHashVisionTowerConfig().expert_width,
            TRHashVisionTowerConfig().top_k,
        ),
        (
            "detector vision tower",
            TRHashDetectorConfig().vision_shared_width,
            TRHashDetectorConfig().vision_expert_width,
            TRHashDetectorConfig().vision_top_k,
        ),
        (
            "audio encoder",
            AudioEncoderConfig().shared_width,
            AudioEncoderConfig().expert_width,
            AudioEncoderConfig().top_k,
        ),
        (
            "speech-to-text audio tower",
            TRHashSpeechToTextConfig().audio_shared_width,
            TRHashSpeechToTextConfig().audio_expert_width,
            TRHashSpeechToTextConfig().audio_top_k,
        ),
        (
            "video tower",
            TRHashVideoTowerConfig().shared_width,
            TRHashVideoTowerConfig().expert_width,
            TRHashVideoTowerConfig().top_k,
        ),
        (
            "vision-language tower",
            TRHashVisionLanguageConfig().vision_shared_width,
            TRHashVisionLanguageConfig().vision_expert_width,
            TRHashVisionLanguageConfig().vision_top_k,
        ),
        (
            "sensor-fusion core",
            TRHashSensorFusionConfig().shared_width,
            TRHashSensorFusionConfig().expert_width,
            TRHashSensorFusionConfig().top_k,
        ),
        (
            "sensor-fusion vision tower",
            TRHashSensorFusionConfig().vision_shared_width,
            TRHashSensorFusionConfig().vision_expert_width,
            TRHashSensorFusionConfig().vision_top_k,
        ),
        (
            "sensor-fusion class hash",
            TRHashSensorFusionConfig().class_hash_shared_width,
            TRHashSensorFusionConfig().class_hash_expert_width,
            TRHashSensorFusionConfig().top_k,
        ),
        (
            "text-to-audio",
            TRHashAudioConfig().shared_width,
            TRHashAudioConfig().expert_width,
            TRHashAudioConfig().top_k,
        ),
        (
            "text-to-image",
            TRHashImageConfig().shared_width,
            TRHashImageConfig().expert_width,
            TRHashImageConfig().top_k,
        ),
        (
            "legacy multimodal vision",
            VisionConfig(intermediate_size=64).intermediate_size,
            VisionConfig(intermediate_size=64).intermediate_size
            // VisionConfig(intermediate_size=64).num_experts,
            1,
        ),
        (
            "legacy multimodal audio",
            AudioConfig(intermediate_size=64).intermediate_size,
            AudioConfig(intermediate_size=64).intermediate_size
            // AudioConfig(intermediate_size=64).num_experts,
            1,
        ),
        (
            "legacy multimodal video",
            VideoConfig(intermediate_size=64).intermediate_size,
            VideoConfig(intermediate_size=64).intermediate_size
            // VideoConfig(intermediate_size=64).num_experts,
            1,
        ),
        (
            "legacy multimodal fusion",
            FusionConfig(hidden_size=16).hidden_size * 4,
            (FusionConfig(hidden_size=16).hidden_size * 4)
            // FusionConfig(hidden_size=16).num_experts,
            1,
        ),
    ),
)
def test_production_routed_experts_are_smaller_than_the_shared_path(
    architecture: str,
    shared_width: int,
    expert_width: int,
    top_k: int,
) -> None:
    assert shared_width > expert_width, (
        f"{architecture} uses expert_width={expert_width} with "
        f"shared_width={shared_width}; routed experts must remain residual capacity"
    )
    assert expert_width <= 0.60 * shared_width, (
        f"{architecture} expert width grew beyond 60% of the shared path: "
        f"expert_width={expert_width}, shared_width={shared_width}"
    )
    assert top_k * expert_width <= 1.25 * shared_width, (
        f"{architecture} active routed capacity dominates the shared path: "
        f"top_k={top_k}, expert_width={expert_width}, shared_width={shared_width}"
    )


@pytest.mark.parametrize(
    "mlp",
    (
        VisionTokenRoutedMLP(VisionConfig(hidden_size=16, intermediate_size=64, num_experts=4)),
        AudioTokenRoutedMLP(AudioConfig(hidden_size=16, intermediate_size=64, num_experts=4)),
        VideoTokenRoutedMLP(VideoConfig(hidden_size=16, intermediate_size=64, num_experts=4)),
        FusionTokenRoutedMLP(FusionConfig(hidden_size=16, num_attention_heads=4, num_experts=4)),
    ),
)
def test_legacy_multimodal_routed_mlp_keeps_shared_swiglu_and_gradients(mlp) -> None:
    values = torch.randn(2, 4, 16)
    output = mlp(values, torch.arange(4))
    output.square().mean().backward()

    assert output.shape == values.shape
    assert mlp.shared_gate_up_proj.weight.grad is not None
    assert mlp.shared_down_proj.weight.grad is not None
