"""Global regression contract: TR-Hash route tables must be deterministic.

Routing is supposed to be config-derived only (``route_seed`` / ``layer_index``
/ ``vocab_size`` / expert count) -- never learned, never dependent on global
random state. This file builds every production TR-Hash path twice, under
different global RNG seeds, and checks that every route table it exposes
comes out bit-identical both times. It also checks that routing genuinely
spreads across more than one expert, so a broken hash that silently always
resolves to the same expert would fail here too.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from complexity.generative.audio import (
    TRHashAudioConfig,
    TRHashSpeechToText,
    TRHashSpeechToTextConfig,
    TRHashTextToSpeech,
)
from complexity.generative.audio.encoder import AudioEncoder, AudioEncoderConfig
from complexity.generative.detection import TRHashDetectorConfig, TRHashObjectDetector
from complexity.generative.image import TRHashImageConfig, TRHashTextToImage
from complexity.generative.sensor_fusion import (
    TRHashSensorFusionClassifier,
    TRHashSensorFusionConfig,
)
from complexity.generative.video import TRHashVideoTower, TRHashVideoTowerConfig
from complexity.generative.vision_language import (
    TRHashImageTextToText,
    TRHashVisionLanguageConfig,
)
from complexity.generative.vision_language.vision_tower import (
    TRHashVisionTower,
    TRHashVisionTowerConfig,
)
from complexity.tr_hash import TRHashEngine, TRHashEngineConfig


def _engine_factory() -> nn.Module:
    return TRHashEngine(TRHashEngineConfig(hidden_size=32, vocab_size=64))


def _vision_tower_factory() -> nn.Module:
    return TRHashVisionTower(
        TRHashVisionTowerConfig(
            image_size=32,
            patch_size=8,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=4,
            top_k=2,
            expert_width=16,
        )
    )


def _detector_factory() -> nn.Module:
    return TRHashObjectDetector(
        TRHashDetectorConfig(
            image_size=32,
            patch_size=8,
            vision_hidden_size=32,
            vision_layers=3,
            vision_stage_depths=(1, 1, 1),
            vision_window_size=2,
            vision_heads=4,
            vision_num_experts=4,
            vision_top_k=2,
            vision_expert_width=16,
            num_classes=5,
        )
    )


def _audio_encoder_factory() -> nn.Module:
    return AudioEncoder(
        AudioEncoderConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=4,
            top_k=2,
            expert_width=16,
        )
    )


def _speech_to_text_factory() -> nn.Module:
    return TRHashSpeechToText(
        TRHashSpeechToTextConfig(
            sample_rate=16_000,
            n_mels=16,
            n_fft=128,
            hop_length=64,
            audio_hidden_size=32,
            audio_layers=1,
            audio_heads=4,
            num_audio_tokens=4,
            vocab_size=101,
            hidden_size=32,
            decoder_layers=2,
            attention_heads=4,
            key_value_heads=2,
            max_position_embeddings=64,
            num_experts=4,
            top_k=2,
            shared_width=64,
            routed_width=64,
        )
    )


def _text_to_speech_factory() -> nn.Module:
    return TRHashTextToSpeech(
        TRHashAudioConfig(
            n_mels=8,
            frame_patch_size=4,
            max_audio_frames=32,
            vocab_size=101,
            max_text_length=16,
            text_hidden_size=32,
            text_layers=1,
            text_heads=4,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_experts=4,
            top_k=2,
            shared_width=96,
            expert_width=16,
            time_buckets=8,
        )
    )


def _video_tower_factory() -> nn.Module:
    return TRHashVideoTower(
        TRHashVideoTowerConfig(
            image_size=32,
            patch_size=8,
            num_frames=8,
            temporal_patch_size=2,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=4,
            top_k=2,
            expert_width=16,
        )
    )


def _vision_language_factory() -> nn.Module:
    return TRHashImageTextToText(
        TRHashVisionLanguageConfig(
            image_size=32,
            patch_size=8,
            vision_hidden_size=32,
            vision_layers=1,
            vision_heads=4,
            num_visual_tokens=4,
            vocab_size=101,
            hidden_size=32,
            decoder_layers=2,
            attention_heads=4,
            key_value_heads=2,
            max_position_embeddings=64,
            num_experts=4,
            top_k=2,
            shared_width=64,
            routed_width=64,
        )
    )


def _text_to_image_factory() -> nn.Module:
    return TRHashTextToImage(
        TRHashImageConfig(
            image_size=32,
            vae_downsample_factor=4,
            latent_channels=4,
            latent_patch_size=2,
            vocab_size=101,
            max_text_length=16,
            text_hidden_size=32,
            text_layers=1,
            text_heads=4,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_experts=4,
            top_k=2,
            shared_width=96,
            expert_width=16,
            time_buckets=8,
        )
    )


def _sensor_fusion_factory() -> nn.Module:
    return TRHashSensorFusionClassifier(
        TRHashSensorFusionConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            top_k=2,
            expert_width=16,
            precision="fp32",
            classifier_dropout=0.0,
            visual_channels=(3, 1, 3),
            visual_token_grid=(2, 2, 2),
            vision_image_size=16,
            vision_patch_size=4,
            vision_hidden_size=32,
            vision_layers=2,
            vision_heads=4,
            vision_num_experts=4,
            vision_top_k=2,
            vision_expert_width=8,
            vision_stage_depths=(1, 1),
            vision_window_size=2,
            radar_features=5,
            sequence_tokens=4,
        )
    )


PRODUCTION_FACTORIES = (
    ("engine", _engine_factory),
    ("vision tower", _vision_tower_factory),
    ("detector", _detector_factory),
    ("audio encoder", _audio_encoder_factory),
    ("speech to text", _speech_to_text_factory),
    ("text to speech", _text_to_speech_factory),
    ("video tower", _video_tower_factory),
    ("vision language", _vision_language_factory),
    ("text to image", _text_to_image_factory),
    ("sensor fusion", _sensor_fusion_factory),
)


def _route_tables(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: submodule.route_table.clone()
        for name, submodule in model.named_modules()
        if torch.is_tensor(getattr(submodule, "route_table", None))
    }


@pytest.mark.parametrize("architecture,factory", PRODUCTION_FACTORIES)
def test_route_tables_are_deterministic_across_independent_constructions(
    architecture: str, factory
) -> None:
    torch.manual_seed(0)
    first = _route_tables(factory())
    assert first, f"{architecture}: no TR-Hash route_table found -- factory is stale"

    torch.manual_seed(999_983)
    second = _route_tables(factory())

    assert set(first) == set(second), f"{architecture}: route table module set changed"
    for key in first:
        assert torch.equal(first[key], second[key]), (
            f"{architecture}.{key}: route table changed across independent "
            "constructions built under different global RNG seeds -- routing "
            "must depend only on config (route_seed/layer_index/vocab_size), "
            "never on global randomness"
        )


@pytest.mark.parametrize("architecture,factory", PRODUCTION_FACTORIES)
def test_route_tables_genuinely_use_more_than_one_expert(architecture: str, factory) -> None:
    tables = _route_tables(factory())
    for key, table in tables.items():
        assert torch.unique(table).numel() > 1, (
            f"{architecture}.{key}: route table only ever selects one expert -- "
            "the hash has degenerated to a dense pass-through instead of real "
            "position-keyed routing"
        )
