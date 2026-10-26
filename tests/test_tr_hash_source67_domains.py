"""Bounded numerical/domain checks; these are not vision accuracy results."""

from dataclasses import asdict
import math

import pytest
import torch
from torch.nn import functional as F

from complexity.tr_hash import TRHashEngine, TRHashStrategy
from complexity.tr_hash.attention import DomainSelfAttention
from complexity.generative.video.config import TRHashVideoTowerConfig
from complexity.generative.video.tower import TRHashVideoTower
from complexity.generative.vision_language.vision_tower import (
    TRHashVisionTower,
    TRHashVisionTowerConfig,
)
from complexity.generative.vision_language.config import TRHashVisionLanguageConfig
from complexity.generative.image.config import TRHashImageConfig
from complexity.generative.image.model import TRHashTextToImage
from complexity.generative.image.editing import TRHashImageEditConfig
from complexity.generative.detection.config import TRHashDetectorConfig
from complexity.generative.detection.model import TRHashObjectDetector
from complexity.generative.audio.encoder import AudioEncoderConfig, AudioEncoder
from complexity.generative.audio.config import TRHashAudioConfig, TRHashSpeechToTextConfig
from complexity.generative.sensor_fusion.config import TRHashSensorFusionConfig
from complexity.generative.sensor_fusion.model import TRHashSensorFusionClassifier
from complexity.multimodal.vision import VisionConfig, VisionTransformer
from complexity.multimodal.video import VideoConfig, VideoTransformer
from complexity.multimodal.audio import AudioConfig, MelSpectrogramEncoder
from complexity.multimodal.fusion import FusionConfig, CrossAttentionFusion, PerceiverResampler


def test_domain_attention_matches_independent_source67_head1_math():
    torch.manual_seed(71)
    attention = DomainSelfAttention(16, 4, layer_index=1).eval()
    x = torch.randn(2, 5, 16)
    ids = torch.tensor([[0, 12, 24, 36, 48], [35, 7, 13, 47, 26]])
    q_w, k_w, v_w = attention.qkv.weight.split((16, 4, 4))
    q = attention.q_norm(F.linear(x, q_w).view(2, 5, 4, 4).transpose(1, 2))
    k = attention.k_norm(F.linear(x, k_w).view(2, 5, 1, 4).transpose(1, 2))
    v = F.linear(x, v_w).view(2, 5, 1, 4).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1))
    primary = (ids // 12) % 4
    mask = F.one_hot(primary, 4) + F.one_hot((primary + 1) % 4, 4)
    y = y * mask.permute(0, 2, 1).unsqueeze(-1) * math.sqrt(2)
    expected = attention.out(y.transpose(1, 2).reshape(2, 5, 16))
    torch.testing.assert_close(attention(x, ids), expected)


def test_domain_attention_v0_binds_values_only_and_preserves_original_ids():
    torch.manual_seed(72)
    attention = DomainSelfAttention(16, 4, layer_index=0).eval()
    x = torch.randn(1, 5, 16)
    ids = torch.tensor([[1, 12, 2, 35, 4]])
    q_w, k_w, v_w = attention.qkv.weight.split((16, 4, 4))
    indices = (torch.arange(16)[None, None] + 4 * (ids % 4).unsqueeze(-1)) % 16
    bound = x.gather(-1, indices)
    q = attention.q_norm(F.linear(x, q_w).view(1, 5, 4, 4).transpose(1, 2))
    k = attention.k_norm(F.linear(x, k_w).view(1, 5, 1, 4).transpose(1, 2))
    v = F.linear(bound, v_w).view(1, 5, 1, 4).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1))
    primary = ids % 4
    secondary = (primary + 1 + (ids // 4) % 3) % 4
    mask = F.one_hot(primary, 4) + F.one_hot(secondary, 4)
    expected = attention.out(
        (y * mask.permute(0, 2, 1).unsqueeze(-1) * math.sqrt(2)).transpose(1, 2).reshape(1, 5, 16)
    )
    torch.testing.assert_close(attention(x, ids), expected)


def test_legacy_attention_retains_original_mha_state_and_math():
    torch.manual_seed(73)
    attention = DomainSelfAttention(16, 4, architecture_profile="legacy").eval()
    assert set(attention.state_dict()) == {"qkv.weight", "qkv.bias", "out.weight", "out.bias"}
    x = torch.randn(2, 7, 16)
    q, k, v = attention.qkv(x).view(2, 7, 3, 4, 4).unbind(2)
    q, k, v = (value.transpose(1, 2) for value in (q, k, v))
    expected = attention.out(
        F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 7, 16)
    )
    torch.testing.assert_close(attention(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "config_class",
    [
        TRHashVideoTowerConfig,
        TRHashVisionTowerConfig,
        TRHashImageConfig,
        TRHashImageEditConfig,
        TRHashDetectorConfig,
        AudioEncoderConfig,
        TRHashAudioConfig,
        TRHashSpeechToTextConfig,
        TRHashVisionLanguageConfig,
        TRHashSensorFusionConfig,
        VisionConfig,
        VideoConfig,
        AudioConfig,
        FusionConfig,
    ],
)
def test_domain_profile_serialization_and_missing_profile_legacy(config_class):
    current = config_class()
    assert current.architecture_profile == "source67"
    assert current.architecture_profile_version == 1
    assert config_class.from_dict(current.to_dict()) == current
    old = config_class.from_dict(
        {"architecture_version": 8} if config_class is TRHashDetectorConfig else {}
    )
    assert old.architecture_profile == "legacy"
    assert old.architecture_profile_version == 1
    if hasattr(old, "num_key_value_heads"):
        assert old.num_key_value_heads == old.num_attention_heads
    if config_class is TRHashSensorFusionConfig:
        assert old.num_experts == 8
        assert current.num_experts == 4
    if hasattr(current, "num_key_value_heads"):
        assert current.num_key_value_heads == 1


def test_legacy_tower_checkpoint_load_is_strict_and_architecture_preserving():
    config = TRHashVisionTowerConfig.from_dict(
        {
            "image_size": 16,
            "patch_size": 4,
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
        }
    )
    assert (config.shared_width, config.expert_width) == (128, 48)
    before = TRHashVisionTower(config).eval()
    old_metadata = asdict(config)
    for key in ("architecture_profile", "architecture_profile_version", "num_key_value_heads"):
        old_metadata.pop(key)
    restored = TRHashVisionTower(TRHashVisionTowerConfig.from_dict(old_metadata)).eval()
    restored.load_state_dict(before.state_dict(), strict=True)
    pixels = torch.randn(1, 3, 16, 16)
    torch.testing.assert_close(restored(pixels), before(pixels), rtol=0, atol=0)
    source = TRHashVisionTower(
        TRHashVisionTowerConfig(
            image_size=16, patch_size=4, hidden_size=16, num_attention_heads=4, num_hidden_layers=2
        )
    )
    with pytest.raises(RuntimeError):
        source.load_state_dict(before.state_dict(), strict=True)


def _assert_source_engines(model):
    engines = [module for module in model.modules() if isinstance(module, TRHashEngine)]
    assert engines
    for engine in engines:
        config = engine.config
        assert config.routing_strategy is TRHashStrategy.COMPOSED_MODULO
        assert config.num_experts == 4 and config.top_k == 2
        assert config.route_weights == (0.5, 0.5)
        assert config.shared_output_scale == 1 and config.routed_output_scale == 2
    return engines


def test_all_source_domain_sites_use_the_common_engine_and_backpropagate():
    models_and_inputs = [
        (
            TRHashVisionTower(
                TRHashVisionTowerConfig(
                    image_size=16,
                    patch_size=4,
                    hidden_size=16,
                    num_attention_heads=4,
                    num_hidden_layers=2,
                )
            ),
            (torch.randn(1, 3, 16, 16),),
        ),
        (
            TRHashVideoTower(
                TRHashVideoTowerConfig(
                    image_size=8,
                    patch_size=4,
                    num_frames=4,
                    temporal_patch_size=2,
                    hidden_size=16,
                    num_attention_heads=4,
                    num_hidden_layers=2,
                )
            ),
            (torch.randn(1, 3, 4, 8, 8),),
        ),
        (
            AudioEncoder(
                AudioEncoderConfig(
                    n_mels=8,
                    n_fft=32,
                    hop_length=8,
                    hidden_size=16,
                    num_attention_heads=4,
                    num_hidden_layers=2,
                    max_frames=64,
                )
            ),
            (torch.randn(1, 128),),
        ),
        (
            TRHashObjectDetector(
                TRHashDetectorConfig(
                    image_size=32,
                    patch_size=4,
                    vision_hidden_size=16,
                    vision_heads=4,
                    vision_layers=2,
                    vision_stage_depths=(1, 1),
                    scale_factors=(1, 2),
                    num_classes=4,
                    class_level_hash_gate_enabled=True,
                    vision_window_size=2,
                )
            ),
            (torch.randn(1, 3, 24, 32),),
        ),
        (
            TRHashTextToImage(
                TRHashImageConfig(
                    image_size=16,
                    vae_downsample_factor=2,
                    latent_patch_size=2,
                    hidden_size=16,
                    num_attention_heads=4,
                    num_layers=2,
                    vocab_size=64,
                    max_text_length=8,
                    text_hidden_size=16,
                    text_heads=4,
                    text_layers=2,
                    time_buckets=4,
                )
            ),
            (
                torch.randn(1, 4, 8, 8),
                torch.tensor([0.5]),
                torch.tensor([[12, 3, 35, 7]]),
                torch.ones(1, 4, dtype=torch.bool),
            ),
        ),
    ]
    for model, inputs in models_and_inputs:
        engines = _assert_source_engines(model)
        output = model(*inputs)
        output.square().mean().backward()
        assert all(engine.expert_down.grad is not None for engine in engines)


def test_source_sensor_fusion_preserves_modality_routes_and_all_three_engine_sites():
    config = TRHashSensorFusionConfig(
        num_classes=4,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=2,
        vision_image_size=16,
        vision_patch_size=4,
        vision_hidden_size=16,
        vision_heads=4,
        vision_layers=2,
        vision_stage_depths=(1, 1),
        visual_token_grid=(1, 2, 2),
        sequence_tokens=2,
    )
    model = TRHashSensorFusionClassifier(config)
    engines = _assert_source_engines(model)
    assert torch.equal(model.route_ids, torch.arange(config.route_vocab_size))
    result = model({"skeleton": torch.randn(1, 4, 17, 3), "depth": torch.randn(1, 3, 2, 16, 16)})
    assert result["logits"].shape == (1, 4)
    result["logits"].square().mean().backward()
    assert all(engine.expert_down.grad is not None for engine in engines)


def test_multimodal_wrappers_keep_raw_ids_and_source67_mlp_backbones():
    vision = VisionTransformer(
        VisionConfig(
            image_size=16, patch_size=4, hidden_size=16, num_attention_heads=4, num_hidden_layers=2
        )
    )
    video = VideoTransformer(
        VideoConfig(
            image_size=8,
            patch_size=4,
            num_frames=4,
            temporal_patch_size=2,
            hidden_size=16,
            num_attention_heads=4,
            num_hidden_layers=2,
        )
    )
    audio = MelSpectrogramEncoder(
        AudioConfig(
            n_mels=8, max_length=32, hidden_size=16, num_attention_heads=4, num_hidden_layers=2
        )
    )
    fusion = CrossAttentionFusion(hidden_size=16, num_heads=4, num_layers=2, dropout=0)
    resampler = PerceiverResampler(
        hidden_size=16, num_heads=4, num_layers=2, num_latents=16, dropout=0
    )
    assert torch.equal(vision.expert_ids, torch.arange(17))
    assert torch.equal(video.expert_ids, torch.arange(8))
    assert torch.equal(audio.expert_ids_table, torch.arange(audio.expert_ids_table.numel()))
    assert torch.equal(resampler.expert_ids, torch.arange(16))
    for model, args in [
        (vision, (torch.randn(1, 3, 16, 16),)),
        (video, (torch.randn(1, 3, 4, 8, 8),)),
        (audio, (torch.randn(1, 8, 24),)),
        (fusion, (torch.randn(1, 16, 16), torch.randn(1, 5, 16))),
        (resampler, (torch.randn(1, 5, 16),)),
    ]:
        engines = _assert_source_engines(model)
        output = model(*args)
        hidden = output["last_hidden_state"] if isinstance(output, dict) else output
        hidden.square().mean().backward()
        assert all(engine.expert_down.grad is not None for engine in engines)


def test_image_parameter_estimate_follows_gqa_and_contextual_text_encoder():
    for profile in ("source67", "legacy"):
        config = TRHashImageConfig(
            architecture_profile=profile,
            image_size=16,
            vae_downsample_factor=2,
            latent_patch_size=2,
            hidden_size=16,
            num_attention_heads=4,
            num_layers=2,
            vocab_size=64,
            max_text_length=8,
            text_hidden_size=16,
            text_heads=4,
            text_layers=2,
            time_buckets=4,
        )
        assert TRHashTextToImage(config).num_parameters() == config.estimated_parameter_count()


def test_rectangular_detector_windows_keep_max_grid_raster_identity():
    config = TRHashDetectorConfig(
        image_size=32,
        patch_size=4,
        vision_hidden_size=16,
        vision_heads=4,
        vision_layers=2,
        vision_stage_depths=(1, 1),
        scale_factors=(1, 2),
        num_classes=4,
        vision_window_size=2,
    )
    model = TRHashObjectDetector(config)
    block = model.tower.stages[0][0]
    seen = {}
    block.mlp.register_forward_pre_hook(lambda _module, args: seen.update(mlp=args[1].clone()))
    block.attention.register_forward_pre_hook(
        lambda _module, args: seen.update(attention=args[2].clone())
    )
    result = model(torch.randn(1, 3, 32, 24))
    expected = (torch.arange(8)[:, None] * 8 + torch.arange(6)[None]).reshape(1, -1)
    assert torch.equal(seen["mlp"], expected)
    assert set(seen["attention"].flatten().tolist()) == set(expected.flatten().tolist())
    assert torch.isfinite(result).all()


def test_native_legacy_detector_checkpoint_without_profile_loads_identically(tmp_path):
    import json
    from safetensors.torch import save_file
    from complexity.generative.detection.hub import load_detector_checkpoint

    config = TRHashDetectorConfig(
        architecture_profile="legacy",
        image_size=16,
        patch_size=4,
        vision_hidden_size=16,
        vision_heads=4,
        vision_layers=2,
        vision_stage_depths=(1, 1),
        scale_factors=(1, 2),
        num_classes=4,
    )
    original = TRHashObjectDetector(config).eval()
    metadata = config.to_dict()
    for name in ("architecture_profile", "architecture_profile_version", "vision_key_value_heads"):
        metadata.pop(name)
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    save_file(
        {name: value.detach().contiguous() for name, value in original.state_dict().items()},
        str(tmp_path / "model.safetensors"),
    )
    restored = load_detector_checkpoint(tmp_path)
    assert restored.config.architecture_profile == "legacy"
    pixels = torch.randn(1, 3, 16, 16)
    torch.testing.assert_close(restored(pixels), original(pixels), rtol=0, atol=0)
