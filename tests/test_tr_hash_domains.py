"""Bounded numerical/domain checks; these are not vision accuracy results."""

import math
import pytest
import torch
from torch.nn import functional as F
from complexity.tr_hash import TRHashEngine, TRHashStrategy
from complexity.tr_hash.attention import DomainSelfAttention
from complexity.vision.video.config import TRHashVideoTowerConfig
from complexity.vision.video.tower import TRHashVideoTower
from complexity.vision.vision_language.vision_tower import (
    TRHashVisionTower,
    TRHashVisionTowerConfig,
)
from complexity.vision.vision_language.config import TRHashVisionLanguageConfig
from complexity.vision.image.config import TRHashImageConfig
from complexity.vision.image.model import TRHashTextToImage
from complexity.vision.image.editing import TRHashImageEditConfig
from complexity.vision.detection.config import TRHashDetectorConfig
from complexity.vision.detection.model import TRHashObjectDetector
from complexity.generative.audio.encoder import AudioEncoderConfig, AudioEncoder
from complexity.generative.audio.config import TRHashAudioConfig, TRHashSpeechToTextConfig
from complexity.generative.sensor_fusion.config import TRHashSensorFusionConfig
from complexity.generative.sensor_fusion.model import TRHashSensorFusionClassifier
from complexity.multimodal.vision import VisionConfig, VisionTransformer
from complexity.multimodal.video import VideoConfig, VideoTransformer
from complexity.multimodal.audio import AudioConfig, MelSpectrogramEncoder
from complexity.multimodal.fusion import FusionConfig, CrossAttentionFusion, PerceiverResampler


def test_domain_attention_matches_independent_canonical_head1_math():
    torch.manual_seed(71)
    attention = DomainSelfAttention(16, 4, layer_index=1).eval()
    x = torch.randn(2, 5, 16)
    ids = torch.tensor([[0, 12, 24, 36, 48], [35, 7, 13, 47, 26]])
    q_w, k_w, v_w = attention.qkv.weight.split((16, 4, 4))
    q = attention.q_norm(F.linear(x, q_w).view(2, 5, 4, 4).transpose(1, 2))
    k = attention.k_norm(F.linear(x, k_w).view(2, 5, 1, 4).transpose(1, 2))
    v = F.linear(x, v_w).view(2, 5, 1, 4).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1))
    primary = ids // 12 % 4
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
    secondary = (primary + 1 + ids // 4 % 3) % 4
    mask = F.one_hot(primary, 4) + F.one_hot(secondary, 4)
    expected = attention.out(
        (y * mask.permute(0, 2, 1).unsqueeze(-1) * math.sqrt(2)).transpose(1, 2).reshape(1, 5, 16)
    )
    torch.testing.assert_close(attention(x, ids), expected)


def _assert_canonical_engines(model):
    engines = [module for module in model.modules() if isinstance(module, TRHashEngine)]
    assert engines
    for engine in engines:
        config = engine.config
        assert config.routing_strategy is TRHashStrategy.COMPOSED_MODULO
        assert config.num_experts == 4 and config.top_k == 2
        assert config.route_weights == (0.5, 0.5)
        assert config.shared_output_scale == 1 and config.routed_output_scale == 2
    return engines


def test_all_canonical_domain_sites_use_the_common_engine_and_backpropagate():
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
        engines = _assert_canonical_engines(model)
        output = model(*inputs)
        output.square().mean().backward()
        assert all((engine.expert_down.grad is not None for engine in engines))


def test_canonical_sensor_fusion_preserves_modality_routes_and_all_three_engine_sites():
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
    engines = _assert_canonical_engines(model)
    assert torch.equal(model.route_ids, torch.arange(config.route_vocab_size))
    result = model({"skeleton": torch.randn(1, 4, 17, 3), "depth": torch.randn(1, 3, 2, 16, 16)})
    assert result["logits"].shape == (1, 4)
    result["logits"].square().mean().backward()
    assert all((engine.expert_down.grad is not None for engine in engines))


def test_multimodal_wrappers_keep_raw_ids_and_canonical_mlp_backbones():
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
        engines = _assert_canonical_engines(model)
        output = model(*args)
        hidden = output["last_hidden_state"] if isinstance(output, dict) else output
        hidden.square().mean().backward()
        assert all((engine.expert_down.grad is not None for engine in engines))


def test_image_parameter_estimate_follows_gqa_and_contextual_text_encoder():
    config = TRHashImageConfig(
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


CONFIG_CLASSES = [
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
]


@pytest.mark.parametrize("config_class", CONFIG_CLASSES)
def test_domain_configuration_roundtrip_and_canonical_defaults(config_class):
    config = config_class()
    assert config_class.from_dict(config.to_dict()) == config
    assert not any(
        key in config.to_dict()
        for key in ["architecture_profile", "architecture_profile_version", "route_seed"]
    )
    for name in [
        "num_attention_heads",
        "vision_heads",
        "text_heads",
        "audio_heads",
        "attention_heads",
    ]:
        if hasattr(config, name):
            assert getattr(config, name) == 4
    for name in [
        "num_key_value_heads",
        "vision_key_value_heads",
        "text_key_value_heads",
        "audio_key_value_heads",
        "key_value_heads",
    ]:
        if hasattr(config, name):
            assert getattr(config, name) == 1
    if hasattr(config, "num_experts"):
        assert config.num_experts == 4
    if hasattr(config, "top_k"):
        assert config.top_k == 2


@pytest.mark.parametrize("config_class", CONFIG_CLASSES)
@pytest.mark.parametrize(
    "field", ["architecture_profile", "architecture_profile_version", "route_seed"]
)
def test_removed_architecture_options_are_rejected(config_class, field):
    values = config_class().to_dict()
    values[field] = "retired"
    with pytest.raises((TypeError, ValueError)):
        config_class.from_dict(values)


@pytest.mark.parametrize("heads,kv", [(2, 1), (8, 1), (4, 2)])
def test_domain_attention_rejects_noncanonical_head_layout(heads, kv):
    with pytest.raises(ValueError, match="four query heads"):
        DomainSelfAttention(16, heads, num_key_value_heads=kv)


def test_canonical_attention_state_and_strict_checkpoint_roundtrip():
    attention = DomainSelfAttention(16, 4).eval()
    assert set(attention.state_dict()) == {
        "qkv.weight",
        "out.weight",
        "q_norm.weight",
        "k_norm.weight",
    }
    assert attention.qkv.weight.shape == (24, 16)
    restored = DomainSelfAttention(16, 4).eval()
    restored.load_state_dict(attention.state_dict(), strict=True)
    x = torch.randn(2, 5, 16)
    ids = torch.arange(5).expand(2, -1)
    torch.testing.assert_close(restored(x, ids), attention(x, ids), rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        restored.load_state_dict(
            {
                "qkv.weight": torch.randn(48, 16),
                "qkv.bias": torch.randn(48),
                "out.weight": torch.randn(16, 16),
                "out.bias": torch.randn(16),
            },
            strict=True,
        )


def test_native_detector_checkpoint_loads_the_canonical_architecture(tmp_path):
    import json
    from safetensors.torch import save_file
    from complexity.vision.detection.hub import load_detector_checkpoint

    config = TRHashDetectorConfig(
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
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    save_file(
        {name: value.detach().contiguous() for name, value in original.state_dict().items()},
        str(tmp_path / "model.safetensors"),
    )
    restored = load_detector_checkpoint(tmp_path)
    assert restored.config == config
    pixels = torch.randn(1, 3, 16, 16)
    torch.testing.assert_close(restored(pixels), original(pixels), rtol=0, atol=0)


@pytest.mark.parametrize("config_class", CONFIG_CLASSES)
def test_domain_configurations_reject_other_head_and_expert_layouts(config_class):
    values = config_class().to_dict()
    for name in [
        "num_attention_heads",
        "vision_heads",
        "text_heads",
        "audio_heads",
        "attention_heads",
    ]:
        if name in values:
            invalid = dict(values, **{name: 8})
            with pytest.raises(ValueError):
                config_class.from_dict(invalid)
    for name in [
        "num_key_value_heads",
        "vision_key_value_heads",
        "text_key_value_heads",
        "audio_key_value_heads",
        "key_value_heads",
    ]:
        if name in values:
            with pytest.raises(ValueError):
                config_class.from_dict(dict(values, **{name: 2}))
    for name in ["num_experts", "vision_num_experts", "audio_num_experts"]:
        if name in values:
            with pytest.raises(ValueError):
                config_class.from_dict(dict(values, **{name: 8}))


def test_public_video_factory_uses_canonical_attention_and_raw_ids():
    from complexity.api.multimodal import Video

    model = Video.encoder(
        image_size=8,
        patch_size=4,
        num_frames=4,
        temporal_patch_size=2,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        num_key_value_heads=1,
    )
    _assert_canonical_engines(model)
    result = model(torch.randn(1, 3, 4, 8, 8))
    output = result["last_hidden_state"] if isinstance(result, dict) else result
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError, match="four query heads"):
        Video.encoder(
            image_size=8,
            patch_size=4,
            num_frames=4,
            temporal_patch_size=2,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            num_key_value_heads=2,
        )
