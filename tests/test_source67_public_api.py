"""Small public API regressions; no training, model download or quality scoring."""

import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from complexity.api.model import Model
from complexity.api.multimodal import Audio, Fusion, Robot, Video, Vision
from complexity.config import ModelConfig


def _small_config():
    return ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=16,
    )


def test_unversioned_public_checkpoint_load_keeps_legacy_architecture(tmp_path):
    from complexity.models import ComplexityModel

    raw = {
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "max_position_embeddings": 16,
    }
    reference = ComplexityModel(ModelConfig.from_dict(raw)).eval()
    (tmp_path / "config.json").write_text(json.dumps(raw))
    torch.save(reference.state_dict(), tmp_path / "model.pt")
    loaded = Model.load(str(tmp_path)).eval()
    assert loaded.config.architecture_profile == "legacy"
    assert loaded.config.num_experts == loaded.config.top_k == 1
    assert loaded.config.routing_strategy == "modulo_cyclic"
    ids = torch.tensor([[1, 3, 12, 5]])
    torch.testing.assert_close(loaded(ids)["logits"], reference(ids)["logits"], rtol=0, atol=0)


def test_create_revalidates_a_copy_and_preserves_explicit_widths():
    original = _small_config()
    snapshot = original.to_dict()
    created = Model.create(config=original, num_attention_heads=8)
    assert created.config is not original
    assert original.to_dict() == snapshot
    assert created.config.num_attention_heads == 8
    assert created.config.head_dim == 2
    assert created.config.intermediate_size == original.intermediate_size
    assert created.config.shared_intermediate_size == original.shared_intermediate_size
    assert created.config.architecture_profile == "source67"


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"num_attention_heads": 3}, ValueError),
        ({"num_experts": 8}, ValueError),
        ({"unknown_option": True}, TypeError),
    ],
)
def test_create_invalid_override_fails_before_model_and_leaves_config_intact(
    monkeypatch, overrides, error
):
    import complexity.api.model as api_model

    original = _small_config()
    snapshot = original.to_dict()
    monkeypatch.setattr(
        api_model,
        "ComplexityModel",
        lambda _config: pytest.fail("invalid config reached construction"),
    )
    with pytest.raises(error):
        Model.create(config=original, **overrides)
    assert original.to_dict() == snapshot


def test_preset_overrides_are_validated_without_mutating_preset(monkeypatch):
    import complexity.api.model as api_model

    original = _small_config()
    snapshot = original.to_dict()
    monkeypatch.setitem(api_model.PRESET_CONFIGS, "source67-api-test", lambda: original)
    result = Model.load("source67-api-test", num_attention_heads=8)
    assert result.config.num_attention_heads == 8
    assert result.config.head_dim == 2
    assert result.config is not original
    with pytest.raises(ValueError, match="divisible"):
        Model.load("source67-api-test", num_attention_heads=3)
    assert original.to_dict() == snapshot


def test_public_save_uses_canonical_versioned_config(tmp_path):
    model = Model.create(config=_small_config())
    model.config._runtime_private = "excluded"
    model.save(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["architecture_profile"] == "source67"
    assert saved["architecture_profile_version"] == 1
    assert "_runtime_private" not in saved
    restored = Model.load(str(tmp_path))
    assert restored.config.to_dict() == model.config.to_dict()


@pytest.mark.parametrize("kind", ["encoder", "transformer"])
def test_vision_public_factory_constructs_source67_with_real_arguments(kind):
    if kind == "encoder":
        model = Vision.encoder(
            image_size=8, patch_size=4, hidden_size=16, num_layers=1, num_heads=4
        )
        assert model.encoder.config.architecture_profile == "source67"
    else:
        model = Vision.create(
            "transformer", image_size=8, patch_size=4, hidden_size=16, num_layers=1, num_heads=4
        )
        assert model.config.architecture_profile == "source67"
    output = model(torch.randn(1, 3, 8, 8))
    hidden = output["last_hidden_state"] if isinstance(output, dict) else output
    assert hidden.shape[-1] == 16


@pytest.mark.parametrize("kind", ["standard", "whisper", "mel"])
def test_audio_public_factory_uses_wrapper_or_config_as_required(kind):
    kwargs = {"n_mels": 8, "hidden_size": 16, "num_layers": 1, "num_heads": 4}
    if kind != "standard":
        kwargs["max_length"] = 16
    model = Audio.create(kind, **kwargs)
    output = model(torch.randn(1, 8, 8))
    hidden = output["last_hidden_state"] if isinstance(output, dict) else output
    assert hidden.shape[-1] == 16


@pytest.mark.parametrize("kind", ["cross_attention", "multimodal"])
def test_fusion_public_factory_keeps_source67_mlp_and_cross_attention_contract(kind):
    model = Fusion.create(
        kind, hidden_size=16, num_attention_heads=4, num_hidden_layers=1, dropout=0
    )
    hidden = model(torch.randn(1, 4, 16), torch.randn(1, 3, 16))
    assert hidden.shape == (1, 4, 16)
    core = model if kind == "cross_attention" else model.fusion
    assert core.layers[0].mlp.engine.config.baseline_profile == "source67"


@pytest.mark.parametrize("profile,kv", [("source67", 2), ("legacy", 4)])
def test_video_public_factory_exposes_profile_and_kv_count(profile, kv):
    model = Video.encoder(
        image_size=8,
        patch_size=4,
        num_frames=4,
        temporal_patch_size=2,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        architecture_profile=profile,
        num_key_value_heads=kv,
    )
    assert model.encoder.config.architecture_profile == profile
    assert model.encoder.config.num_key_value_heads == kv
    assert model(torch.randn(1, 3, 4, 8, 8)).shape == (1, 16)


def test_factory_alias_conflicts_are_rejected():
    with pytest.raises(ValueError, match="conflicting factory arguments"):
        Vision.create("vit", num_layers=1, num_hidden_layers=2)
    with pytest.raises(ValueError, match="conflicting factory arguments"):
        Audio.create("mel", num_layers=1, num_hidden_layers=2)


def test_robot_public_defaults_follow_source67_and_explicit_legacy():
    assert Robot.config().num_experts == 4
    assert Robot.config().architecture_profile == "source67"
    assert Robot.config(architecture_profile="legacy").num_experts == 8


def test_specialized_coco_default_architecture_matches_the_supported_v8_parser():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "DRY_RUN": "1", "VENV_ACTIVATE": "/dev/null", "REPO_ROOT": str(root)}
    env.pop("ARCHITECTURE_VERSION", None)
    env.pop("ARCHITECTURE_PROFILE", None)
    env.pop("KEY_VALUE_HEADS", None)
    result = subprocess.run(
        ["bash", "scripts/vast_train_detector_specialized_coco.sh"],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--architecture-version 8" in result.stdout
    assert "--architecture-profile source67" in result.stdout
    assert "--key-value-heads 1" in result.stdout
