"""Configuration contracts for new authoring and historical checkpoint loading."""

from dataclasses import fields
from pathlib import Path

import pytest
import torch

from complexity import ComplexityModel, ModelConfig
from complexity.training.config import TrainingConfig


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".YML"])
def test_model_pathlike_roundtrip(suffix, tmp_path):
    config = ModelConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4)
    path = tmp_path / f"config{suffix}"
    config.save(path)
    assert ModelConfig.load(path, strict=True).to_dict() == config.to_dict()


def test_new_config_rejects_typo_but_old_hf_metadata_still_loads():
    settings = {"hidden_size": 64, "num_attention_heads": 4, "model_type": "tr_hash_moe"}
    assert ModelConfig.from_dict(settings).hidden_size == 64
    with pytest.raises(ValueError, match="Unknown model config keys: model_type"):
        ModelConfig.from_dict(settings, strict=True)
    with pytest.raises(ValueError, match="num_layers"):
        ModelConfig.from_dict({"num_layers": 3}, strict=True)


def test_strict_routing_does_not_silently_change_authored_route():
    with pytest.raises(ValueError, match="token_id_composed_modulo"):
        ModelConfig.from_dict({"routing_strategy": "token_id_multi_hahs"}, strict=True)
    with pytest.raises(ValueError, match="token_id_composed_modulo"):
        ModelConfig.from_dict({"routing_strategy": "old_unknown"})


@pytest.mark.parametrize(
    "values",
    [
        {"hidden_size": True},
        {"num_hidden_layers": 2.5},
        {"shared_output_scale": float("nan")},
        {"use_qk_norm": "false"},
    ],
)
def test_strict_model_config_rejects_ambiguous_types(values):
    with pytest.raises(ValueError, match="invalid type or non-finite"):
        ModelConfig.from_dict(values, strict=True)


@pytest.mark.parametrize("value", [None, [], 42, "text"])
def test_non_mapping_config_has_actionable_error(value):
    with pytest.raises(ValueError, match="must be a mapping"):
        ModelConfig.from_dict(value)
    with pytest.raises(ValueError, match="must be a mapping"):
        TrainingConfig.from_dict(value)


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_training_roundtrip_preserves_resume_and_optimizer_policy(suffix, tmp_path):
    config = TrainingConfig(
        resume_from="checkpoints/step_4000",
        save_total_limit=5,
        expert_lr_pack=True,
        expert_lr_scale=2.0,
        expert_weight_decay=0.02,
        wsd_decay_ratio=0.25,
        eval_steps=17,
        log_steps=13,
        num_workers=0,
    )
    assert set(config.to_dict()) == {field.name for field in fields(TrainingConfig)}
    path = tmp_path / f"training{suffix}"
    config.save(path)
    assert TrainingConfig.load(path) == config
    with pytest.raises(ValueError, match="learing_rate"):
        TrainingConfig.from_dict({"learing_rate": 0.1})


def test_documented_cpu_example_forward_and_native_export(tmp_path):
    config = ModelConfig.load(
        Path(__file__).parents[1] / "configs/examples/tr_gqa_cpu.json", strict=True
    )
    model = ComplexityModel(config).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(ids)["logits"]
    model.save_pretrained(tmp_path)
    restored = ComplexityModel.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        assert torch.equal(restored(ids)["logits"], expected)
