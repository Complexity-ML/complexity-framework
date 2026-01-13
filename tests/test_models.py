"""Tests for complexity.models module."""

import pytest
import torch


class TestComplexityModel:
    """Test ComplexityModel."""

    def test_create_tiny_model(self):
        """Test creating a tiny model."""
        from complexity.models import ComplexityModel
        from complexity.config import ModelConfig

        config = ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=1000,
        )
        model = ComplexityModel(config)

        assert model is not None
        assert model.config.hidden_size == 128
        assert model.config.num_hidden_layers == 2

    def test_forward_pass(self):
        """Test forward pass."""
        from complexity.models import ComplexityModel
        from complexity.config import ModelConfig

        config = ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=1000,
        )
        model = ComplexityModel(config)

        x = torch.randint(0, 1000, (2, 16))
        out = model(x)

        assert "logits" in out
        assert out["logits"].shape == (2, 16, 1000)

    def test_generate(self):
        """Test generation."""
        from complexity.models import ComplexityModel
        from complexity.config import ModelConfig

        config = ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=1000,
        )
        model = ComplexityModel(config)

        prompt = torch.randint(0, 1000, (1, 8))
        generated = model.generate(prompt, max_new_tokens=5, do_sample=False)

        assert generated.shape[0] == 1
        assert generated.shape[1] == 8 + 5  # prompt + generated

    def test_with_inl_dynamics(self):
        """Test model with INL Dynamics enabled."""
        from complexity.models import ComplexityModel
        from complexity.config import ModelConfig

        config = ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=1000,
            use_inl_dynamics=True,
        )
        model = ComplexityModel(config)

        x = torch.randint(0, 1000, (2, 16))
        out = model(x)

        assert "velocity_states" in out
        assert out["velocity_states"] is not None

    def test_with_token_routed_mlp(self):
        """Test model with TokenRouted MLP (MoE)."""
        from complexity.models import ComplexityModel
        from complexity.config import ModelConfig

        config = ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=1000,
            mlp_type="token_routed",
            num_experts=4,
        )
        model = ComplexityModel(config)

        x = torch.randint(0, 1000, (2, 16))
        out = model(x)

        assert out["logits"].shape == (2, 16, 1000)


class TestModelPresets:
    """Test model presets."""

    def test_get_preset(self):
        """Test getting preset configs."""
        from complexity.config import get_preset

        config = get_preset("complexity-tiny")
        assert config.hidden_size == 256
        assert config.num_hidden_layers == 6

    def test_from_preset(self):
        """Test creating model from preset."""
        from complexity.models import ComplexityModel

        model = ComplexityModel.from_preset("complexity-tiny")
        assert model is not None
        assert model.config.hidden_size == 256


class TestModelSaveLoad:
    """Test model save/load."""

    def test_save_load(self, tmp_path):
        """Test saving and loading model."""
        from complexity.models import ComplexityModel
        from complexity.config import ModelConfig

        config = ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=1000,
        )
        model = ComplexityModel(config)

        # Save
        save_path = tmp_path / "model"
        model.save_pretrained(str(save_path))

        # Load
        loaded = ComplexityModel.from_pretrained(str(save_path))
        assert loaded.config.hidden_size == 128

        # Verify same output
        x = torch.randint(0, 1000, (1, 8))
        torch.manual_seed(42)
        out1 = model(x)["logits"]
        torch.manual_seed(42)
        out2 = loaded(x)["logits"]

        assert torch.allclose(out1, out2, atol=1e-5)
