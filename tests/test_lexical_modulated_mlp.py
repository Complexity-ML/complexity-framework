import pytest
import torch


def test_lexical_modulated_mlp_is_token_conditioned_and_differentiable():
    from complexity.core.mlp import LexicalModulatedMLP, MLPConfig

    config = MLPConfig(
        hidden_size=8,
        intermediate_size=12,
        vocab_size=16,
        lexical_object_rank=4,
        lexical_object_gate_init=0.1,
    )
    mlp = LexicalModulatedMLP(config)
    with torch.no_grad():
        mlp.token_scale.weight.zero_()
        mlp.token_scale.weight[1].fill_(1.0)

    hidden = torch.randn(1, 2, 8, requires_grad=True)
    hidden.data[:, 1] = hidden.data[:, 0]
    output = mlp(hidden, token_ids=torch.tensor([[0, 1]]))

    assert output.shape == hidden.shape
    assert not torch.allclose(output[:, 0], output[:, 1])

    output.square().mean().backward()
    assert mlp.token_scale.weight.grad is not None
    assert mlp.object_up.weight.grad is not None
    assert mlp.object_down.weight.grad is not None


def test_lexical_modulated_mlp_requires_matching_token_ids():
    from complexity.core.mlp import LexicalModulatedMLP, MLPConfig

    mlp = LexicalModulatedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=12,
            vocab_size=16,
            lexical_object_rank=4,
        )
    )
    hidden = torch.randn(2, 3, 8)

    with pytest.raises(ValueError, match="token_ids"):
        mlp(hidden)
    with pytest.raises(ValueError, match="shape"):
        mlp(hidden, token_ids=torch.zeros(2, 2, dtype=torch.long))


def test_model_config_builds_lexical_modulated_mlp():
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel

    config = ModelConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=48,
        vocab_size=64,
        mlp_type="lexical_modulated",
        lexical_object_rank=8,
        lexical_object_gate_init=0.1,
    )
    model = ComplexityModel(config)
    output = model(torch.randint(0, 64, (2, 5)))

    assert output["logits"].shape == (2, 5, 64)
    assert all(block.mlp.object_rank == 8 for block in model.layers)


def test_100m_o200k_lexical_object_is_tied_and_parameter_matched():
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel

    lexical = ComplexityModel(
        ModelConfig(
            hidden_size=384,
            num_hidden_layers=10,
            num_attention_heads=8,
            num_key_value_heads=2,
            intermediate_size=1248,
            vocab_size=200019,
            attention_type="gqa",
            mlp_type="lexical_modulated",
            norm_type="rmsnorm",
            use_qk_norm=True,
            use_mu_guidance=False,
            lexical_object_rank=16,
            lexical_object_gate_init=0.1,
            tie_lexical_object_embeddings=True,
        )
    )

    first_table = lexical.layers[0].mlp.token_scale
    assert all(layer.mlp.token_scale is first_table for layer in lexical.layers)
    assert abs(lexical.num_parameters() - 98_197_440) / 98_197_440 < 0.0001


def test_lexical_channel_modulation_uses_no_extra_object_projections():
    from complexity.core.mlp import LexicalChannelModulatedMLP, MLPConfig

    mlp = LexicalChannelModulatedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=10,
            vocab_size=16,
            lexical_object_rank=4,
        )
    )
    with torch.no_grad():
        mlp.token_scale.weight.zero_()
        mlp.token_scale.weight[1].fill_(1.0)

    hidden = torch.randn(1, 2, 8)
    hidden[:, 1] = hidden[:, 0]
    output = mlp(hidden, token_ids=torch.tensor([[0, 1]]))

    assert output.shape == hidden.shape
    assert not torch.allclose(output[:, 0], output[:, 1])
    assert not hasattr(mlp, "object_up")
    assert not hasattr(mlp, "object_down")


def test_lexical_object_micro_experts_are_token_selected_and_differentiable():
    from complexity.core.mlp import LexicalObjectMicroExpertMLP, MLPConfig

    mlp = LexicalObjectMicroExpertMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=12,
            vocab_size=16,
            lexical_object_rank=4,
            micro_num_experts=4,
            micro_expert_width=2,
        )
    )
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    token_ids = torch.tensor([[0, 1, 2, 3]])
    output = mlp(hidden, token_ids=token_ids)

    assert output.shape == hidden.shape
    assert torch.equal(mlp.token_to_micro_expert[:4], torch.arange(4))
    output.square().mean().backward()
    assert mlp.micro_gate.weight.grad is not None
    assert mlp.micro_up.weight.grad is not None
    assert mlp.micro_down.grad is not None
