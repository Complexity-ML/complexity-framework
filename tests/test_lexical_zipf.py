import torch

from complexity.core.attention.base import AttentionConfig
from complexity.core.attention.lexical_key_gqa import ProjectedLexicalKeyGQA
from complexity.config import ModelConfig
from complexity.models import ComplexityModel
from complexity.models.builder import build_lexical_zipf_weights


def test_ordered_zipf_weights_follow_frequency_rank_and_have_unit_rms() -> None:
    counts = torch.tensor([1000, 100, 10, 1, 0], dtype=torch.int64)
    weights = build_lexical_zipf_weights(
        counts,
        mode="ordered",
        alpha=0.25,
        floor=0.1,
        permutation_seed=7,
    )
    assert torch.all(weights[:-1] <= weights[1:])
    torch.testing.assert_close(weights.square().mean(), torch.tensor(1.0))


def test_permuted_zipf_is_deterministic_and_histogram_matched() -> None:
    counts = torch.arange(64, 0, -1)
    ordered = build_lexical_zipf_weights(
        counts,
        mode="ordered",
        alpha=0.25,
        floor=0.1,
        permutation_seed=11,
    )
    permuted_a = build_lexical_zipf_weights(
        counts,
        mode="permuted",
        alpha=0.25,
        floor=0.1,
        permutation_seed=11,
    )
    permuted_b = build_lexical_zipf_weights(
        counts,
        mode="permuted",
        alpha=0.25,
        floor=0.1,
        permutation_seed=11,
    )
    torch.testing.assert_close(permuted_a, permuted_b)
    torch.testing.assert_close(permuted_a.sort().values, ordered.sort().values)
    assert not torch.equal(permuted_a, ordered)


def test_projected_lexical_key_applies_token_weights_only_to_residual() -> None:
    config = AttentionConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        use_qk_norm=True,
        use_sdpa=False,
        lexical_object_rank=8,
        lexical_key_gate_init=0.05,
    )
    torch.manual_seed(29)
    module = ProjectedLexicalKeyGQA(config).eval()
    hidden = torch.randn(1, 5, 32)
    lexical_scale = torch.randn(1, 5, 8)
    zero_weight = torch.zeros(1, 5)
    unit_weight = torch.ones(1, 5)
    without_lexical, _ = module(
        hidden,
        lexical_scale=lexical_scale,
        lexical_weight=zero_weight,
    )
    with_lexical, _ = module(
        hidden,
        lexical_scale=lexical_scale,
        lexical_weight=unit_weight,
    )
    assert not torch.allclose(with_lexical, without_lexical)


def test_model_loads_zipf_artifact_and_runs(tmp_path) -> None:
    artifact = tmp_path / "counts.pt"
    torch.save({"counts": torch.arange(64, 0, -1)}, artifact)
    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        shared_intermediate_size=32,
        attention_type="projected_lexical_key_gqa",
        mlp_type="lexical_object_micro_expert",
        lexical_object_rank=8,
        tie_lexical_object_embeddings=True,
        micro_num_experts=2,
        micro_expert_width=8,
        lexical_zipf_path=str(artifact),
        lexical_zipf_mode="ordered",
    )
    model = ComplexityModel(config).eval()
    assert model.lexical_zipf_weights.shape == (64,)
    output = model(torch.randint(0, 64, (2, 7)))
    assert output["logits"].shape == (2, 7, 64)
