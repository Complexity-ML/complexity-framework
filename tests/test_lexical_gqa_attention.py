import torch

from complexity.core.attention.base import AttentionConfig
from complexity.core.attention.gqa import GroupedQueryAttention


def _config(
    *,
    use_sdpa: bool = False,
    gate_init: float = 0.0,
    use_token_code: bool = True,
) -> AttentionConfig:
    return AttentionConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        use_qk_norm=True,
        use_sdpa=use_sdpa,
        lexical_object_rank=8,
        lexical_gqa_rank=8,
        lexical_gqa_gate_init=gate_init,
        lexical_gqa_use_token_code=use_token_code,
    )


def _copy_gqa_weights(source: GroupedQueryAttention, target: torch.nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in source_state.items():
        if name in target_state:
            target_state[name].copy_(value)


def test_lexical_gqa_is_baseline_gqa_at_zero_gate() -> None:
    from complexity.core.attention.lexical_gqa import LexicalBiasGQA

    torch.manual_seed(3)
    baseline = GroupedQueryAttention(_config()).eval()
    lexical = LexicalBiasGQA(_config()).eval()
    _copy_gqa_weights(baseline, lexical)
    hidden = torch.randn(2, 6, 32)
    lexical_scale = torch.randn(2, 6, 8)
    token_ids = torch.randint(0, 64, (2, 6))

    expected, _ = baseline(hidden)
    actual, _ = lexical(
        hidden, lexical_scale=lexical_scale, token_ids=token_ids
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_lexical_gate_changes_attention_without_lexical_values() -> None:
    from complexity.core.attention.lexical_gqa import LexicalBiasGQA

    torch.manual_seed(5)
    module = LexicalBiasGQA(_config()).eval()
    hidden = torch.randn(1, 5, 32)
    lexical_scale = torch.randn(1, 5, 8)
    token_ids = torch.randint(0, 64, (1, 5))
    baseline, _ = module(
        hidden, lexical_scale=lexical_scale, token_ids=token_ids
    )
    with torch.no_grad():
        module.lexical_gate.fill_(0.25)
    changed, _ = module(
        hidden, lexical_scale=lexical_scale, token_ids=token_ids
    )

    assert not torch.allclose(changed, baseline)
    assert not hasattr(module, "lexical_v_proj")


def test_zero_initialized_lexical_gate_receives_gradient() -> None:
    from complexity.core.attention.lexical_gqa import LexicalBiasGQA

    torch.manual_seed(7)
    module = LexicalBiasGQA(_config()).train()
    hidden = torch.randn(2, 5, 32)
    lexical_scale = torch.zeros(2, 5, 8)
    token_ids = torch.randint(0, 64, (2, 5))
    output, _ = module(
        hidden, lexical_scale=lexical_scale, token_ids=token_ids
    )
    output.square().mean().backward()

    assert module.lexical_gate.grad is not None
    assert torch.isfinite(module.lexical_gate.grad).all()
    assert module.lexical_gate.grad.abs().sum() > 0


def test_learned_only_key_starts_as_gqa_but_trains_lexical_objects() -> None:
    from complexity.core.attention.lexical_gqa import LexicalBiasGQA

    torch.manual_seed(9)
    baseline = GroupedQueryAttention(_config()).eval()
    lexical = LexicalBiasGQA(
        _config(gate_init=0.05, use_token_code=False)
    ).eval()
    _copy_gqa_weights(baseline, lexical)
    hidden = torch.randn(2, 6, 32)
    lexical_scale = torch.zeros(2, 6, 8, requires_grad=True)
    token_ids = torch.randint(0, 64, (2, 6))

    expected, _ = baseline(hidden)
    actual, _ = lexical(
        hidden, lexical_scale=lexical_scale, token_ids=token_ids
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().mean().backward()
    assert lexical_scale.grad is not None
    assert torch.isfinite(lexical_scale.grad).all()
    assert torch.count_nonzero(lexical_scale.grad) > 0


def test_lexical_gqa_incremental_cache_matches_full_sequence() -> None:
    from complexity.core.attention.lexical_gqa import LexicalBiasGQA

    torch.manual_seed(11)
    module = LexicalBiasGQA(_config()).eval()
    with torch.no_grad():
        module.lexical_gate.fill_(0.2)
    hidden = torch.randn(1, 6, 32)
    lexical_scale = torch.randn(1, 6, 8)
    token_ids = torch.randint(0, 64, (1, 6))
    full, _ = module(
        hidden, lexical_scale=lexical_scale, token_ids=token_ids
    )

    cache = None
    pieces = []
    for index in range(hidden.shape[1]):
        piece, cache = module(
            hidden[:, index : index + 1],
            lexical_scale=lexical_scale[:, index : index + 1],
            token_ids=token_ids[:, index : index + 1],
            past_key_value=cache,
            use_cache=True,
        )
        pieces.append(piece)
    incremental = torch.cat(pieces, dim=1)
    torch.testing.assert_close(incremental, full, atol=2e-5, rtol=2e-5)


def test_full_lexical_gqa_model_trains_from_exact_gqa_initialization() -> None:
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel

    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        attention_type="lexical_gqa",
        lexical_gqa_rank=8,
        lexical_gqa_gate_init=0.0,
        mlp_type="lexical_object_micro_expert",
        lexical_object_rank=8,
        tie_lexical_object_embeddings=True,
    )
    model = ComplexityModel(config)
    torch.testing.assert_close(
        model.lexical_token_scale.weight,
        torch.zeros_like(model.lexical_token_scale.weight),
    )
    token_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(token_ids)["logits"]
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, config.vocab_size),
        token_ids[:, 1:].reshape(-1),
    )
    loss.backward()
    for layer in model.layers:
        gradient = layer.self_attn.lexical_gate.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
