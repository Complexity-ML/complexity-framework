import torch

from complexity.core.attention.base import AttentionConfig
from complexity.core.attention.gqa import GroupedQueryAttention


def _config() -> AttentionConfig:
    return AttentionConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        use_qk_norm=True,
        use_sdpa=False,
        lexical_object_rank=8,
        lexical_key_gate_init=0.05,
    )


def _copy_gqa(source: GroupedQueryAttention, target: torch.nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in source_state.items():
        if name in target_state:
            target_state[name].copy_(value)


def test_zero_lexical_objects_are_exact_gqa() -> None:
    from complexity.core.attention.lexical_key_gqa import LexicalKeyGQA

    torch.manual_seed(13)
    baseline = GroupedQueryAttention(_config()).eval()
    lexical = LexicalKeyGQA(_config()).eval()
    _copy_gqa(baseline, lexical)
    hidden = torch.randn(2, 6, 32)
    lexical_scale = torch.zeros(2, 6, 8, requires_grad=True)

    expected, _ = baseline(hidden)
    actual, _ = lexical(hidden, lexical_scale=lexical_scale)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().mean().backward()
    assert lexical_scale.grad is not None
    assert torch.count_nonzero(lexical_scale.grad) > 0


def test_learned_lexical_objects_change_gqa_without_value_path() -> None:
    from complexity.core.attention.lexical_key_gqa import LexicalKeyGQA

    torch.manual_seed(17)
    module = LexicalKeyGQA(_config()).eval()
    hidden = torch.randn(1, 5, 32)
    zero = torch.zeros(1, 5, 8)
    learned = torch.randn(1, 5, 8)
    baseline, _ = module(hidden, lexical_scale=zero)
    changed, _ = module(hidden, lexical_scale=learned)
    assert not torch.allclose(changed, baseline)
    assert not hasattr(module, "lexical_v_proj")


def test_lexical_key_cache_matches_full_sequence() -> None:
    from complexity.core.attention.lexical_key_gqa import LexicalKeyGQA

    torch.manual_seed(19)
    module = LexicalKeyGQA(_config()).eval()
    hidden = torch.randn(1, 6, 32)
    lexical_scale = torch.randn(1, 6, 8)
    full, _ = module(hidden, lexical_scale=lexical_scale)
    cache = None
    pieces = []
    for index in range(6):
        piece, cache = module(
            hidden[:, index : index + 1],
            lexical_scale=lexical_scale[:, index : index + 1],
            past_key_value=cache,
            use_cache=True,
        )
        pieces.append(piece)
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), full, atol=2e-5, rtol=2e-5
    )
