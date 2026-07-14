import math

import pytest
import torch

from complexity.core.attention import AttentionConfig
from complexity.core.attention.lexical_wrv import LexicalWRVAttention


def _config() -> AttentionConfig:
    return AttentionConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        causal_state_rank=8,
        causal_contextual_mix_init=0.1,
        max_position_embeddings=64,
        use_sdpa=True,
    )


def test_lexical_wrv_is_causal_and_has_no_qk_projections() -> None:
    torch.manual_seed(11)
    module = LexicalWRVAttention(_config()).eval()
    prefix = torch.randn(2, 6, 32)
    token_prefix = torch.randint(0, 64, (2, 6))
    future_a = torch.randn(2, 3, 32)
    future_b = torch.randn(2, 3, 32)
    token_future = torch.randint(0, 64, (2, 3))
    with torch.inference_mode():
        output_a, _ = module(
            torch.cat((prefix, future_a), dim=1),
            token_ids=torch.cat((token_prefix, token_future), dim=1),
        )
        output_b, _ = module(
            torch.cat((prefix, future_b), dim=1),
            token_ids=torch.cat((token_prefix, token_future), dim=1),
        )
    torch.testing.assert_close(output_a[:, :6], output_b[:, :6])
    assert not hasattr(module, "q_proj")
    assert not hasattr(module, "k_proj")
    assert hasattr(module, "read_proj")
    assert hasattr(module, "write_context_proj")
    assert hasattr(module, "value_proj")
    assert hasattr(module, "read_norm")
    assert hasattr(module, "write_norm")
    assert module.num_read_heads == 4
    assert module.num_write_heads == 2
    assert module.head_dim == 8



def test_lexical_wrv_starts_from_contextual_attention_with_neutral_lexical_residual() -> None:
    module = LexicalWRVAttention(_config())
    assert module.scale == 1.0 / math.sqrt(module.head_dim)
    torch.testing.assert_close(module.lexical_gate, torch.zeros(2))


def test_lexical_wrv_can_fix_lexical_residual_at_zero() -> None:
    config = _config()
    config.disable_lexical_wrv_residual = True
    module = LexicalWRVAttention(config).eval()
    hidden = torch.randn(2, 7, 32)
    token_ids_a = torch.randint(0, 64, (2, 7))
    token_ids_b = torch.randint(0, 64, (2, 7))
    output_a, _ = module(hidden, token_ids=token_ids_a)
    output_b, _ = module(hidden, token_ids=token_ids_b)
    torch.testing.assert_close(output_a, output_b)
    assert not module.lexical_gate.requires_grad
    torch.testing.assert_close(module.lexical_gate, torch.zeros(2))


def test_hybrid_lexical_wrv_reuses_the_write_address_for_grouped_reads() -> None:
    config = _config()
    config.lexical_wrv_hybrid = True
    module = LexicalWRVAttention(config)
    token_ids = torch.randint(0, 64, (2, 7))
    lexical_writes = module.lexical_base_writes(token_ids)
    lexical_reads = module.lexical_base_reads(lexical_writes)
    assert lexical_reads.shape == (2, 7, 4, 8)
    torch.testing.assert_close(lexical_reads[:, :, 0], lexical_writes[:, :, 0])
    torch.testing.assert_close(lexical_reads[:, :, 1], lexical_writes[:, :, 0])
    torch.testing.assert_close(lexical_reads[:, :, 2], lexical_writes[:, :, 1])
    torch.testing.assert_close(lexical_reads[:, :, 3], lexical_writes[:, :, 1])


def test_hybrid_lexical_reads_change_the_operator_beyond_write_only_wrv() -> None:
    torch.manual_seed(23)
    write_only = LexicalWRVAttention(_config()).eval()
    hybrid_config = _config()
    hybrid_config.lexical_wrv_hybrid = True
    hybrid = LexicalWRVAttention(hybrid_config).eval()
    hybrid.load_state_dict(write_only.state_dict())
    with torch.no_grad():
        write_only.lexical_gate.fill_(0.5)
        hybrid.lexical_gate.fill_(0.5)
    hidden = torch.randn(2, 7, 32)
    token_ids = torch.randint(0, 64, (2, 7))
    write_only_output, _ = write_only(hidden, token_ids=token_ids)
    hybrid_output, _ = hybrid(hidden, token_ids=token_ids)
    assert not torch.allclose(write_only_output, hybrid_output)


def test_hybrid_lexical_wrv_rejects_a_disabled_lexical_residual() -> None:
    config = _config()
    config.lexical_wrv_hybrid = True
    config.disable_lexical_wrv_residual = True
    with pytest.raises(ValueError, match="hybrid"):
        LexicalWRVAttention(config)


def test_hybrid_lexical_wrv_incremental_cache_matches_full_sequence() -> None:
    torch.manual_seed(29)
    config = _config()
    config.lexical_wrv_hybrid = True
    module = LexicalWRVAttention(config).eval()
    with torch.no_grad():
        module.lexical_gate.fill_(0.25)
    hidden = torch.randn(1, 9, 32)
    token_ids = torch.randint(0, 64, (1, 9))
    with torch.inference_mode():
        full, _ = module(hidden, token_ids=token_ids)
        cache = None
        pieces = []
        for position in range(hidden.shape[1]):
            piece, cache = module(
                hidden[:, position : position + 1],
                token_ids=token_ids[:, position : position + 1],
                past_key_value=cache,
                use_cache=True,
            )
            pieces.append(piece)
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), full, atol=2e-5, rtol=2e-5
    )


def test_lexical_wrv_can_bypass_and_freeze_read_write_norms() -> None:
    config = _config()
    config.disable_lexical_wrv_norms = True
    module = LexicalWRVAttention(config)
    assert module.disable_read_write_norms
    assert not module.read_norm.weight.requires_grad
    assert not module.write_norm.weight.requires_grad


def test_lexical_wrv_reuses_precomputed_base_writes_exactly() -> None:
    module = LexicalWRVAttention(_config())
    token_ids = torch.randint(0, 64, (2, 7))
    base = module.lexical_base_writes(token_ids)
    expected = module._lexical_writes(token_ids)
    actual = module._lexical_writes(token_ids, lexical_base_writes=base)
    torch.testing.assert_close(actual, expected)


def test_lexical_wrv_rotates_repeated_writes_by_position() -> None:
    module = LexicalWRVAttention(_config())
    repeated = torch.ones(1, 2, 2, module.head_dim)
    rotated = module._apply_rotary(repeated, position_offset=0)
    assert not torch.allclose(rotated[:, :, 0], rotated[:, :, 1])


def test_lexical_wrv_respects_configured_rope_theta() -> None:
    config = _config()
    config.rope_theta = 500000.0
    module = LexicalWRVAttention(config)
    assert module.rope_theta == 500000.0


def test_lexical_wrv_incremental_cache_matches_full_sequence() -> None:
    torch.manual_seed(13)
    module = LexicalWRVAttention(_config()).eval()
    hidden = torch.randn(1, 10, 32)
    token_ids = torch.randint(0, 64, (1, 10))
    with torch.inference_mode():
        full, _ = module(hidden, token_ids=token_ids)
        cache = None
        pieces = []
        for position in range(hidden.shape[1]):
            piece, cache = module(
                hidden[:, position : position + 1],
                token_ids=token_ids[:, position : position + 1],
                past_key_value=cache,
                use_cache=True,
            )
            pieces.append(piece)
    assert cache is not None
    assert len(cache) == 2
    assert cache[0].shape == (1, 2, 10, 8)
    assert cache[1].shape == (1, 2, 10, 8)
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), full, atol=2e-5, rtol=2e-5
    )


def test_o200k_cli_parses_lexical_attention_layer_indices() -> None:
    from complexity.training.o200k.cli import build_parser

    args = build_parser().parse_args(
        ["--lexical-attention-layer-indices", "4", "9"]
    )
    assert args.lexical_attention_layer_indices == [4, 9]


def test_o200k_cli_parses_hybrid_lexical_wrv() -> None:
    from complexity.training.o200k.cli import build_parser

    args = build_parser().parse_args(["--lexical-wrv-hybrid"])
    assert args.lexical_wrv_hybrid


def test_hybrid_h200_yaml_loads_through_the_real_runner_parser() -> None:
    from complexity.training.o200k.cli import build_parser
    from complexity.training.run_config import parse_args_with_yaml_config

    args = parse_args_with_yaml_config(
        build_parser(),
        [
            "--config",
            "configs/run_configs/review_h200/"
            "h200_review_matched_wrv_hybrid_seed42.yaml",
        ],
    )
    assert args.lexical_wrv_hybrid
    assert args.lexical_wrv_gate_init == 0.1
    assert not args.disable_lexical_wrv_residual
    assert args.attention_type == "lexical_wrv"
    assert args.run_name == "h200-review-wrv-hybrid-seed42"


def test_global_lexical_wrv_layers_share_the_lexical_object_table() -> None:
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel

    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        attention_type="lexical_wrv",
        mlp_type="lexical_object_micro_expert",
        lexical_object_rank=16,
        tie_lexical_object_embeddings=True,
    )
    model = ComplexityModel(config)
    assert hasattr(model, "lexical_token_scale")
    assert all(not hasattr(layer.mlp, "token_scale") for layer in model.layers)
    token_scale_parameters = [
        name for name, _ in model.named_parameters() if name.endswith("token_scale.weight")
    ]
    assert token_scale_parameters == ["lexical_token_scale.weight"]
    expected_std = config.initializer_range / (2 * config.num_hidden_layers) ** 0.5
    for layer in model.layers:
        output_proj = getattr(layer.self_attn, "output_proj")
        actual_std = output_proj.weight.std().item()
        assert abs(actual_std - expected_std) < expected_std * 0.15


def test_hybrid_flag_reaches_every_global_wrv_layer_without_extra_parameters() -> None:
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel

    write_only_config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        attention_type="lexical_wrv",
        mlp_type="lexical_object_micro_expert",
        lexical_object_rank=16,
        tie_lexical_object_embeddings=True,
    )
    hybrid_config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        attention_type="lexical_wrv",
        mlp_type="lexical_object_micro_expert",
        lexical_object_rank=16,
        tie_lexical_object_embeddings=True,
        lexical_wrv_hybrid=True,
        lexical_wrv_gate_init=0.1,
    )
    write_only = ComplexityModel(write_only_config)
    hybrid = ComplexityModel(hybrid_config)
    assert all(layer.self_attn.lexical_wrv_hybrid for layer in hybrid.layers)
    for layer in hybrid.layers:
        torch.testing.assert_close(
            layer.self_attn.lexical_gate,
            torch.full_like(layer.self_attn.lexical_gate, 0.1),
        )
    assert sum(parameter.numel() for parameter in write_only.parameters()) == sum(
        parameter.numel() for parameter in hybrid.parameters()
    )


def test_hybrid_model_trains_lexical_addresses_from_neutral_gates() -> None:
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel

    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        attention_type="lexical_wrv",
        lexical_wrv_hybrid=True,
        mlp_type="lexical_object_micro_expert",
        lexical_object_rank=16,
        tie_lexical_object_embeddings=True,
    )
    model = ComplexityModel(config)
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
    lexical_gradient = model.lexical_token_scale.weight.grad
    assert lexical_gradient is not None
    assert torch.isfinite(lexical_gradient).all()


def test_lexical_wrv_remains_causal_with_padding_mask() -> None:
    torch.manual_seed(17)
    module = LexicalWRVAttention(_config()).eval()
    prefix = torch.randn(2, 5, 32)
    future_a = torch.randn(2, 3, 32)
    future_b = torch.randn(2, 3, 32)
    token_ids = torch.randint(0, 64, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.bool)
    with torch.inference_mode():
        output_a, _ = module(
            torch.cat((prefix, future_a), dim=1),
            token_ids=token_ids,
            attention_mask=attention_mask,
        )
        output_b, _ = module(
            torch.cat((prefix, future_b), dim=1),
            token_ids=token_ids,
            attention_mask=attention_mask,
        )
    torch.testing.assert_close(output_a[:, :5], output_b[:, :5])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("use_qk_norm", False),
        ("use_sdpa", False),
        ("sliding_window", 32),
        ("rope_type", "yarn"),
    ],
)
def test_lexical_wrv_rejects_unsupported_attention_options(
    field: str, value: object
) -> None:
    config = _config()
    setattr(config, field, value)
    with pytest.raises(ValueError):
        LexicalWRVAttention(config)


def test_lexical_wrv_cached_chunk_matches_full_sequence() -> None:
    torch.manual_seed(19)
    module = LexicalWRVAttention(_config()).eval()
    hidden = torch.randn(1, 9, 32)
    token_ids = torch.randint(0, 64, (1, 9))
    with torch.inference_mode():
        full, _ = module(hidden, token_ids=token_ids)
        prefix, cache = module(
            hidden[:, :5], token_ids=token_ids[:, :5], use_cache=True
        )
        chunk, _ = module(
            hidden[:, 5:],
            token_ids=token_ids[:, 5:],
            past_key_value=cache,
            use_cache=True,
        )
    torch.testing.assert_close(
        torch.cat((prefix, chunk), dim=1), full, atol=2e-5, rtol=2e-5
    )
