import torch

from complexity.core.context.associative import SharedAssociativeContext
from complexity.experiments.shared_associative_attention import (
    SharedAssociativeAttention,
    SharedContextFusion,
    StableDeltaAssociativeAttention,
    MultiOrderLexicalDeltaAttention,
    MultiTimescaleDeltaAttention,
    CollisionNormalizedDeltaAttention,
    LexicalValueDeltaAttention,
    LexicalForgeDeltaAttention,
    PositionAwareCollisionForgeDeltaAttention,
)


def test_zero_contextual_mix_matches_lexical_baseline() -> None:
    torch.manual_seed(7)
    baseline = SharedAssociativeContext(hidden_size=16, rank=8, vocab_size=64)
    attention = SharedAssociativeAttention(
        hidden_size=16,
        rank=8,
        vocab_size=64,
        contextual_mix_init=0.0,
    )
    attention.load_state_dict(baseline.state_dict(), strict=False)
    hidden = torch.randn(2, 12, 16)
    token_ids = torch.randint(0, 64, (2, 12))
    baseline_state = baseline.initial_state(2, device=hidden.device, dtype=hidden.dtype)
    attention_state = attention.initial_state(2, device=hidden.device, dtype=hidden.dtype)

    baseline_output, baseline_next = baseline(hidden, token_ids, baseline_state)
    attention_output, attention_next = attention(hidden, token_ids, attention_state)

    torch.testing.assert_close(attention_output, baseline_output, rtol=0.0, atol=0.0)
    for actual, expected in zip(attention_next, baseline_next):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_contextual_addresses_train_shared_projection() -> None:
    torch.manual_seed(11)
    module = SharedAssociativeAttention(
        hidden_size=16,
        rank=8,
        vocab_size=64,
        contextual_mix_init=0.5,
    )
    hidden = torch.randn(2, 12, 16, requires_grad=True)
    token_ids = torch.randint(0, 64, (2, 12))
    state = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)

    output, _ = module(hidden, token_ids, state)
    output.square().mean().backward()

    assert module.address_proj.weight.grad is not None
    assert module.address_proj.weight.grad.norm() > 0


def test_shared_context_fusion_trains_tokenwise_fusion_without_scalar_gate() -> None:
    torch.manual_seed(13)
    module = SharedContextFusion(
        hidden_size=16,
        rank=8,
        fusion_size=12,
        vocab_size=64,
        contextual_mix_init=0.5,
    )
    hidden = torch.randn(2, 12, 16, requires_grad=True)
    token_ids = torch.randint(0, 64, (2, 12))
    state = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)

    output, _ = module(hidden, token_ids, state)
    output.square().mean().backward()

    assert output.shape == hidden.shape
    assert "output_gate" not in dict(module.named_parameters())
    assert "output_proj.weight" not in dict(module.named_parameters())
    for projection in (module.fusion_gate, module.fusion_value, module.fusion_out):
        assert projection.weight.grad is not None
        assert projection.weight.grad.norm() > 0


def test_delta_memory_repeated_writes_stay_bounded_and_replace_values() -> None:
    module = StableDeltaAssociativeAttention(
        hidden_size=8, rank=4, vocab_size=32, contextual_mix_init=0.0
    )
    key = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).expand(1, 128, 4)
    first = torch.tensor([[[1.0, -0.5, 0.25, 0.0]]]).expand(1, 64, 4)
    second = torch.tensor([[[-0.5, 0.75, 0.0, 0.25]]]).expand(1, 64, 4)
    state = torch.zeros(1, 4, 4)

    state, _ = module.delta_scan(key[:, :64], first, state)
    assert state.norm() < 2.0
    state, reads = module.delta_scan(key[:, 64:], second, state)

    torch.testing.assert_close(reads[:, -1], second[:, -1], rtol=0.02, atol=0.02)
    assert state.norm() < 2.0


def test_delta_attention_full_and_incremental_outputs_match() -> None:
    torch.manual_seed(17)
    module = StableDeltaAssociativeAttention(
        hidden_size=16,
        rank=8,
        vocab_size=64,
        contextual_mix_init=0.2,
    )
    hidden = torch.randn(2, 24, 16)
    token_ids = torch.randint(0, 64, (2, 24))
    full_state = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)
    full_output, full_next = module(hidden, token_ids, full_state)

    state = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)
    pieces = []
    for position in range(hidden.shape[1]):
        output, state = module(
            hidden[:, position : position + 1],
            token_ids[:, position : position + 1],
            state,
        )
        pieces.append(output)

    torch.testing.assert_close(torch.cat(pieces, dim=1), full_output)
    for actual, expected in zip(state, full_next):
        torch.testing.assert_close(actual, expected)
    assert full_next[0].dtype == torch.float32


def test_vectorized_delta_scan_matches_sequential_recurrence() -> None:
    torch.manual_seed(19)
    module = StableDeltaAssociativeAttention(
        hidden_size=8, rank=4, vocab_size=32, contextual_mix_init=0.0
    )
    keys = torch.nn.functional.normalize(torch.randn(2, 31, 4), dim=-1)
    values = torch.randn(2, 31, 4)
    state = torch.randn(2, 4, 4) * 0.1

    sequential_state, sequential_reads = module.delta_scan(keys, values, state)
    vector_state, vector_reads = module.vectorized_delta_scan(keys, values, state)

    torch.testing.assert_close(vector_reads, sequential_reads, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(vector_state, sequential_state, rtol=1e-5, atol=1e-5)


def test_multi_order_lexical_delta_full_and_incremental_match() -> None:
    torch.manual_seed(23)
    module = MultiOrderLexicalDeltaAttention(
        hidden_size=16,
        head_rank=4,
        orders=(1, 2, 4, 8),
        vocab_size=64,
        delta_chunk_size=16,
    )
    hidden = torch.randn(2, 24, 16)
    token_ids = torch.randint(0, 64, (2, 24))
    full_state = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)
    full_output, full_next = module(hidden, token_ids, full_state)

    state = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)
    pieces = []
    for position in range(hidden.shape[1]):
        output, state = module(
            hidden[:, position : position + 1],
            token_ids[:, position : position + 1],
            state,
        )
        pieces.append(output)

    torch.testing.assert_close(
        torch.cat(pieces, dim=1), full_output, rtol=1e-5, atol=1e-5
    )
    for actual, expected in zip(state, full_next):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert full_next[0].shape == (2, 4, 4, 4)
    assert full_next[2].shape == (2, 7)


def test_multi_timescale_delta_full_and_incremental_match() -> None:
    torch.manual_seed(29)
    module = MultiTimescaleDeltaAttention(
        hidden_size=16,
        state_rank=8,
        num_timescales=4,
        vocab_size=64,
        delta_chunk_size=16,
    )
    hidden = torch.randn(2, 20, 16)
    token_ids = torch.randint(0, 64, (2, 20))
    full_state = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)
    full_output, full_next = module(hidden, token_ids, full_state)

    state = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)
    pieces = []
    for position in range(hidden.shape[1]):
        output, state = module(
            hidden[:, position : position + 1],
            token_ids[:, position : position + 1],
            state,
        )
        pieces.append(output)
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), full_output, rtol=1e-5, atol=1e-5
    )
    for actual, expected in zip(state, full_next):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert module.decay_logits.numel() == 4
    assert module.write_logits.numel() == 4


def test_collision_normalized_delta_full_and_incremental_match() -> None:
    torch.manual_seed(31)
    module = CollisionNormalizedDeltaAttention(
        hidden_size=16,
        state_rank=8,
        vocab_size=64,
        delta_chunk_size=16,
    )
    hidden = torch.randn(2, 20, 16)
    token_ids = torch.randint(0, 64, (2, 20))
    full_state = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)
    full_output, full_next = module(hidden, token_ids, full_state)

    state = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)
    pieces = []
    for position in range(hidden.shape[1]):
        output, state = module(
            hidden[:, position : position + 1],
            token_ids[:, position : position + 1],
            state,
        )
        pieces.append(output)
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), full_output, rtol=1e-5, atol=1e-5
    )
    for actual, expected in zip(state, full_next):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert full_next[2].shape == (2, 8)
    assert torch.all(full_next[2] >= 0)


def test_lexical_value_delta_full_and_incremental_match() -> None:
    torch.manual_seed(37)
    module = LexicalValueDeltaAttention(
        hidden_size=16,
        state_rank=8,
        vocab_size=64,
        delta_chunk_size=16,
    )
    assert not hasattr(module, "value_proj")
    hidden = torch.randn(2, 20, 16)
    token_ids = torch.randint(0, 64, (2, 20))
    full_state = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)
    full_output, full_next = module(hidden, token_ids, full_state)

    state = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)
    pieces = []
    for position in range(hidden.shape[1]):
        output, state = module(
            hidden[:, position : position + 1],
            token_ids[:, position : position + 1],
            state,
        )
        pieces.append(output)
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), full_output, rtol=1e-5, atol=1e-5
    )
    for actual, expected in zip(state, full_next):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    values = module._lexical_values(torch.tensor([[3, 4]]))
    assert not torch.allclose(values[:, 0], values[:, 1])


def test_lexical_forge_delta_is_lexical_and_incremental() -> None:
    torch.manual_seed(41)
    module = LexicalForgeDeltaAttention(
        hidden_size=16,
        state_rank=8,
        vocab_size=64,
        delta_chunk_size=16,
    )
    token_ids = torch.randint(0, 64, (2, 20))
    hidden = torch.randn(2, 20, 16)
    other_hidden = torch.randn_like(hidden)
    read_codes, write_codes = module.forge_lexical_codes(token_ids)
    other_read_codes, other_write_codes = module.forge_lexical_codes(token_ids)
    torch.testing.assert_close(read_codes, other_read_codes)
    torch.testing.assert_close(write_codes, other_write_codes)
    forged_addresses = module.forge_addresses(hidden, token_ids)
    other_addresses = module.forge_addresses(other_hidden, token_ids)
    assert not torch.allclose(forged_addresses[0], other_addresses[0])
    initial = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)
    full_output, full_next = module(hidden, token_ids, initial)
    baseline = StableDeltaAssociativeAttention(
        hidden_size=16,
        rank=8,
        vocab_size=64,
        contextual_mix_init=0.1,
        delta_chunk_size=16,
    )
    baseline.load_state_dict(
        {
            key: value
            for key, value in module.state_dict().items()
            if key not in {"read_log_scale", "write_log_scale"}
            and not key.startswith("forge_")
        }
    )
    baseline_output, _ = baseline(
        hidden,
        token_ids,
        baseline.initial_state(2, device=hidden.device, dtype=hidden.dtype),
    )
    torch.testing.assert_close(full_output, baseline_output)
    token_scale = torch.nn.Embedding(64, 16)
    torch.nn.init.zeros_(token_scale.weight)
    module.attach_token_scale(token_scale)
    attached_output, _ = module(
        hidden,
        token_ids,
        module.initial_state(2, device=hidden.device, dtype=hidden.dtype),
    )
    torch.testing.assert_close(attached_output, baseline_output)
    other_output, _ = module(
        other_hidden,
        token_ids,
        module.initial_state(2, device=hidden.device, dtype=hidden.dtype),
    )
    assert not torch.allclose(full_output, other_output)

    state = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)
    pieces = []
    for position in range(hidden.shape[1]):
        output, state = module(
            hidden[:, position : position + 1],
            token_ids[:, position : position + 1],
            state,
        )
        pieces.append(output)
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), full_output, rtol=1e-5, atol=1e-5
    )
    for actual, expected in zip(state, full_next):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)








def test_position_aware_collision_forge_full_incremental_and_gate() -> None:
    torch.manual_seed(59)
    module = PositionAwareCollisionForgeDeltaAttention(
        16, 8, vocab_size=64, delta_chunk_size=16
    )
    token_scale = torch.nn.Embedding(64, 16)
    torch.nn.init.normal_(token_scale.weight, std=0.1)
    module.attach_token_scale(token_scale)
    token_ids = torch.randint(0, 64, (2, 20))
    hidden = torch.randn(2, 20, 16)
    initial = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)
    closed, closed_state = module(hidden, token_ids, initial)
    assert len(closed_state) == 4
    assert torch.all(closed_state[3] == 20)
    state = module.initial_state(2, device=hidden.device, dtype=hidden.dtype)
    pieces = []
    for position in range(hidden.shape[1]):
        output, state = module(
            hidden[:, position : position + 1],
            token_ids[:, position : position + 1], state
        )
        pieces.append(output)
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), closed, rtol=1e-5, atol=1e-5
    )
    for actual, expected in zip(state, closed_state):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        module.position_gate.fill_(0.5)
    opened, _ = module(
        hidden, token_ids,
        module.initial_state(2, device=hidden.device, dtype=hidden.dtype),
    )
    assert not torch.allclose(opened, closed)
