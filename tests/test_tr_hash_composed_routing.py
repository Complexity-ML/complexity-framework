"""TR-HASH formulas, causal history, fusion, and honest backend contracts."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from complexity.core.mlp.base import MLPConfig
from complexity.core.mlp.tr_hash_engine import TRHashEngineMLP
from complexity.tr_hash import (
    GraphBucket,
    TRHashBackend,
    TRHashEngine,
    TRHashEngineConfig,
    TRHashPhase,
    build_route_table,
    select_backend,
    supports_fused_cuda,
)
from complexity.tr_hash.composed_routing import (
    bind_value_channels,
    composed_head_routes,
    composed_mlp_routes,
    update_routing_history,
)


def _oracle_routes(ids, layer, history=None):
    output = torch.empty((2, *ids.shape), dtype=torch.long)
    for batch in range(ids.size(0)):
        prefix = [0, 0] if history is None else history[batch].tolist()
        sequence = prefix + ids[batch].tolist()
        for position, token in enumerate(ids[batch].tolist()):
            u = (sequence[position + 1] + 5 * sequence[position]) % 9
            if layer == 1:
                primary = ((token + 1) % 4) ^ (2 * ((u % 3) % 2))
                stride = 1 + (token // 4 + 2 * (u // 3) + 1) % 3
            else:
                primary = (token + layer) % 4
                stride = 1 + (token // 4 + (1 + layer % 2) * (u % 3) + layer) % 3
            output[:, batch, position] = torch.tensor([primary, (primary + stride) % 4])
    return output


@pytest.mark.parametrize("layer", [0, 1, 2, 3])
@pytest.mark.parametrize("length", [0, 1, 2, 7])
def test_composed_routes_match_scalar_source_oracle(layer, length):
    ids = torch.tensor([[17, 9, 5, 5, 253, 0, 12], [3, 44, 25, 5, 71, 8, 255]])[:, :length]
    history = torch.tensor([[19, 11], [2, 7]])
    for prefix in (None, history):
        routes = composed_mlp_routes(ids, layer, prefix)
        assert torch.equal(routes, _oracle_routes(ids, layer, prefix))
        assert torch.all(routes[0] != routes[1])


@pytest.mark.parametrize("layer", [0, 1, 3])
def test_composed_chunked_routes_preserve_two_token_history(layer):
    ids = torch.tensor([[17, 9, 5, 5, 253, 0], [3, 44, 25, 5, 71, 8]])
    expected = composed_mlp_routes(ids, layer)
    history = None
    pieces = []
    for start, end in ((0, 1), (1, 3), (3, 4), (4, 6)):
        chunk = ids[:, start:end]
        pieces.append(composed_mlp_routes(chunk, layer, history))
        history = update_routing_history(chunk, history)
    assert torch.equal(torch.cat(pieces, dim=-1), expected)
    assert torch.equal(history, ids[:, -2:])
    assert torch.equal(composed_mlp_routes(ids.flip(0), layer).flip(1), expected)


def test_history_is_validated_and_never_mutated():
    ids = torch.tensor([[4], [8]])
    previous = torch.tensor([[1, 2], [3, 5]])
    assert torch.equal(update_routing_history(ids, previous), torch.tensor([[2, 4], [5, 8]]))
    assert torch.equal(previous, torch.tensor([[1, 2], [3, 5]]))
    assert torch.equal(update_routing_history(ids), torch.tensor([[0, 4], [0, 8]]))
    for invalid in (
        torch.zeros(2, 1, dtype=torch.long),
        previous.float(),
        torch.zeros(1, 2, dtype=torch.long),
    ):
        with pytest.raises(ValueError, match="previous_token_ids"):
            composed_mlp_routes(ids, 0, invalid)
    with pytest.raises(ValueError, match="torch.long"):
        composed_mlp_routes(ids.float(), 0)


def test_route_table_is_zero_history_anchor_and_not_effective_sequence_table():
    config = TRHashEngineConfig(hidden_size=8, vocab_size=256, layer_index=1)
    table = build_route_table(config)
    for token in range(256):
        assert torch.equal(table[:, token], _oracle_routes(torch.tensor([[token]]), 1)[:, 0, 0])
    ids = torch.tensor([[5, 5, 5, 5]])
    assert not torch.equal(table[:, ids], composed_mlp_routes(ids, 1))


def test_composed_engine_fusion_matches_scalar_experts_and_gradients():
    engine = TRHashEngine(TRHashEngineConfig(hidden_size=8, vocab_size=256, layer_index=1))
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    ids = torch.tensor([[7, 5, 3, 5, 9], [45, 91, 5, 3, 9]])
    history = torch.tensor([[8, 6], [1, 3]])
    routes = _oracle_routes(ids, 1, history)
    expected = []
    for batch in range(2):
        outputs = []
        for position in range(5):
            x = hidden[batch, position]
            output = engine.shared_down(F.silu(engine.shared_gate(x)) * engine.shared_up(x))
            for route in range(2):
                expert = int(routes[route, batch, position])
                output = (
                    output
                    + (F.silu(x @ engine.expert_gate[expert]) * (x @ engine.expert_up[expert]))
                    @ engine.expert_down[expert]
                )
            outputs.append(output)
        expected.append(torch.stack(outputs))
    actual = engine(hidden, ids, previous_token_ids=history)
    torch.testing.assert_close(actual, torch.stack(expected))
    actual.square().sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(parameter.grad is not None for parameter in engine.parameters())


def test_composed_mlp_adapter_passes_history_and_refuses_weight_schedule():
    mlp = TRHashEngineMLP(MLPConfig(hidden_size=8, intermediate_size=8, vocab_size=32))
    ids = torch.tensor([[5, 5]])
    hidden = torch.randn(1, 2, 8)
    previous = torch.tensor([[8, 3]])
    torch.testing.assert_close(
        mlp(hidden, ids, previous_token_ids=previous), mlp.engine(hidden, ids, previous)
    )
    mlp.set_top_k_primary_weight(0.5)
    with pytest.raises(ValueError, match="fixed"):
        mlp.set_top_k_primary_weight(0.9)
    with pytest.raises(ValueError, match="four active experts"):
        mlp.set_active_experts(2)
    assert "topk_primary_weight" not in mlp.training_control_capabilities()


def test_composed_rejects_token_table_override():
    engine = TRHashEngine(TRHashEngineConfig(hidden_size=8, vocab_size=32))
    engine.load_route_table(engine.route_table.clone())
    with pytest.raises(ValueError, match="contextual routing"):
        engine.load_route_table(engine.route_table.roll(1, dims=1))


def test_composed_backend_selection_is_explicit_about_context():
    config = TRHashEngineConfig(hidden_size=8, vocab_size=32)
    assert not supports_fused_cuda(config)
    cuda = select_backend(
        config, device_type="cuda", cggr_available=True, fused_cuda_available=True
    )
    assert cuda.selected is TRHashBackend.CGGR
    assert any("contextual" in reason for reason in cuda.reasons)
    cpu = select_backend(
        config, device_type="cpu", cggr_available=False, fused_cuda_available=False
    )
    assert cpu.selected is TRHashBackend.PYTORCH
    for backend in (TRHashBackend.FUSED_CUDA, TRHashBackend.CUDA_GRAPH):
        explicit = replace(
            config, backend=backend, phase=TRHashPhase.INFERENCE, graph_buckets=(GraphBucket(1, 4),)
        )
        with pytest.raises(RuntimeError, match="contextual routing"):
            select_backend(
                explicit, device_type="cuda", cggr_available=True, fused_cuda_available=True
            )


@pytest.mark.parametrize("heads", [4])
def test_composed_head_routes_and_value_input_rotation(heads):
    ids = torch.arange(24).reshape(2, 12)
    actual = composed_head_routes(ids, 1, heads)
    assert actual.shape == (2, 12, min(2, heads))
    primary = torch.zeros_like(ids) if heads == 1 else (ids // (heads * (heads - 1))) % heads
    assert torch.equal(actual[..., 0], primary)
    if heads > 1:
        assert torch.equal(actual[..., 1], (primary + 1) % heads)
    hidden = torch.arange(64.0).expand(2, 12, 64)
    bound = bind_value_channels(hidden, ids, 0)
    expected = torch.stack(
        [
            torch.roll(row, -16 * int(token % 4))
            for row, token in zip(hidden.reshape(-1, 64), ids.reshape(-1))
        ]
    ).reshape_as(hidden)
    assert torch.equal(bound, expected)
    assert bind_value_channels(hidden, ids, 1) is hidden


def test_canonical_engine_has_one_route_family_and_fixed_capacity():
    config = TRHashEngineConfig(hidden_size=64, vocab_size=256)
    assert (config.shared_width, config.expert_width) == (128, 16)
    assert config.route_weights == (0.5, 0.5)
    assert (config.shared_output_scale, config.routed_output_scale) == (1, 2)
    for options in (
        {"route_weights": (0.9, 0.1)},
        {"routed_output_scale": 1},
        {"shared_output_scale": 2},
        {"num_experts": 8},
        {"top_k": 1},
    ):
        with pytest.raises(ValueError, match="TR-HASH"):
            TRHashEngineConfig(hidden_size=64, vocab_size=256, **options)
    for count in (1, 2, 8, 16):
        with pytest.raises(ValueError):
            TRHashEngineConfig(hidden_size=64, vocab_size=256, num_experts=count)


@pytest.mark.parametrize("heads", [1, 2, 8, 16])
def test_head_routing_rejects_other_head_counts(heads):
    with pytest.raises(ValueError, match="four query heads"):
        composed_head_routes(torch.tensor([[1, 2]]), 1, heads)
