"""Execution and configuration contracts for canonical contextual TR-HASH."""

import pytest
import torch
from complexity.tr_hash import (
    TRHashBackend,
    TRHashEngine,
    TRHashEngineConfig,
    TRHashPrecision,
    TRHashStrategy,
    select_backend,
    supports_fused_cuda,
)


def test_single_architecture_capacity_and_width_metadata():
    config = TRHashEngineConfig(hidden_size=64, vocab_size=256)
    assert list(TRHashStrategy) == [TRHashStrategy.COMPOSED_MODULO]
    assert (config.num_experts, config.top_k, config.shared_width, config.expert_width) == (
        4,
        2,
        128,
        16,
    )
    assert config.stored_mlp_width == 192 and config.active_mlp_width == 160
    assert config.active_width_fraction == 160 / 192


@pytest.mark.parametrize(
    "options",
    [
        {"num_experts": 1},
        {"num_experts": 2},
        {"num_experts": 8},
        {"num_experts": 16},
        {"top_k": 1},
        {"top_k": 4},
        {"route_weights": (0.9, 0.1)},
        {"shared_width": 0},
        {"shared_output_scale": 2},
        {"routed_output_scale": 1},
        {"routing_strategy": "modulo_cyclic"},
        {"routing_strategy": "token_id_multi_hash"},
        {"routing_strategy": "token_id_balanced_hash"},
        {"routing_strategy": "token_id_hierarchical_hash"},
    ],
)
def test_incompatible_architecture_options_fail_explicitly(options):
    with pytest.raises(ValueError):
        TRHashEngineConfig(hidden_size=8, vocab_size=32, **options)


def test_canonical_anchor_state_roundtrip_and_bad_dtype_rejection():
    engine = TRHashEngine(TRHashEngineConfig(hidden_size=8, vocab_size=32))
    copied = TRHashEngine(engine.config)
    copied.load_state_dict(engine.state_dict())
    ids = torch.tensor([[1, 5, 9]])
    x = torch.randn(1, 3, 8)
    torch.testing.assert_close(engine(x, ids), copied(x, ids), rtol=0, atol=0)
    invalid = engine.state_dict()
    invalid["route_table"] = invalid["route_table"].float()
    with pytest.raises(RuntimeError, match="canonical zero-history"):
        copied.load_state_dict(invalid)


def test_auto_cpu_fallback_and_cuda_cggr_selection_are_truthful():
    config = TRHashEngineConfig(hidden_size=8, vocab_size=32)
    assert not supports_fused_cuda(config)
    cpu = select_backend(config, device_type="cpu", cggr_available=False, fused_cuda_available=True)
    assert cpu.selected is TRHashBackend.PYTORCH
    assert "CGGR requires CUDA" in cpu.reasons
    cuda = select_backend(
        config, device_type="cuda", cggr_available=True, fused_cuda_available=True
    )
    assert cuda.selected is TRHashBackend.CGGR


@pytest.mark.parametrize("backend", [TRHashBackend.FUSED_CUDA, TRHashBackend.CUDA_GRAPH])
def test_static_routing_backends_fail_instead_of_changing_routes(backend):
    from complexity.tr_hash import GraphBucket, TRHashPhase

    config = TRHashEngineConfig(
        hidden_size=8,
        vocab_size=32,
        backend=backend,
        phase=TRHashPhase.INFERENCE,
        graph_buckets=(GraphBucket(1, 4),),
    )
    with pytest.raises(RuntimeError, match="contextual routing"):
        select_backend(config, device_type="cuda", cggr_available=True, fused_cuda_available=True)


@pytest.mark.parametrize("precision", [TRHashPrecision.FP8, TRHashPrecision.INT8])
def test_unimplemented_quantization_is_rejected(precision):
    with pytest.raises(NotImplementedError):
        select_backend(
            TRHashEngineConfig(hidden_size=8, vocab_size=32, precision=precision), device_type="cpu"
        )


def test_task_only_partial_state_keeps_canonical_anchor():
    engine = TRHashEngine(TRHashEngineConfig(hidden_size=8, vocab_size=32))
    anchor = engine.route_table.clone()
    replacement = torch.randn_like(engine.shared_gate.weight)
    missing, unexpected = engine.load_state_dict({"shared_gate.weight": replacement}, strict=False)
    assert "route_table" in missing and not unexpected
    assert torch.equal(engine.route_table, anchor)
    assert torch.equal(engine.shared_gate.weight, replacement)
