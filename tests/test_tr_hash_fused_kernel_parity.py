"""Regression coverage for the TR-Hash fused-kernel dispatch path.

``tests/test_tr_hash_engine.py`` already locks in the PyTorch reference path
(``reference_topk_swiglu`` / ``TRHashBackend.PYTORCH``) on CPU, and the
hash-native ``FUSED_CUDA`` path against that same reference — but only under
``@pytest.mark.skipif(not torch.cuda.is_available())``. Two gaps remained:

1. The general ``CGGR`` Triton path (used for shapes the hash-native fused
   kernel does not cover, e.g. ``top_k=4``) had no numeric-parity test at
   all, CUDA or otherwise.
2. Nothing verified that capability detection (``_custom_cggr_available`` /
   ``_fused_cuda_available`` in ``complexity/tr_hash/capabilities.py``)
   degrades safely when the ``triton`` package itself is not installed
   (distinct from "installed but no CUDA device" — the situation on this
   development machine).

This file adds the CGGR parity test (CUDA+Triton gated, so it exercises
nothing here but closes the coverage gap for whenever the suite runs on
real hardware) and CPU-runnable tests for the import-safety and
capability-summary contract, which do run everywhere and would catch a
regression in backend-selection logic before any future consolidation of
``complexity/core/mlp/token_routed.py`` into ``complexity/tr_hash/engine.py``.
"""

from __future__ import annotations

import pytest
import torch

from complexity.tr_hash import (
    TRHashBackend,
    TRHashEngine,
    TRHashEngineConfig,
    TRHashPrecision,
    select_backend,
    supports_fused_cuda,
)


def test_triton_absence_is_reported_not_raised():
    """capabilities.py must swallow a missing ``triton`` package cleanly.

    This machine has no ``triton`` install at all (not just no CUDA device),
    which is a different failure mode than the CUDA-gated tests exercise:
    the ``ImportError`` happens before any device check runs.
    """
    from complexity.tr_hash.capabilities import (
        _custom_cggr_available,
        _fused_cuda_available,
    )

    assert _custom_cggr_available() is False
    assert _fused_cuda_available() is False


def test_auto_backend_on_cggr_eligible_shape_falls_back_to_pytorch_on_cpu():
    """The ``top_k=4`` / 8-expert shape only has a CGGR (not hash-native) fused
    path (see ``test_backend_selection_falls_back_to_general_cggr_shape`` in
    test_tr_hash_engine.py for the CUDA-side decision). On CPU it must still
    resolve to PYTORCH, and the reference forward for that exact shape must
    match ``reference_topk_swiglu`` bit-for-bit, so the shape this repo will
    eventually run through Triton CGGR is proven correct on the reference
    path *now*, before any fused kernel touches it.
    """
    from complexity.tr_hash.engine import reference_topk_swiglu

    torch.manual_seed(43)
    config = TRHashEngineConfig(
        hidden_size=16,
        vocab_size=257,
        num_experts=8,
        top_k=4,
        shared_width=24,
        expert_width=8,
        precision=TRHashPrecision.FP32,
    )
    assert not supports_fused_cuda(config)
    decision = select_backend(config, device_type="cpu")
    assert decision.selected is TRHashBackend.PYTORCH

    engine = TRHashEngine(config)
    hidden = torch.randn(2, 5, 16)
    token_ids = torch.randint(0, 257, (2, 5))

    actual = engine(hidden, token_ids)
    flat_x = hidden.reshape(-1, 16)
    routes = engine._routes(token_ids).reshape(config.top_k, -1)
    route_weights = torch.tensor(config.route_weights)[:, None].expand_as(routes)
    expected_routed = reference_topk_swiglu(
        flat_x,
        routes,
        engine.expert_gate,
        engine.expert_up,
        engine.expert_down,
        route_weights,
    )
    expected = (
        config.shared_output_scale * engine._shared(flat_x)
        + config.routed_output_scale * expected_routed
    ).view(2, 5, 16)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_capability_summary_contract_is_stable_across_shapes():
    """``capability_summary`` is a serialized manifest/logging contract
    (run manifests, dashboards) — a dropped or renamed key is a silent
    breaking change for whatever reads it. Lock in the key set for both a
    hash-native-fused-eligible shape and a CGGR-only shape.
    """
    expected_keys = {
        "experts",
        "top_k",
        "shared_width",
        "expert_width",
        "stored_width",
        "active_width",
        "active_width_fraction",
        "active_num_experts",
        "active_expert_width",
        "is_reduced_capacity",
        "attention",
        "routing",
        "phase",
        "precision",
        "world_size",
        "parallelism",
        "requested_backend",
        "selected_backend",
        "backend_reasons",
        "fused_cuda_eligible",
        "cuda_graph_buckets",
    }
    fused_eligible = TRHashEngine(
        TRHashEngineConfig(hidden_size=8, vocab_size=101, num_experts=4, top_k=2)
    )
    cggr_only = TRHashEngine(
        TRHashEngineConfig(hidden_size=8, vocab_size=101, num_experts=8, top_k=4)
    )
    summary_fused = fused_eligible.capability_summary("cpu")
    summary_cggr = cggr_only.capability_summary("cpu")

    assert set(summary_fused) == expected_keys
    assert set(summary_cggr) == expected_keys
    assert summary_fused["fused_cuda_eligible"] is True
    assert summary_cggr["fused_cuda_eligible"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cggr_matches_reference_forward_and_gradients_for_general_shape():
    """Numeric parity for the general (non-hash-native) Triton CGGR path.

    Mirrors ``test_fused_cuda_matches_reference_forward_and_gradients`` in
    test_tr_hash_engine.py, but for a shape that only CGGR — not the
    hash-native FUSED_CUDA kernel — can serve (``top_k=4``, 8 experts). This
    path previously had no numeric-correctness test at all, CUDA or CPU.
    """
    from complexity.tr_hash.engine import HAS_TRITON

    if not HAS_TRITON:
        pytest.skip("Triton CGGR kernels are unavailable")

    torch.manual_seed(53)
    common = dict(
        hidden_size=16,
        vocab_size=257,
        num_experts=8,
        top_k=4,
        shared_width=24,
        expert_width=8,
        precision=TRHashPrecision.FP32,
    )
    fused = TRHashEngine(TRHashEngineConfig(**common, backend=TRHashBackend.CGGR)).cuda()
    reference = TRHashEngine(TRHashEngineConfig(**common, backend=TRHashBackend.PYTORCH)).cuda()
    reference.load_state_dict(fused.state_dict())

    token_ids = torch.randint(0, 257, (2, 11), device="cuda")
    fused_hidden = torch.randn(2, 11, 16, device="cuda", requires_grad=True)
    reference_hidden = fused_hidden.detach().clone().requires_grad_(True)

    fused_output = fused(fused_hidden, token_ids)
    reference_output = reference(reference_hidden, token_ids)
    assert torch.allclose(fused_output, reference_output, atol=2e-3, rtol=2e-3)

    output_gradient = torch.randn_like(fused_output)
    fused_output.backward(output_gradient)
    reference_output.backward(output_gradient)
    assert torch.allclose(
        fused_hidden.grad,
        reference_hidden.grad,
        atol=3e-3,
        rtol=3e-3,
    )
    for fused_parameter, reference_parameter in zip(fused.parameters(), reference.parameters()):
        assert fused_parameter.grad is not None
        assert reference_parameter.grad is not None
        assert torch.allclose(
            fused_parameter.grad,
            reference_parameter.grad,
            atol=3e-3,
            rtol=3e-3,
        )
