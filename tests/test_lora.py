from __future__ import annotations

import copy
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from complexity.core.attention.base import AttentionConfig
from complexity.core.attention.gqa import GroupedQueryAttention
from complexity.tr_hash import (
    AttentionBackbone,
    TRHashBackend,
    TRHashEngine,
    TRHashEngineConfig,
)
from complexity.training.lora import (
    LoRALinear,
    adapter_state_dict,
    apply_lora,
    load_adapter_state_dict,
    merged_model_state_dict,
    unmerge_adapter_from_base,
)
from scripts.sft_500m_32k_tr import build_optimizer


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.other = nn.Linear(4, 4, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.other(self.q_proj(inputs))


def test_lora_freezes_base_and_only_wraps_targets() -> None:
    model = TinyModel()
    stats = apply_lora(model, rank=2, alpha=4, dropout=0, targets=("q_proj",))

    assert stats["modules"] == 1
    assert stats["trainable"] == 16
    assert not model.q_proj.base.weight.requires_grad
    assert not model.other.weight.requires_grad


def test_linear_lora_does_not_reintroduce_slow_addmm_epilogue() -> None:
    """Guard the RTX 5060 Ti result: addmm made the full block 2.61% slower."""

    adapter = LoRALinear(nn.Linear(8, 12, bias=False), rank=4, alpha=8, dropout=0)
    inputs = torch.randn(3, 5, 8, requires_grad=True)

    with mock.patch(
        "torch.addmm",
        side_effect=AssertionError("the regressing linear addmm path was restored"),
    ):
        output = adapter(inputs)
        output.square().mean().backward()

    assert output.shape == (3, 5, 12)
    assert adapter.lora_B.grad is not None


def test_merged_checkpoint_matches_adapter_output() -> None:
    torch.manual_seed(7)
    model = TinyModel().eval()
    apply_lora(model, rank=2, alpha=4, dropout=0, targets=("q_proj",))
    with torch.no_grad():
        model.q_proj.lora_B.normal_()
    inputs = torch.randn(3, 4)
    expected = model(inputs)

    state = merged_model_state_dict(model)
    restored = TinyModel().eval()
    restored.load_state_dict(state)

    torch.testing.assert_close(restored(inputs), expected)
    assert not any("lora" in key or ".base." in key for key in state)


def test_resume_reconstructs_unmerged_adapter_without_double_counting() -> None:
    torch.manual_seed(11)
    model = TinyModel().eval()
    apply_lora(model, rank=2, alpha=4, dropout=0, targets=("q_proj",))
    with torch.no_grad():
        model.q_proj.lora_B.normal_()
    inputs = torch.randn(2, 4)
    expected = model(inputs)
    merged = merged_model_state_dict(model)
    adapter = adapter_state_dict(model)

    resumed = TinyModel().eval()
    resumed.load_state_dict(merged)
    apply_lora(resumed, rank=2, alpha=4, dropout=0, targets=("q_proj",))
    load_adapter_state_dict(resumed, adapter)
    unmerge_adapter_from_base(resumed)

    torch.testing.assert_close(resumed(inputs), expected)


def test_gqa_executes_qv_adapter_branches() -> None:
    torch.manual_seed(19)
    attention = GroupedQueryAttention(
        AttentionConfig(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            use_qk_norm=False,
            use_sdpa=False,
        )
    ).eval()
    inputs = torch.randn(2, 5, 16)
    baseline, _ = attention(inputs)
    apply_lora(attention, rank=2, alpha=4, dropout=0, targets=("q_proj", "v_proj"))
    initial, _ = attention(inputs)
    torch.testing.assert_close(initial, baseline)

    with torch.no_grad():
        attention.q_proj.lora_B.normal_()
    adapted, _ = attention(inputs)

    assert not torch.allclose(adapted, baseline)


def _lora_gqa(*, dropout: float = 0.0) -> GroupedQueryAttention:
    attention = GroupedQueryAttention(
        AttentionConfig(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            use_qk_norm=False,
            use_sdpa=False,
        )
    )
    apply_lora(
        attention,
        rank=2,
        alpha=4,
        dropout=dropout,
        targets=("k_proj", "q_proj", "v_proj"),
    )
    with torch.no_grad():
        for projection in (attention.k_proj, attention.q_proj, attention.v_proj):
            projection.lora_B.normal_()
    return attention


def test_gqa_fused_lora_kqv_matches_separate_forward_and_gradients() -> None:
    torch.manual_seed(31)
    fused = _lora_gqa().eval()
    reference = copy.deepcopy(fused).eval()
    fused_inputs = torch.randn(2, 5, 16, requires_grad=True)
    reference_inputs = fused_inputs.detach().clone().requires_grad_(True)

    actual = fused._project_kqv(fused_inputs)
    expected = tuple(
        projection(reference_inputs)
        for projection in (reference.k_proj, reference.q_proj, reference.v_proj)
    )
    for actual_projection, expected_projection in zip(actual, expected):
        torch.testing.assert_close(actual_projection, expected_projection)

    actual_loss = sum(projection.square().mean() for projection in actual)
    expected_loss = sum(projection.square().mean() for projection in expected)
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(fused_inputs.grad, reference_inputs.grad)
    for (_, actual_parameter), (_, expected_parameter) in zip(
        fused.named_parameters(),
        reference.named_parameters(),
    ):
        if actual_parameter.requires_grad:
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


def test_gqa_fused_lora_kqv_preserves_independent_dropout_streams() -> None:
    torch.manual_seed(37)
    attention = _lora_gqa(dropout=0.25).train()
    inputs = torch.randn(2, 5, 16)

    torch.manual_seed(101)
    expected = tuple(
        projection(inputs)
        for projection in (attention.k_proj, attention.q_proj, attention.v_proj)
    )
    torch.manual_seed(101)
    actual = attention._project_kqv(inputs)

    for actual_projection, expected_projection in zip(actual, expected):
        torch.testing.assert_close(actual_projection, expected_projection)


def test_gqa_fused_lora_kqv_bypasses_three_base_module_calls() -> None:
    attention = _lora_gqa().eval()
    inputs = torch.randn(2, 5, 16)
    projections = (attention.k_proj, attention.q_proj, attention.v_proj)

    with ExitStack() as stack:
        k, q, v = (
            stack.enter_context(
                mock.patch.object(projection.base, "forward", wraps=projection.base.forward)
            )
            for projection in projections
        )
        attention._project_kqv(inputs)

    assert (k.call_count, q.call_count, v.call_count) == (0, 0, 0)


def test_gqa_fused_lora_kqv_executes_each_adapter_residual_once() -> None:
    attention = _lora_gqa().eval()
    inputs = torch.randn(2, 5, 16)
    projections = (attention.k_proj, attention.q_proj, attention.v_proj)

    with ExitStack() as stack:
        residuals = tuple(
            stack.enter_context(
                mock.patch.object(
                    projection,
                    "lora_residual",
                    wraps=projection.lora_residual,
                )
            )
            for projection in projections
        )
        attention._project_kqv(inputs)

    assert tuple(residual.call_count for residual in residuals) == (1, 1, 1)


def _tiny_engine() -> TRHashEngine:
    return TRHashEngine(
        TRHashEngineConfig(
            hidden_size=8,
            vocab_size=32,
            num_experts=4,
            top_k=2,
            expert_width=4,
            shared_width=0,
            backend=TRHashBackend.PYTORCH,
            attention_backbone=AttentionBackbone.GQA,
        )
    )


def _tiny_shared_engine() -> TRHashEngine:
    return TRHashEngine(
        TRHashEngineConfig(
            hidden_size=8,
            vocab_size=32,
            num_experts=4,
            top_k=2,
            expert_width=4,
            shared_width=12,
            backend=TRHashBackend.PYTORCH,
            attention_backbone=AttentionBackbone.GQA,
        )
    )


def test_shared_lora_fused_gate_up_matches_reference_and_gradients() -> None:
    torch.manual_seed(41)
    fused = _tiny_shared_engine().eval()
    apply_lora(
        fused,
        rank=2,
        alpha=4,
        dropout=0,
        targets=("shared_gate", "shared_up", "shared_down"),
    )
    with torch.no_grad():
        for projection in (fused.shared_gate, fused.shared_up, fused.shared_down):
            projection.lora_B.normal_()
    reference = copy.deepcopy(fused)
    reference._fused_shared_gate_up_weight = None
    fused_inputs = torch.randn(11, 8, requires_grad=True)
    reference_inputs = fused_inputs.detach().clone().requires_grad_(True)

    actual = fused._shared(fused_inputs)
    expected = reference._shared(reference_inputs)
    torch.testing.assert_close(actual, expected)

    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(fused_inputs.grad, reference_inputs.grad)
    for (_, actual_parameter), (_, expected_parameter) in zip(
        fused.named_parameters(), reference.named_parameters()
    ):
        if actual_parameter.requires_grad:
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


def test_shared_lora_fused_gate_up_bypasses_two_base_calls() -> None:
    engine = _tiny_shared_engine().eval()
    apply_lora(
        engine,
        rank=2,
        alpha=4,
        dropout=0,
        targets=("shared_gate", "shared_up", "shared_down"),
    )
    inputs = torch.randn(11, 8)

    with ExitStack() as stack:
        gate = stack.enter_context(
            mock.patch.object(
                engine.shared_gate.base,
                "forward",
                wraps=engine.shared_gate.base.forward,
            )
        )
        up = stack.enter_context(
            mock.patch.object(
                engine.shared_up.base,
                "forward",
                wraps=engine.shared_up.base.forward,
            )
        )
        engine._shared(inputs)

    assert gate.call_count == 0
    assert up.call_count == 0


def test_expert_lora_preserves_engine_tensor_contract_and_trains_factors() -> None:
    torch.manual_seed(23)
    engine = _tiny_engine().eval()
    hidden = torch.randn(2, 5, 8)
    token_ids = torch.randint(0, 32, (2, 5))
    baseline = engine(hidden, token_ids)

    stats = apply_lora(
        engine,
        rank=2,
        alpha=4,
        dropout=0,
        targets=("expert_gate", "expert_up", "expert_down"),
    )
    initial = engine(hidden, token_ids)

    assert stats["linear_modules"] == 0
    assert stats["expert_tensors"] == 3
    assert engine.expert_gate.shape == (4, 8, 4)
    torch.testing.assert_close(initial, baseline)

    loss = initial.square().mean()
    loss.backward()
    assert engine.parametrizations.expert_gate[0].lora_B.grad is not None
    assert engine.parametrizations.expert_up[0].lora_B.grad is not None
    assert engine.parametrizations.expert_down[0].lora_B.grad is not None


def test_expert_lora_fuses_delta_materialization_with_baddbmm() -> None:
    engine = _tiny_engine()
    apply_lora(
        engine,
        rank=2,
        alpha=4,
        dropout=0,
        targets=("expert_gate",),
    )

    adapter = engine.parametrizations.expert_gate[0]
    with ExitStack() as stack:
        baddbmm = stack.enter_context(
            mock.patch("torch.baddbmm", wraps=torch.baddbmm)
        )
        stack.enter_context(
            mock.patch.object(
                adapter,
                "delta_weight",
                side_effect=AssertionError(
                    "dense LoRA delta was materialized separately"
                ),
            )
        )
        materialized = engine.expert_gate

    assert materialized.shape == (4, 8, 4)
    assert baddbmm.call_count == 1
    assert baddbmm.call_args.kwargs == {"beta": 1.0, "alpha": 2.0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fused_cuda_materializes_each_expert_lora_weight_once() -> None:
    from complexity.tr_hash.engine import HAS_FUSED_CUDA

    if not HAS_FUSED_CUDA:
        pytest.skip("hash-native Triton kernels are unavailable")
    engine = TRHashEngine(
        TRHashEngineConfig(
            hidden_size=16,
            vocab_size=257,
            num_experts=4,
            top_k=2,
            expert_width=8,
            shared_width=0,
            backend=TRHashBackend.FUSED_CUDA,
            attention_backbone=AttentionBackbone.GQA,
        )
    ).cuda().to(torch.bfloat16)
    apply_lora(
        engine,
        rank=2,
        alpha=4,
        dropout=0,
        targets=("expert_gate", "expert_up", "expert_down"),
    )
    adapters = [
        engine.parametrizations[name][0]
        for name in ("expert_gate", "expert_up", "expert_down")
    ]
    hidden = torch.randn(
        2,
        11,
        16,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    token_ids = torch.randint(0, 257, (2, 11), device="cuda")

    with ExitStack() as stack:
        gate, up, down = (
            stack.enter_context(
                mock.patch.object(adapter, "forward", wraps=adapter.forward)
            )
            for adapter in adapters
        )
        engine(hidden, token_ids).float().square().mean().backward()

    assert (gate.call_count, up.call_count, down.call_count) == (1, 1, 1)
    for adapter in adapters:
        assert adapter.lora_B.grad is not None


def test_expert_lora_merged_checkpoint_and_resume_are_exact() -> None:
    torch.manual_seed(29)
    engine = _tiny_engine().eval()
    apply_lora(
        engine,
        rank=2,
        alpha=4,
        dropout=0,
        targets=("expert_gate", "expert_up", "expert_down"),
    )
    with torch.no_grad():
        for tensor_name in ("expert_gate", "expert_up", "expert_down"):
            engine.parametrizations[tensor_name][0].lora_B.normal_()
    hidden = torch.randn(2, 5, 8)
    token_ids = torch.randint(0, 32, (2, 5))
    expected = engine(hidden, token_ids)
    merged = merged_model_state_dict(engine)
    adapter = adapter_state_dict(engine)

    assert set(merged).issuperset(
        {"expert_gate", "expert_up", "expert_down", "route_table"}
    )
    assert not any("parametrizations" in key or "lora_" in key for key in merged)

    restored = _tiny_engine().eval()
    restored.load_state_dict(merged)
    torch.testing.assert_close(restored(hidden, token_ids), expected)

    resumed = _tiny_engine().eval()
    resumed.load_state_dict(merged)
    apply_lora(
        resumed,
        rank=2,
        alpha=4,
        dropout=0,
        targets=("expert_gate", "expert_up", "expert_down"),
    )
    load_adapter_state_dict(resumed, adapter)
    unmerge_adapter_from_base(resumed)
    torch.testing.assert_close(resumed(hidden, token_ids), expected)


def test_expert_lora_uses_its_optimizer_lr_multiplier() -> None:
    engine = _tiny_engine()
    apply_lora(
        engine,
        rank=2,
        alpha=4,
        dropout=0,
        targets=("expert_gate",),
    )
    optimizer = build_optimizer(
        SimpleNamespace(
            lr=2e-5,
            expert_lr_multiplier=1.5,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.95,
        ),
        engine,
    )

    assert {group["name"] for group in optimizer.param_groups} == {"expert_decay"}
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
