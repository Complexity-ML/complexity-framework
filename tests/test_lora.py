from __future__ import annotations

from types import SimpleNamespace

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
