from __future__ import annotations

import torch
import torch.nn as nn

from complexity.core.attention.base import AttentionConfig
from complexity.core.attention.gqa import GroupedQueryAttention
from complexity.training.lora import (
    adapter_state_dict,
    apply_lora,
    load_adapter_state_dict,
    merged_model_state_dict,
    unmerge_adapter_from_base,
)


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
