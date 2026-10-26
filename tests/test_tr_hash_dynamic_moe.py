"""The canonical model can vary expert channel width, never expert count."""

import pytest
import torch
from complexity.tr_hash import TRHashEngine, TRHashEngineConfig


def test_reduced_width_gradients_and_full_width_restore():
    engine = TRHashEngine(TRHashEngineConfig(hidden_size=16, vocab_size=64, expert_width=8))
    x = torch.randn(2, 5, 16)
    ids = torch.tensor([[1, 9, 17, 23, 29], [4, 2, 11, 12, 15]])
    before = engine(x, ids).detach()
    engine.set_active_capacity(expert_width=3)
    reduced = engine(x, ids)
    assert not torch.allclose(before, reduced)
    reduced.square().sum().backward()
    assert torch.count_nonzero(engine.expert_gate.grad[:, :, 3:]) == 0
    engine.set_active_capacity(expert_width=8)
    torch.testing.assert_close(engine(x, ids), before, rtol=0, atol=0)


@pytest.mark.parametrize("count", [1, 2, 3, 8, 16])
def test_number_of_experts_cannot_select_a_different_architecture(count):
    engine = TRHashEngine(TRHashEngineConfig(hidden_size=8, vocab_size=32))
    with pytest.raises(ValueError, match="four active experts"):
        engine.set_active_capacity(num_experts=count)
