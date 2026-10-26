"""CGGR receives the same computed contextual routes as the PyTorch path."""

from dataclasses import replace
import pytest
import torch
from complexity.tr_hash import TRHashBackend, TRHashEngine, TRHashEngineConfig
from complexity.tr_hash.capabilities import _custom_cggr_available


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _custom_cggr_available(), reason="CUDA CGGR is unavailable"
)
def test_dynamic_contextual_cggr_matches_reference_values_and_gradients():
    config = TRHashEngineConfig(
        hidden_size=32,
        vocab_size=256,
        shared_width=64,
        expert_width=16,
        backend=TRHashBackend.PYTORCH,
    )
    reference = TRHashEngine(config).cuda().bfloat16()
    actual = TRHashEngine(replace(config, backend=TRHashBackend.CGGR)).cuda().bfloat16()
    actual.load_state_dict(reference.state_dict())
    ids = torch.tensor([[5, 5, 5, 19, 2, 3], [9, 9, 9, 2, 3, 5]], device="cuda")
    history = torch.tensor([[1, 9], [3, 8]], device="cuda")
    x = torch.randn(2, 6, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = reference(x, ids, history)
    observed = actual(y, ids, history)
    torch.testing.assert_close(observed, expected, rtol=0.04, atol=0.02)
    expected.float().square().sum().backward()
    observed.float().square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0.06, atol=0.03)
