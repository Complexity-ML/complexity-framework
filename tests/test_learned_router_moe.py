import torch

from complexity.core.mlp import MLPConfig, MixtralMoE, TokenRoutedMLP
from complexity.training.o200k.runtime import learned_router_aux_loss


def _config(**overrides):
    values = {
        "hidden_size": 16,
        "intermediate_size": 16,
        "num_experts": 4,
        "vocab_size": 32,
        "top_k": 2,
        "shared_expert": True,
        "shared_intermediate_size": 24,
    }
    values.update(overrides)
    return MLPConfig(**values)


def test_learned_router_matches_fixed_expert_and_shared_shapes():
    learned = MixtralMoE(_config())
    fixed = TokenRoutedMLP(_config(routing_strategy="modulo"))

    assert learned.gate_proj_w.shape == fixed.gate_proj_w.shape
    assert learned.up_proj_w.shape == fixed.up_proj_w.shape
    assert learned.down_proj_w.shape == fixed.down_proj_w.shape
    assert learned.shared_gate.weight.shape == fixed.shared_gate.weight.shape
    assert learned.shared_up.weight.shape == fixed.shared_up.weight.shape
    assert learned.shared_down.weight.shape == fixed.shared_down.weight.shape

    learned_count = sum(parameter.numel() for parameter in learned.parameters())
    fixed_count = sum(parameter.numel() for parameter in fixed.parameters())
    assert learned_count - fixed_count == 16 * 4


def test_learned_router_preserves_common_initialization_for_same_seed():
    torch.manual_seed(42)
    fixed = TokenRoutedMLP(_config(routing_strategy="modulo"))
    fixed_next_random = torch.rand(3)

    torch.manual_seed(42)
    learned = MixtralMoE(_config())
    learned_next_random = torch.rand(3)

    assert torch.equal(learned.gate_proj_w, fixed.gate_proj_w)
    assert torch.equal(learned.up_proj_w, fixed.up_proj_w)
    assert torch.equal(learned.down_proj_w, fixed.down_proj_w)
    assert torch.equal(learned.shared_gate.weight, fixed.shared_gate.weight)
    assert torch.equal(learned.shared_up.weight, fixed.shared_up.weight)
    assert torch.equal(learned.shared_down.weight, fixed.shared_down.weight)
    assert torch.equal(learned_next_random, fixed_next_random)


def test_learned_top2_forward_and_auxiliary_loss_are_differentiable():
    torch.manual_seed(7)
    learned = MixtralMoE(_config())
    hidden = torch.randn(2, 5, 16, requires_grad=True)

    output = learned(hidden)
    auxiliary = learned_router_aux_loss(learned)

    assert output.shape == hidden.shape
    assert auxiliary is not None
    assert auxiliary.requires_grad
    (output.square().mean() + 0.01 * auxiliary).backward()
    assert learned.router.weight.grad is not None
    assert torch.isfinite(learned.router.weight.grad).all()


def test_learned_top2_dispatch_does_not_require_balanced_assignments():
    learned = MixtralMoE(_config(shared_expert=False))
    with torch.no_grad():
        learned.router.weight.zero_()
        learned.router.weight[0].fill_(1.0)
        learned.router.weight[1].fill_(0.5)

    hidden = torch.ones(1, 7, 16)
    output = learned(hidden)

    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
