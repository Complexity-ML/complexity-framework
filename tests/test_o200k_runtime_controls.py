import torch
import torch.nn as nn

from complexity.core.mlp.base import MLPConfig
from complexity.core.mlp.lexical_object_micro_expert import LexicalObjectMicroExpertMLP
from complexity.training.o200k.runtime import runtime_controls


class LexicalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.object_output_gate = nn.Parameter(torch.tensor(0.2))
        self.micro_output_gate = nn.Parameter(torch.tensor(0.3))

    def training_control_capabilities(self):
        return frozenset({"lexical_object_gate", "micro_expert_gate"})

    def training_telemetry(self):
        return {"object_gate": 0.2, "micro_gate": 0.3}


class TokenRoutedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_output_gate = nn.Parameter(torch.tensor(1.0))
        self.routed_output_gate = nn.Parameter(torch.tensor(0.1))

    def set_top_k_primary_weight(self, value):
        self.primary_weight = value

    def training_control_capabilities(self):
        return frozenset({"topk_primary_weight", "shared_routed_gates"})

    def training_telemetry(self):
        return {"topk_w": 0.5, "shared_gate": 1.0, "routed_gate": 0.1}


def test_runtime_controls_detect_lexical_model_without_topk_controls():
    model = nn.Sequential(LexicalLayer(), LexicalLayer())

    controls = runtime_controls(model)

    assert controls.token_routed_layers == 0
    assert controls.lexical_layers == 2
    assert abs(controls.object_gate - 0.2) < 1e-6
    assert abs(controls.micro_gate - 0.3) < 1e-6


def test_runtime_controls_detect_token_routed_model():
    model = nn.Sequential(TokenRoutedLayer())

    controls = runtime_controls(model)

    assert controls.token_routed_layers == 1
    assert controls.lexical_layers == 0
    assert controls.object_gate is None
    assert controls.micro_gate is None
    assert controls.capabilities == frozenset(
        {"topk_primary_weight", "shared_routed_gates"}
    )
    assert controls.telemetry == {
        "topk_w": 0.5,
        "shared_gate": 1.0,
        "routed_gate": 0.1,
    }


def test_lexical_module_declares_its_own_training_controls_and_telemetry():
    module = LexicalObjectMicroExpertMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            vocab_size=64,
            lexical_object_rank=4,
            micro_num_experts=2,
            micro_expert_width=4,
        )
    )

    assert module.training_control_capabilities() == frozenset(
        {"lexical_object_gate", "micro_expert_gate"}
    )
    telemetry = module.training_telemetry()
    assert abs(telemetry["object_gate"] - 0.1) < 1e-6
    assert abs(telemetry["micro_gate"] - 0.1) < 1e-6
