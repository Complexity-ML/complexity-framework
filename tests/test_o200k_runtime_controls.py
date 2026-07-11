import torch
import torch.nn as nn

from complexity.training.o200k.runtime import runtime_controls


class LexicalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.object_output_gate = nn.Parameter(torch.tensor(0.2))
        self.micro_output_gate = nn.Parameter(torch.tensor(0.3))


class TokenRoutedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_output_gate = nn.Parameter(torch.tensor(1.0))
        self.routed_output_gate = nn.Parameter(torch.tensor(0.1))

    def set_top_k_primary_weight(self, value):
        self.primary_weight = value


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
