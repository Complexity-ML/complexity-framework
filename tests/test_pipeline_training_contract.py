"""The pipeline prototype must never train on its former placeholder mean."""

import pytest
import torch

from complexity.parallel.pipeline_parallel import PipelineModel


def _pipeline():
    source = torch.nn.Module()
    source.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3)])
    return PipelineModel(source, num_stages=1, num_micro_batches=2)


def test_pipeline_training_rejects_placeholder_loss_before_model_execution():
    pipeline = _pipeline()
    before = {key: value.clone() for key, value in pipeline.state_dict().items()}
    forwards = []
    pipeline.stage.register_forward_hook(lambda *args: forwards.append(True))
    with pytest.raises(NotImplementedError, match="causal-LM target/loss integration"):
        pipeline(torch.randn(4, 3), labels=torch.zeros(4, 3, dtype=torch.long))
    assert not forwards
    assert all(parameter.grad is None for parameter in pipeline.parameters())
    for key, value in pipeline.state_dict().items():
        assert torch.equal(value, before[key])


def test_pipeline_explicit_hidden_state_stage_execution_is_preserved():
    pipeline = _pipeline()
    hidden = torch.randn(4, 3)
    torch.testing.assert_close(pipeline(hidden), pipeline.stage(hidden))
