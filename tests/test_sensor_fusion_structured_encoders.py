import torch

from complexity.generative.sensor_fusion.model import gradient_reverse
from complexity.generative.sensor_fusion.structured_encoders import (
    IMUDeviceGraphTokenizer,
    SkeletonGraphTokenizer,
)


def test_structured_encoders_preserve_batch_and_token_contract():
    skeleton = SkeletonGraphTokenizer(17, 3, hidden_size=64, token_count=8)
    imu = IMUDeviceGraphTokenizer(45, hidden_size=64, token_count=8)
    skeleton_output = skeleton(torch.randn(2, 16, 17, 3))
    imu_output = imu(torch.randn(2, 16, 45))
    assert skeleton_output.shape == (2, 8, 64)
    assert imu_output.shape == (2, 8, 64)
    (skeleton_output.mean() + imu_output.mean()).backward()
    assert skeleton.input_projection.weight.grad is not None
    assert imu.input_projection.weight.grad is not None


def test_gradient_reverse_changes_only_backward_direction():
    values = torch.tensor([1.0, 2.0], requires_grad=True)
    output = gradient_reverse(values, 0.25)
    assert torch.equal(output, values)
    output.sum().backward()
    assert torch.allclose(values.grad, torch.full_like(values, -0.25))
