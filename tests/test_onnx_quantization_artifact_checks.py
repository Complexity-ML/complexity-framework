import pytest

from scripts.check_onnx_quantized_artifacts import (
    check_provider_precision_supported,
    check_unexpected_fp32_nodes,
)


def test_unsupported_provider_precision_fails_clearly() -> None:
    thresholds = {"providers": {"CPUExecutionProvider": ["fp32", "int8"]}}

    with pytest.raises(ValueError, match="does not support fp16"):
        check_provider_precision_supported("CPUExecutionProvider", "fp16", thresholds)


def test_unknown_provider_fails_clearly() -> None:
    thresholds = {"providers": {"CPUExecutionProvider": ["fp32", "int8"]}}

    with pytest.raises(ValueError, match="not configured"):
        check_provider_precision_supported("MagicExecutionProvider", "fp16", thresholds)


def test_unexpected_fp32_nodes_are_reported() -> None:
    report = {
        "fp32_nodes": [
            {"name": "Conv_1", "op_type": "Conv"},
            {"name": "ReduceSum_1", "op_type": "ReduceSum"},
        ]
    }

    unexpected = check_unexpected_fp32_nodes(report, allowlist=["ReduceSum"])

    assert unexpected == ["Conv_1:Conv"]
