from scripts.onnx_detect import provider_names


def test_provider_aliases_expand_with_fallbacks() -> None:
    assert provider_names(None) == ("CPUExecutionProvider",)
    assert provider_names(["cuda"]) == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    assert provider_names(["tensorrt"]) == (
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )


def test_provider_names_accept_repeated_and_explicit_values() -> None:
    assert provider_names(["cuda,CPUExecutionProvider", "CustomProvider"]) == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
        "CustomProvider",
    )
