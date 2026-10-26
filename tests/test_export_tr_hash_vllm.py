"""Custom table-only serving layouts cannot masquerade as composed TR-HASH."""
import pytest
from complexity.export.routed import build_vllm_config, vllm_tensor_name


@pytest.mark.parametrize("converter", [build_vllm_config, vllm_tensor_name])
def test_static_vllm_conversion_requires_complete_architecture_support(converter):
    with pytest.raises(ValueError, match="cannot represent composed"):
        converter({"routing_strategy":"token_id_composed_modulo"})
