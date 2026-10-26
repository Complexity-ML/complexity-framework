"""Static layouts must not silently discard the composed architecture."""
import pytest
from complexity.export.routed import build_mlx_config, remap_state_dict, remap_hf_state_dict


@pytest.mark.parametrize("converter", [build_mlx_config, remap_state_dict, remap_hf_state_dict])
def test_static_mlx_conversion_rejects_composed_architecture(converter):
    with pytest.raises(ValueError, match="cannot represent composed"):
        converter({"routing_strategy":"token_id_composed_modulo"}, dtype="float32")
