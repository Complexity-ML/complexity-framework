"""Explicit refusal of custom layouts that omit causal routing or routed attention.

Native checkpoints and autonomous Transformers bundles preserve the complete
TR-HASH architecture. A static expert table cannot encode its two-token context,
query-head selection and value-channel binding.
"""
from .assets import copy_tokenizer_files


def _unsupported(*args, **kwargs):
    raise ValueError("This static export layout cannot represent composed TR-HASH context routes, selected attention heads and V channel binding; use complexity.export.transformers.export_bundle.")


build_mlx_config = _unsupported
build_vllm_config = _unsupported
remap_state_dict = _unsupported
remap_hf_state_dict = _unsupported
vllm_tensor_name = _unsupported
__all__ = ["build_mlx_config", "build_vllm_config", "remap_state_dict", "remap_hf_state_dict", "vllm_tensor_name", "copy_tokenizer_files"]
