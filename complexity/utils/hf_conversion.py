"""
HuggingFace Conversion utilities.

Convert between framework-complexity models and HuggingFace transformers.

Supports:
- Loading HF models into complexity format
- Saving complexity models to HF format
- Push to HuggingFace Hub
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Weight Mapping
# =============================================================================

# Mapping from HuggingFace Llama to Complexity
HF_TO_COMPLEXITY_LLAMA = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
    # Layer mappings (with {i} placeholder for layer index)
    "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj.weight",
    "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj.weight",
    "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj.weight",
    "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj.weight",
    "model.layers.{i}.mlp.gate_proj.weight": "layers.{i}.mlp.gate_proj.weight",
    "model.layers.{i}.mlp.up_proj.weight": "layers.{i}.mlp.up_proj.weight",
    "model.layers.{i}.mlp.down_proj.weight": "layers.{i}.mlp.down_proj.weight",
    "model.layers.{i}.input_layernorm.weight": "layers.{i}.input_layernorm.weight",
    "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.post_attention_layernorm.weight",
}

# Mapping from HuggingFace Mistral (same as Llama for most parts)
HF_TO_COMPLEXITY_MISTRAL = HF_TO_COMPLEXITY_LLAMA.copy()

# Mapping from HuggingFace GPT-2 to Complexity
HF_TO_COMPLEXITY_GPT2 = {
    "wte.weight": "embed_tokens.weight",
    "wpe.weight": "position_embedding.weight",
    "ln_f.weight": "norm.weight",
    "ln_f.bias": "norm.bias",
    # Layer mappings
    "h.{i}.ln_1.weight": "layers.{i}.input_layernorm.weight",
    "h.{i}.ln_1.bias": "layers.{i}.input_layernorm.bias",
    "h.{i}.attn.c_attn.weight": "layers.{i}.self_attn.qkv_proj.weight",  # Fused QKV
    "h.{i}.attn.c_attn.bias": "layers.{i}.self_attn.qkv_proj.bias",
    "h.{i}.attn.c_proj.weight": "layers.{i}.self_attn.o_proj.weight",
    "h.{i}.attn.c_proj.bias": "layers.{i}.self_attn.o_proj.bias",
    "h.{i}.ln_2.weight": "layers.{i}.post_attention_layernorm.weight",
    "h.{i}.ln_2.bias": "layers.{i}.post_attention_layernorm.bias",
    "h.{i}.mlp.c_fc.weight": "layers.{i}.mlp.up_proj.weight",
    "h.{i}.mlp.c_fc.bias": "layers.{i}.mlp.up_proj.bias",
    "h.{i}.mlp.c_proj.weight": "layers.{i}.mlp.down_proj.weight",
    "h.{i}.mlp.c_proj.bias": "layers.{i}.mlp.down_proj.bias",
}


def get_weight_mapping(model_type: str) -> Dict[str, str]:
    """Get weight mapping for a model type."""
    mappings = {
        "llama": HF_TO_COMPLEXITY_LLAMA,
        "mistral": HF_TO_COMPLEXITY_MISTRAL,
        "gpt2": HF_TO_COMPLEXITY_GPT2,
    }
    if model_type not in mappings:
        raise ValueError(f"Unknown model type: {model_type}. Supported: {list(mappings.keys())}")
    return mappings[model_type]


# =============================================================================
# Loading from HuggingFace
# =============================================================================

def convert_hf_state_dict(
    hf_state_dict: Dict[str, torch.Tensor],
    model_type: str,
    num_layers: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert HuggingFace state dict to Complexity format.

    Args:
        hf_state_dict: HuggingFace model state dict
        model_type: Type of model (llama, mistral, gpt2)
        num_layers: Number of transformer layers

    Returns:
        State dict in Complexity format
    """
    mapping = get_weight_mapping(model_type)
    complexity_state_dict = {}

    for hf_key, tensor in hf_state_dict.items():
        # Try to find mapping
        matched = False

        for hf_pattern, complexity_pattern in mapping.items():
            if "{i}" in hf_pattern:
                # Layer-specific mapping
                for i in range(num_layers):
                    hf_key_i = hf_pattern.format(i=i)
                    if hf_key == hf_key_i:
                        complexity_key = complexity_pattern.format(i=i)
                        complexity_state_dict[complexity_key] = tensor
                        matched = True
                        break
            else:
                if hf_key == hf_pattern:
                    complexity_state_dict[complexity_pattern] = tensor
                    matched = True
                    break

            if matched:
                break

        if not matched:
            logger.warning(f"Unmapped key: {hf_key}")

    return complexity_state_dict


def load_from_huggingface(
    model_name_or_path: str,
    model: nn.Module,
    model_type: str = "llama",
    device: str = "cpu",
) -> nn.Module:
    """
    Load weights from a HuggingFace model into a Complexity model.

    Args:
        model_name_or_path: HuggingFace model name or local path
        model: Complexity model to load weights into
        model_type: Type of model (llama, mistral, gpt2)
        device: Device to load weights to

    Returns:
        Model with loaded weights

    Example:
        from complexity import ComplexityModel, ModelConfig
        from complexity.utils.hf_conversion import load_from_huggingface

        config = get_preset("llama-7b")
        model = ComplexityModel(config)
        model = load_from_huggingface("meta-llama/Llama-2-7b-hf", model)
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise ImportError("transformers is required. Install with: pip install transformers")

    # Load HF model
    logger.info(f"Loading HuggingFace model: {model_name_or_path}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    # Get number of layers
    num_layers = len(model.layers)

    # Convert state dict
    hf_state_dict = hf_model.state_dict()
    complexity_state_dict = convert_hf_state_dict(hf_state_dict, model_type, num_layers)

    # Load into model
    missing, unexpected = model.load_state_dict(complexity_state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    logger.info(f"Loaded {len(complexity_state_dict)} tensors from HuggingFace model")

    # Clean up
    del hf_model

    return model.to(device)


# =============================================================================
# Saving to HuggingFace
# =============================================================================

def convert_to_hf_state_dict(
    complexity_state_dict: Dict[str, torch.Tensor],
    model_type: str,
    num_layers: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert Complexity state dict to HuggingFace format.

    Args:
        complexity_state_dict: Complexity model state dict
        model_type: Type of model
        num_layers: Number of layers

    Returns:
        State dict in HuggingFace format
    """
    mapping = get_weight_mapping(model_type)
    # Reverse the mapping
    reverse_mapping = {v: k for k, v in mapping.items()}

    hf_state_dict = {}

    for complexity_key, tensor in complexity_state_dict.items():
        matched = False

        for complexity_pattern, hf_pattern in reverse_mapping.items():
            if "{i}" in complexity_pattern:
                for i in range(num_layers):
                    complexity_key_i = complexity_pattern.format(i=i)
                    if complexity_key == complexity_key_i:
                        hf_key = hf_pattern.format(i=i)
                        hf_state_dict[hf_key] = tensor
                        matched = True
                        break
            else:
                if complexity_key == complexity_pattern:
                    hf_state_dict[hf_pattern] = tensor
                    matched = True
                    break

            if matched:
                break

        if not matched:
            logger.warning(f"Unmapped key: {complexity_key}")

    return hf_state_dict


def save_to_huggingface(
    model: nn.Module,
    save_path: str,
    model_type: str = "llama",
    config: Optional[Any] = None,
) -> str:
    """
    Save a Complexity model in HuggingFace format.

    Args:
        model: Complexity model to save
        save_path: Path to save the model
        model_type: Type of model
        config: Model config

    Returns:
        Path to saved model

    Example:
        from complexity.utils.hf_conversion import save_to_huggingface

        save_to_huggingface(model, "my_model_hf", model_type="llama")
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    num_layers = len(model.layers)

    # Convert state dict
    complexity_state_dict = model.state_dict()
    hf_state_dict = convert_to_hf_state_dict(complexity_state_dict, model_type, num_layers)

    # Save weights
    torch.save(hf_state_dict, save_path / "pytorch_model.bin")

    # Save config
    if config is not None:
        hf_config = create_hf_config(config, model_type)
        with open(save_path / "config.json", 'w') as f:
            json.dump(hf_config, f, indent=2)

    logger.info(f"Saved model to {save_path}")
    return str(save_path)


def create_hf_config(config: Any, model_type: str) -> Dict[str, Any]:
    """Create HuggingFace config from Complexity config."""
    if model_type in ["llama", "mistral"]:
        return {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "hidden_act": config.hidden_act,
            "max_position_embeddings": config.max_position_embeddings,
            "rms_norm_eps": config.norm_eps,
            "vocab_size": config.vocab_size,
            "rope_theta": config.rope_theta,
            "tie_word_embeddings": config.tie_word_embeddings,
        }
    elif model_type == "gpt2":
        return {
            "architectures": ["GPT2LMHeadModel"],
            "model_type": "gpt2",
            "n_embd": config.hidden_size,
            "n_layer": config.num_hidden_layers,
            "n_head": config.num_attention_heads,
            "n_positions": config.max_position_embeddings,
            "vocab_size": config.vocab_size,
        }
    else:
        return config.to_dict() if hasattr(config, 'to_dict') else {}


# =============================================================================
# HuggingFace Hub Integration
# =============================================================================

def push_to_hub(
    model: nn.Module,
    repo_id: str,
    model_type: str = "llama",
    config: Optional[Any] = None,
    private: bool = False,
    token: Optional[str] = None,
):
    """
    Push a Complexity model to the HuggingFace Hub.

    Args:
        model: Model to push
        repo_id: Repository ID (e.g., "username/model-name")
        model_type: Type of model
        config: Model config
        private: Whether to make the repo private
        token: HuggingFace token

    Example:
        from complexity.utils.hf_conversion import push_to_hub

        push_to_hub(model, "myuser/my-complexity-model")
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

    import tempfile

    # Create repo if it doesn't exist
    api = HfApi(token=token)
    try:
        create_repo(repo_id, private=private, token=token, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create repo: {e}")

    # Save model to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        save_to_huggingface(model, tmpdir, model_type, config)

        # Upload
        api.upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            token=token,
        )

    logger.info(f"Pushed model to: https://huggingface.co/{repo_id}")


def from_pretrained(
    model_name_or_path: str,
    model_type: str = "auto",
    device: str = "cuda",
    **kwargs,
) -> Tuple[nn.Module, Any]:
    """
    Load a model from HuggingFace Hub or local path.

    Args:
        model_name_or_path: Model name or path
        model_type: Model type (auto-detected if "auto")
        device: Device to load to
        **kwargs: Additional arguments

    Returns:
        (model, config) tuple

    Example:
        from complexity.utils.hf_conversion import from_pretrained

        model, config = from_pretrained("meta-llama/Llama-2-7b-hf")
    """
    from complexity import ComplexityModel, ModelConfig

    # Auto-detect model type
    if model_type == "auto":
        if "llama" in model_name_or_path.lower():
            model_type = "llama"
        elif "mistral" in model_name_or_path.lower():
            model_type = "mistral"
        elif "gpt2" in model_name_or_path.lower():
            model_type = "gpt2"
        else:
            model_type = "llama"  # Default

    # Get config from HuggingFace
    try:
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_name_or_path)

        # Convert to Complexity config
        config = ModelConfig(
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads),
            intermediate_size=hf_config.intermediate_size,
            vocab_size=hf_config.vocab_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            rope_theta=getattr(hf_config, 'rope_theta', 10000.0),
            norm_eps=getattr(hf_config, 'rms_norm_eps', 1e-6),
        )
    except Exception as e:
        logger.warning(f"Could not load HF config: {e}")
        from complexity.config import get_preset
        config = get_preset(f"{model_type}-7b")

    # Create model
    model = ComplexityModel(config)

    # Load weights
    model = load_from_huggingface(model_name_or_path, model, model_type, device)

    return model, config
