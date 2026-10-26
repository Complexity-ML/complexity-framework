"""Lossless Hugging Face bundles for the canonical TR-HASH architecture."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from complexity._optional import require_optional


def _require_model_type(model_type: str, *, allow_auto: bool = False) -> None:
    if model_type != "tr_hash_moe" and not (allow_auto and model_type == "auto"):
        raise ValueError(
            "TR-HASH requires model_type='tr_hash_moe'; dense weight remapping was removed"
        )


def _is_main_process() -> bool:
    import torch.distributed as dist

    return not dist.is_initialized() or dist.get_rank() == 0


def _validate_checkpoint_config(raw: dict) -> None:
    from complexity.export.transformers import validate_transformers_export_config

    if raw.get("model_type", "tr_hash_moe") != "tr_hash_moe":
        raise ValueError("Checkpoint is not a canonical TR-HASH model")
    if raw.get("mlp_type") not in {"tr_hash_engine", "tr_hash_moe"}:
        raise ValueError("Checkpoint must explicitly declare the TR-HASH architecture")
    validate_transformers_export_config(raw)


def _checkpoint_directory(model_name_or_path: str, **hub_options) -> Path:
    path = Path(model_name_or_path).expanduser()
    if path.is_dir():
        raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
        _validate_checkpoint_config(raw)
        return path
    if path.exists():
        raise ValueError("Expected a TR-HASH checkpoint directory")
    hub = require_optional("huggingface_hub", "hf", feature="Loading a TR-HASH checkpoint")
    config_path = hub.hf_hub_download(
        repo_id=str(model_name_or_path), filename="config.json", **hub_options
    )
    _validate_checkpoint_config(json.loads(Path(config_path).read_text(encoding="utf-8")))
    return Path(
        hub.snapshot_download(
            repo_id=str(model_name_or_path),
            allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin", "model.pt"],
            **hub_options,
        )
    )


def convert_hf_state_dict(
    hf_state_dict: Dict[str, torch.Tensor],
    model_type: str = "tr_hash_moe",
    num_layers: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Preserve every tensor; native and Transformers TR-HASH share their layout."""
    _require_model_type(model_type)
    return dict(hf_state_dict)


def convert_to_hf_state_dict(
    complexity_state_dict: Dict[str, torch.Tensor],
    model_type: str = "tr_hash_moe",
    num_layers: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Preserve expert weights, shared weights, norms, and routing anchors."""
    _require_model_type(model_type)
    return dict(complexity_state_dict)


def create_hf_config(config: Any, model_type: str = "tr_hash_moe") -> Dict[str, Any]:
    """Describe the standalone TR-HASH adapter without changing its architecture."""
    from complexity.export.transformers import build_transformers_config

    _require_model_type(model_type)
    raw = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return build_transformers_config(raw)


def save_to_huggingface(
    model: nn.Module,
    save_path: str,
    model_type: str = "tr_hash_moe",
    config: Optional[Any] = None,
) -> str:
    """Save complete native weights with the autonomous Transformers adapter."""
    from complexity.export.transformers import export_bundle, validate_transformers_export_config
    from complexity.models import ComplexityModel

    _require_model_type(model_type)
    if not isinstance(model, ComplexityModel):
        raise ValueError("Hugging Face export requires a canonical ComplexityModel")
    raw = model.export_config()
    validate_transformers_export_config(raw)
    if config is not None:
        supplied = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        if supplied != raw:
            raise ValueError("Export config must match the complete live TR-HASH configuration")
    with tempfile.TemporaryDirectory(prefix="tr_hash_hf_") as temporary:
        source = Path(temporary)
        model.save_pretrained(source)
        if not _is_main_process():
            return str(save_path)
        export_bundle(
            config_path=source / "config.json",
            weights_path=source / "model.safetensors",
            output=Path(save_path),
        )
    return str(save_path)


def load_from_huggingface(
    model_name_or_path: str,
    model: nn.Module,
    model_type: str = "tr_hash_moe",
    device: str = "cpu",
) -> nn.Module:
    """Load a complete canonical checkpoint; never fill absent experts at random."""
    from complexity.models import ComplexityModel

    _require_model_type(model_type)
    if not isinstance(model, ComplexityModel):
        raise ValueError("Hugging Face loading requires a canonical ComplexityModel")
    path = _checkpoint_directory(model_name_or_path)
    restored = ComplexityModel.from_pretrained(path)
    if restored.export_config() != model.export_config():
        raise ValueError("Checkpoint architecture differs from the target TR-HASH model")
    model.load_state_dict(restored.state_dict(), strict=True)
    return model.to(device)


def from_pretrained(
    model_name_or_path: str,
    model_type: str = "auto",
    device: str = "cpu",
    **kwargs,
) -> Tuple[nn.Module, Any]:
    """Load a local or Hub TR-HASH checkpoint through the strict native loader."""
    from complexity.models import ComplexityModel

    _require_model_type(model_type, allow_auto=True)
    dtype = kwargs.pop("dtype", None)
    allowed = {"revision", "token", "cache_dir", "local_files_only", "force_download"}
    if unknown := set(kwargs) - allowed:
        raise ValueError("Unknown checkpoint options: " + ", ".join(sorted(unknown)))
    path = _checkpoint_directory(model_name_or_path, **kwargs)
    model = ComplexityModel.from_pretrained(path, device=device, dtype=dtype)
    return model, model.config


def push_to_hub(
    model: nn.Module,
    repo_id: str,
    model_type: str = "tr_hash_moe",
    config: Optional[Any] = None,
    private: bool = False,
    token: Optional[str] = None,
):
    """Publish a fully validated canonical bundle when explicitly invoked."""
    with tempfile.TemporaryDirectory(prefix="tr_hash_hub_") as temporary:
        save_to_huggingface(model, temporary, model_type, config)
        if not _is_main_process():
            return None
        hub = require_optional("huggingface_hub", "hf", feature="Publishing to Hugging Face")
        hub.create_repo(repo_id, private=private, token=token, exist_ok=True)
        return hub.HfApi(token=token).upload_folder(
            folder_path=temporary, repo_id=repo_id, token=token
        )
