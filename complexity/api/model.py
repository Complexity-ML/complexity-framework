"""Small model wrapper over ComplexityModel and its native export format."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from complexity.config import PRESET_CONFIGS, ModelConfig, get_preset
from complexity.models import ComplexityModel


def _configured(config: ModelConfig, overrides: Dict[str, Any]) -> ModelConfig:
    """Copy caller configuration and validate overrides before allocating weights."""
    valid = {field.name for field in fields(ModelConfig)}
    unknown = overrides.keys() - valid
    if unknown:
        raise ValueError(f"Unknown ModelConfig fields: {', '.join(sorted(unknown))}")
    # replace invokes ModelConfig validation; deepcopy preserves tensor-valued
    # legacy frequency tables without sharing mutable config objects.
    return replace(deepcopy(config), **overrides)


def _on_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _on_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_on_device(item, device) for item in value)
    if isinstance(value, list):
        return [_on_device(item, device) for item in value]
    return value


class Model:
    """Convenience wrapper; ``module`` exposes the canonical PyTorch model.

    ``create`` and preset ``load`` initialize fresh weights. Loading a local
    export requires config.json and actual model weights. Use ``save`` to
    export weights, and Trainer checkpoints to preserve optimizer/run state.
    """

    def __init__(self, model: ComplexityModel, config: ModelConfig, **kwargs):
        self._model = model
        self._config = config
        self._tokenizer = kwargs.get("tokenizer")
        self._device = str(kwargs.get("device", "cpu"))

    @property
    def module(self) -> ComplexityModel:
        return self._model

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def device(self) -> str:
        """Current module device, including moves made by a trainer."""
        return str(next(self._model.parameters()).device)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self._model.parameters())

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "Model":
        """Load local native/legacy weights, or initialize a named preset.

        Supports native model.safetensors/pytorch_model.bin directories and
        historical model.pt files/directories. ModelConfig keyword overrides
        are validated and copied; device/dtype control placement separately.
        This method does not download models from a remote repository.
        """
        device = kwargs.pop("device", "cpu")
        dtype = kwargs.pop("dtype", None)
        if str(path) in PRESET_CONFIGS:
            return cls.create(get_preset(str(path)), device=device, dtype=dtype, **kwargs)

        checkpoint = Path(path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
        config_path = (checkpoint if checkpoint.is_dir() else checkpoint.parent) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        if checkpoint.is_dir() and not any(
            (checkpoint / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin", "model.pt")
        ):
            raise FileNotFoundError(f"No model weights found in {checkpoint}")

        config = _configured(ModelConfig.load(str(config_path)), kwargs)
        model = ComplexityModel(config)
        model._load_pretrained_weights(checkpoint)
        if dtype is not None:
            model.to(dtype=dtype)
        return cls(model, config).to(device or "cpu")

    @classmethod
    def create(cls, config: Optional[ModelConfig] = None, **kwargs) -> "Model":
        """Initialize fresh weights from a copied, validated configuration."""
        device = kwargs.pop("device", "cpu")
        dtype = kwargs.pop("dtype", None)
        if config is None:
            config = ModelConfig.from_dict(kwargs, strict=True)
        elif not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        else:
            config = _configured(config, kwargs)
        model = ComplexityModel(config)
        if dtype is not None:
            model.to(dtype=dtype)
        return cls(model, config).to(device or "cpu")

    def forward(self, input_ids: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        """Move tensor inputs to the module's device and run its forward pass."""
        return self._model(_on_device(input_ids, self.device), **_on_device(kwargs, self.device))

    def __call__(self, input_ids: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        return self.forward(input_ids, **kwargs)

    def generate(self, input_ids, max_tokens: int = 100, **kwargs):
        raise RuntimeError(
            "Model.generate is disabled: use complexity.inference.create_external_backend "
            "with a vLLM or SGLang serving endpoint."
        )

    def chat(self, messages: List[Dict], max_tokens: int = 500, **kwargs) -> Dict[str, Any]:
        raise RuntimeError(
            "Model.chat is disabled: use complexity.inference.create_external_backend "
            "with a vLLM or SGLang serving endpoint."
        )

    def to(self, device: Union[str, torch.device]) -> "Model":
        self._model.to(device)
        self._device = str(device)
        return self

    def half(self) -> "Model":
        self._model.half()
        return self

    def bfloat16(self) -> "Model":
        self._model.to(dtype=torch.bfloat16)
        return self

    def eval(self) -> "Model":
        self._model.eval()
        return self

    def train_mode(self) -> "Model":
        self._model.train()
        return self

    def set_tokenizer(self, tok) -> None:
        self._tokenizer = tok
        self._model.set_tokenizer(tok)

    def save(self, path: Union[str, Path], *, safe_serialization: bool = True) -> None:
        """Export weights/config using the same format as save_pretrained.

        This is a model export. Resuming a training run requires an optimizer
        and training-state checkpoint, available from Trainer.save_checkpoint.
        Distributed callers must call this on every participating rank.
        """
        self._model.save_pretrained(path, safe_serialization=safe_serialization)

    def __repr__(self):
        return f"Model(params={self.num_parameters:,}, device='{self.device}')"
