"""
Model API - Flexible avec defaults.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import torch

from complexity.config import ModelConfig, get_preset, PRESET_CONFIGS
from complexity.models import ComplexityModel


class Model:
    """
    Model flexible.

    Defaults sensibles, TOUT overridable via **kwargs.

    Examples:
        # Simple
        model = Model.load("llama-7b")

        # Override
        model = Model.load("llama-7b", device="cuda", dtype=torch.bfloat16)
        model = Model.create(hidden_size=2048, num_layers=32, mlp_type="gated")

        # Accès direct au nn.Module
        model.module.parameters()
    """

    def __init__(self, model: ComplexityModel, config: ModelConfig, **kwargs):
        self._model = model
        self._config = config
        self._tokenizer = kwargs.get("tokenizer")
        self._device = kwargs.get("device", "cpu")

    @property
    def module(self) -> ComplexityModel:
        """nn.Module PyTorch direct."""
        return self._model

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def device(self) -> str:
        return self._device

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    # ==================== Load ====================

    @classmethod
    def load(cls, path: str, **kwargs) -> "Model":
        """
        Charge modèle. Override tout via kwargs.

        Args:
            path: Preset ou chemin checkpoint
            **kwargs: Override (device, dtype, + tout param de ModelConfig)
        """
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)

        # Preset?
        if path in PRESET_CONFIGS:
            config = get_preset(path)
            # kwargs override config
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

            model = ComplexityModel(config)
            if dtype:
                model = model.to(dtype)

            print(f"[Model] Loaded preset '{path}' ({sum(p.numel() for p in model.parameters()):,} params)")

            result = cls(model, config, device=device or "cpu")
            if device:
                result.to(device)
            return result

        # Checkpoint
        p = Path(path)
        config_file = p / "config.json" if p.is_dir() else p.parent / "config.json"

        if not config_file.exists():
            raise ValueError(f"Config not found: {config_file}")

        with open(config_file) as f:
            cfg = json.load(f)
        # kwargs override
        cfg.update(kwargs)
        config = ModelConfig(**cfg)

        model = ComplexityModel(config)

        weights = p / "model.pt" if p.is_dir() else p
        if weights.exists():
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            print(f"[Model] Loaded from {weights}")

        if dtype:
            model = model.to(dtype)

        result = cls(model, config, device=device or "cpu")
        if device:
            result.to(device)
        return result

    # ==================== Create ====================

    @classmethod
    def create(cls, config: ModelConfig = None, **kwargs) -> "Model":
        """
        Crée modèle custom. Passe config ou kwargs ou les deux.

        Args:
            config: Config (optionnel)
            **kwargs: Override tout (hidden_size, num_layers, device, ...)
        """
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)

        if config:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        else:
            config = ModelConfig(**kwargs)

        model = ComplexityModel(config)
        if dtype:
            model = model.to(dtype)

        print(f"[Model] Created ({sum(p.numel() for p in model.parameters()):,} params)")

        result = cls(model, config, device=device or "cpu")
        if device:
            result.to(device)
        return result

    # ==================== Forward / External Serving Boundary ====================

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward. kwargs passés au modèle."""
        return self._model(input_ids.to(self._device), **kwargs)

    def __call__(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        return self.forward(input_ids, **kwargs)

    def generate(self, input_ids, max_tokens: int = 100, **kwargs):
        """
        Disabled by architecture contract.

        Complexity's Python/PyTorch layer builds, trains, and exports models;
        generation/serving must run through vLLM or SGLang.
        """
        raise RuntimeError(
            "Model.generate is disabled: use a vLLM or SGLang serving backend "
            "for generation."
        )

    def chat(self, messages: List[Dict], max_tokens: int = 500, **kwargs) -> Dict[str, Any]:
        """Disabled: chat generation must run through vLLM or SGLang."""
        raise RuntimeError(
            "Model.chat is disabled: use a vLLM or SGLang serving backend for chat."
        )

    # ==================== Utils ====================

    def to(self, device: str) -> "Model":
        self._device = device
        self._model.to(device)
        return self

    def half(self) -> "Model":
        self._model.half()
        return self

    def bfloat16(self) -> "Model":
        self._model.to(torch.bfloat16)
        return self

    def eval(self) -> "Model":
        self._model.eval()
        return self

    def train_mode(self) -> "Model":
        self._model.train()
        return self

    def set_tokenizer(self, tok) -> None:
        self._tokenizer = tok

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path / "model.pt")
        with open(path / "config.json", "w") as f:
            json.dump(self._config.__dict__, f, indent=2, default=str)
        print(f"[Model] Saved to {path}")

    def __repr__(self):
        return f"Model(params={self.num_parameters:,}, device='{self._device}')"
