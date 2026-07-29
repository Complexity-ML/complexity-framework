"""
MPS inference engine with shared online-RL monitoring.

This is the Apple-Silicon runtime wrapper around ``SharedOnlineRLLoop``. It
loads a local Complexity checkpoint on ``mps``, serves prompts, accumulates
shared RL monitor events, and periodically writes a new shared RL checkpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from complexity.config import ModelConfig
from complexity.models import ComplexityModel
from complexity.tokenizer import Tokenizer
from complexity.utils.device import configure_torch_acceleration, is_mps_available
from complexity.utils.mps import setup_mps

from .shared_online_rl import SharedOnlineRLConfig, SharedOnlineRLLoop
from .tool_rewards import build_verified_tool_episode


logger = logging.getLogger(__name__)


@dataclass
class MPSOnlineRLEngineConfig:
    """Configuration for the local MPS online-RL inference engine."""

    checkpoint: str | Path
    tokenizer: str | Path
    output_dir: str | Path
    min_events_before_update: int = 16
    learning_rate: float = 1e-7
    rl_epochs: int = 1
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 50
    reload_after_save: bool = True
    seed: Optional[int] = None


class MPSOnlineRLEngine:
    """
    Local Apple-Silicon inference engine with shared RL updates.

    A single engine instance is meant to be shared by the local server process.
    User requests add compact monitor events; when the shared threshold is hit,
    the engine runs one full-model RL update, saves a checkpoint, reloads it
    by default, and keeps serving from that RL checkpoint.
    """

    def __init__(self, config: MPSOnlineRLEngineConfig):
        self.config = config
        self.device = setup_mps(seed=config.seed)
        if self.device.type != "mps":
            raise RuntimeError(
                "MPSOnlineRLEngine requires an Apple MPS device. "
                f"Selected device was {self.device}."
            )
        if not is_mps_available():
            raise RuntimeError("torch.backends.mps is not available in this process")

        configure_torch_acceleration(kernel_policy=False, log=False)
        self.checkpoint_path = Path(config.checkpoint)
        self.checkpoint_dir, state = self._load_checkpoint_state(self.checkpoint_path)
        self.model_config = ModelConfig.from_dict(state["config"])
        self._restore_semantic_routing_from_checkpoint(state)
        self.model_config.use_custom_kernels = False
        self.model = ComplexityModel(self.model_config).to(self.device)
        self._validate_semantic_routing()
        missing, unexpected = self.model.load_state_dict(state["model"], strict=False)
        # topk_token_to_expert is a deterministic buffer recomputed from
        # token_to_expert + config in TokenRoutedMLP.__init__. Older
        # checkpoints predate this buffer; tolerate it as missing.
        tolerated_suffix = "mlp.topk_token_to_expert"
        unexpected_missing = [k for k in missing if not k.endswith(tolerated_suffix)]
        if unexpected_missing or unexpected:
            raise RuntimeError(
                "Error(s) in loading state_dict for ComplexityModel:\n"
                f"  Missing key(s): {unexpected_missing}\n"
                f"  Unexpected key(s): {list(unexpected)}"
            )
        self.model.eval()
        self.tokenizer = Tokenizer.load(str(config.tokenizer))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        rl_config = SharedOnlineRLConfig(
            output_dir=config.output_dir,
            min_events_before_update=config.min_events_before_update,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            learning_rate=config.learning_rate,
            rl_epochs=config.rl_epochs,
            reload_after_save=config.reload_after_save,
        )
        self.loop = SharedOnlineRLLoop(
            model=self.model,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            config=rl_config,
            device=self.device,
            extra_state_fn=self._checkpoint_extra_state,
            reload_fn=self.reload_checkpoint,
        )

    def infer(
        self,
        prompt: str,
        user_feedback: Optional[str] = None,
        force_monitor: bool = False,
        force_update: bool = False,
        **generation_overrides,
    ) -> tuple[str, Optional[Dict[str, float]], Optional[Path]]:
        """Serve one prompt through the MPS shared online-RL loop."""

        return self.loop.infer(
            prompt,
            user_feedback=user_feedback,
            force_monitor=force_monitor,
            force_update=force_update,
            **generation_overrides,
        )

    def infer_with_verified_tools(
        self,
        question: str,
        force_update: bool = False,
        **generation_overrides,
    ) -> tuple[str, Optional[Dict[str, float]], Optional[Path]]:
        """
        Serve a question and add verified tool-use supervision when possible.

        For supported tool-shaped prompts, the monitor stores a positive
        verified tool trajectory as ``target_response``. The generated response
        is still returned to the caller, but RL trains on the verified trace.
        """

        prompt = f"User:\n{question}\n\nAssistant:\n"
        verified = build_verified_tool_episode(prompt=prompt, question=question)
        if verified is None:
            return self.infer(prompt, force_update=force_update, **generation_overrides)

        return self.loop.infer(
            prompt,
            target_response=verified.target_response,
            reward=verified.reward,
            tool_calls=[verified.tool_call],
            tool_results=[verified.tool_result],
            metadata=verified.metadata,
            force_monitor=True,
            force_update=force_update,
            **generation_overrides,
        )

    def reload_checkpoint(self, checkpoint_dir: Path) -> None:
        """Reload the served model/optimizer from a freshly saved RL checkpoint."""

        _, state = self._load_checkpoint_state(checkpoint_dir)
        self.model.load_state_dict(state["model"], strict=True)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.model.eval()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = checkpoint_dir / "checkpoint.pt"

    def _restore_semantic_routing_from_checkpoint(self, state: dict) -> None:
        """Preserve hybrid LSH routing even when older checkpoints only saved CLI args.

        Modern checkpoints carry these fields in ``config``. During the early
        semantic-routing experiments, some inference paths relied on ``args``;
        this fallback makes the MPS runtime explicit and prevents a semantic
        checkpoint from silently serving as plain lexical Zipf TR.
        """

        raw_config = state.get("config", {})
        raw_args = state.get("args", {})
        routing_strategy = raw_config.get("routing_strategy") or raw_args.get("routing_strategy")
        if routing_strategy != "lsh_hidden":
            return

        self.model_config.routing_strategy = "lsh_hidden"
        self.model_config.lsh_routing = True
        self.model_config.lsh_from_layer = int(
            raw_config.get("lsh_from_layer", raw_args.get("lsh_from_layer", 0))
        )
        self.model_config.lsh_bits = int(
            raw_config.get("lsh_bits", raw_args.get("lsh_bits", 0))
        )
        self.model_config.lsh_threshold_mode = str(
            raw_config.get("lsh_threshold_mode", raw_args.get("lsh_threshold_mode", "zero"))
        )

    def _validate_semantic_routing(self) -> None:
        if getattr(self.model_config, "routing_strategy", "zipf") != "lsh_hidden":
            return

        lsh_layers = [
            idx for idx, layer in enumerate(getattr(self.model, "layers", []))
            if getattr(getattr(layer, "mlp", None), "has_lsh_routing", False)
        ]
        if not lsh_layers:
            raise RuntimeError(
                "Checkpoint config requests routing_strategy='lsh_hidden', but no "
                "TokenRoutedMLP layer enabled semantic LSH routing."
            )
        logger.info(
            "MPS semantic TR routing enabled: lexical layers < %s, LSH layers=%s, threshold=%s",
            getattr(self.model_config, "lsh_from_layer", 0),
            lsh_layers,
            getattr(self.model_config, "lsh_threshold_mode", "zero"),
        )

    def _checkpoint_extra_state(self) -> dict:
        return {
            "config": self.model_config.to_dict(),
            "source_checkpoint": str(self.checkpoint_dir),
            "backend": {
                "backend": "mps",
                "device": str(self.device),
            },
        }

    @staticmethod
    def _load_checkpoint_state(path: str | Path) -> tuple[Path, dict]:
        ckpt = Path(path)
        if ckpt.is_file():
            return ckpt.parent, torch.load(ckpt, map_location="cpu", weights_only=False)
        checkpoint_file = ckpt / "checkpoint.pt"
        if checkpoint_file.exists():
            return ckpt, torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        latest = ckpt / "latest"
        if latest.exists():
            target = latest.read_text(encoding="utf-8").strip()
            if target:
                return MPSOnlineRLEngine._load_checkpoint_state(ckpt / target)
        raise FileNotFoundError(f"No checkpoint.pt found under {ckpt}")


__all__ = ["MPSOnlineRLEngine", "MPSOnlineRLEngineConfig"]
