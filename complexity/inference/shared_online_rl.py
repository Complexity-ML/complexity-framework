"""
Shared online-RL monitor for served models.

The goal is intentionally simple:

1. inference keeps serving users;
2. uncertain/incorrect episodes are stored as compact shared monitor events;
3. once enough events accumulate, one RL update is run on the shared model;
4. one new checkpoint is saved and becomes the current served state.

This avoids creating one checkpoint per user request. The pending events are
small JSON records, and log-probabilities are recomputed only when the shared
RL update is triggered.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F


ExtraStateFn = Callable[[], Dict[str, Any]]
ReloadFn = Callable[[Path], None]


@dataclass
class SharedRLEvent:
    """Compact monitor record shared across users."""

    prompt: str
    response: str
    target_response: Optional[str] = None
    feedback: Optional[str] = None
    reward: Optional[float] = None
    mean_entropy: Optional[float] = None
    mean_confidence: Optional[float] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    created_at_unix: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SharedOnlineRLConfig:
    """Configuration for shared online RL during serving."""

    output_dir: str | Path
    min_events_before_update: int = 16
    max_pending_events: int = 128
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 50
    entropy_trigger: float = 5.0
    confidence_trigger: float = 0.20
    positive_reward: float = 1.0
    negative_reward: float = -1.0
    uncertainty_reward: float = -0.2
    baseline_beta: float = 0.95
    learning_rate: float = 1e-7
    rl_epochs: int = 1
    max_grad_norm: float = 1.0
    checkpoint_prefix: str = "checkpoint_online_rl"
    monitor_log_name: str = "shared_rl_events.jsonl"
    current_pointer_name: str = "current_online_rl_checkpoint.txt"
    save_optimizer: bool = True
    reload_after_save: bool = False
    prefer_target_response: bool = True
    verified_target_weight: float = 1.0
    keep_last_n_checkpoints: int = 5  # 0 = keep all


class SharedOnlineRLLoop:
    """
    Serve, monitor, periodically self-RL, and checkpoint.

    The loop is shared across users. Individual requests only append compact
    events. A checkpoint is created only after ``min_events_before_update``
    pending events have accumulated.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[SharedOnlineRLConfig] = None,
        device: Optional[str | torch.device] = None,
        extra_state_fn: Optional[ExtraStateFn] = None,
        reload_fn: Optional[ReloadFn] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SharedOnlineRLConfig(output_dir="checkpoints/online_rl")
        self.device = torch.device(device) if device is not None else self._infer_device()
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        self.extra_state_fn = extra_state_fn
        self.reload_fn = reload_fn
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_log = self.output_dir / self.config.monitor_log_name
        self.pending_events: List[SharedRLEvent] = []
        self.baseline = 0.0
        self.rl_step = self._infer_next_step()

    def infer(
        self,
        prompt: str,
        user_feedback: Optional[str] = None,
        target_response: Optional[str] = None,
        reward: Optional[float] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        force_monitor: bool = False,
        force_update: bool = False,
        **generation_overrides: Any,
    ) -> tuple[str, Optional[Dict[str, float]], Optional[Path]]:
        """
        Generate a response and maybe trigger one shared RL update.

        Returns ``response, stats, checkpoint_path``. ``stats`` and
        ``checkpoint_path`` are ``None`` if the shared buffer has not reached
        the update threshold yet.
        """

        response, entropy, confidence = self._generate_with_stats(prompt, **generation_overrides)
        event = SharedRLEvent(
            prompt=prompt,
            response=response,
            target_response=target_response,
            feedback=user_feedback,
            mean_entropy=entropy,
            mean_confidence=confidence,
            reward=reward if reward is not None else self._compute_reward(user_feedback, entropy, confidence),
            tool_calls=tool_calls or [],
            tool_results=tool_results or [],
            metadata=metadata or {},
        )

        should_monitor = force_monitor or self._should_monitor(event)
        if should_monitor:
            self._append_event(event)

        stats = None
        checkpoint_path = None
        if force_update or self._ready_for_update():
            stats = self._run_shared_rl_update()
            checkpoint_path = self._save_checkpoint(stats)
            self._write_current_pointer(checkpoint_path)
            self.pending_events.clear()
            if self.config.reload_after_save:
                if self.reload_fn is None:
                    raise RuntimeError("reload_after_save=True requires a reload_fn")
                self.reload_fn(checkpoint_path)

        return response, stats, checkpoint_path

    def _run_shared_rl_update(self) -> Dict[str, float]:
        events = list(self.pending_events[-self.config.max_pending_events :])
        if not events:
            return {"num_events": 0.0, "loss": 0.0}

        self.model.train()
        rewards = [float(event.reward if event.reward is not None else 0.0) for event in events]
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        last_loss = torch.zeros((), device=self.device)
        last_grad_norm = torch.zeros((), device=self.device)
        rl_epochs = max(1, int(self.config.rl_epochs))

        for _ in range(rl_epochs):
            losses = []
            for event, reward in zip(events, rewards):
                train_response = self._training_response(event)
                logprob = self._response_logprob(event.prompt, train_response)
                if self._uses_verified_target(event):
                    # Verified tool traces are supervised positive targets:
                    # keep their gradient strong instead of letting the RL
                    # baseline shrink it toward zero after repeated epochs.
                    losses.append(-(self.config.verified_target_weight * reward * logprob))
                else:
                    self.baseline = self.config.baseline_beta * self.baseline + (
                        1.0 - self.config.baseline_beta
                    ) * reward
                    advantage = reward - self.baseline
                    losses.append(-(advantage * logprob))

            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            last_loss = loss.detach()
            last_grad_norm = grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else torch.tensor(grad_norm)
        self.model.eval()

        return {
            "num_events": float(len(events)),
            "rl_epochs": float(rl_epochs),
            "optimizer_steps": float(rl_epochs),
            "mean_reward": float(reward_tensor.mean().item()),
            "baseline": float(self.baseline),
            "loss": float(last_loss.cpu()),
            "grad_norm": float(last_grad_norm.cpu()),
        }

    def _training_response(self, event: SharedRLEvent) -> str:
        if self._uses_verified_target(event):
            return event.target_response
        return event.response

    def _uses_verified_target(self, event: SharedRLEvent) -> bool:
        return (
            self.config.prefer_target_response
            and event.target_response is not None
            and event.reward is not None
            and event.reward > 0
        )

    def _response_logprob(self, prompt: str, response: str) -> torch.Tensor:
        prompt_ids = self._encode(prompt)
        response_ids = self._encode(response)
        if response_ids.numel() == 0:
            return torch.zeros((), device=self.device)

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        labels = input_ids[:, 1:]
        out = self.model(input_ids[:, :-1])
        logits = self._logits_from_output(out)
        logprobs = F.log_softmax(logits, dim=-1)

        prompt_len = prompt_ids.shape[1]
        response_start = max(prompt_len - 1, 0)
        response_labels = labels[:, response_start:]
        response_logprobs = logprobs[:, response_start:, :].gather(
            -1,
            response_labels.unsqueeze(-1),
        )
        return response_logprobs.squeeze(-1).mean()

    @torch.no_grad()
    def _generate_with_stats(self, prompt: str, **overrides: Any) -> tuple[str, float, float]:
        max_new_tokens = int(overrides.get("max_new_tokens", self.config.max_new_tokens))
        temperature = float(overrides.get("temperature", self.config.temperature))
        top_k = int(overrides.get("top_k", self.config.top_k))

        self.model.eval()
        generated = self._encode(prompt)
        response_tokens = []
        entropies = []
        confidences = []

        for _ in range(max_new_tokens):
            out = self.model(generated)
            logits = self._logits_from_output(out)[:, -1, :] / max(temperature, 1e-6)
            filtered = self._top_k_filter(logits, top_k)
            probs = F.softmax(filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            confidence = probs.max(dim=-1).values

            response_tokens.append(next_token)
            entropies.append(entropy.squeeze(0))
            confidences.append(confidence.squeeze(0))
            generated = torch.cat([generated, next_token], dim=1)

        response_ids = torch.cat(response_tokens, dim=1) if response_tokens else generated[:, 0:0]
        response = self._decode(response_ids)
        mean_entropy = float(torch.stack(entropies).mean().cpu()) if entropies else 0.0
        mean_confidence = float(torch.stack(confidences).mean().cpu()) if confidences else 1.0
        return response, mean_entropy, mean_confidence

    def _save_checkpoint(self, stats: Dict[str, float]) -> Path:
        self.rl_step += 1
        checkpoint_dir = self.output_dir / f"{self.config.checkpoint_prefix}_{self.rl_step:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=False)

        online_rl = {
            "rl_step": self.rl_step,
            "created_at_unix": time.time(),
            "stats": stats,
            "num_pending_events": len(self.pending_events),
            "config": asdict(self.config),
        }
        payload: Dict[str, Any] = {
            "model": self.model.state_dict(),
            "online_rl": online_rl,
        }
        if self.config.save_optimizer:
            payload["optimizer"] = self.optimizer.state_dict()
        if self.extra_state_fn is not None:
            payload.update(self.extra_state_fn())

        torch.save(payload, checkpoint_dir / "checkpoint.pt")
        with (checkpoint_dir / "online_rl.json").open("w", encoding="utf-8") as f:
            json.dump(online_rl, f, indent=2, sort_keys=True, default=str)
        self._prune_old_checkpoints()
        return checkpoint_dir

    def _prune_old_checkpoints(self) -> None:
        keep = self.config.keep_last_n_checkpoints
        if keep <= 0:
            return
        pattern = f"{self.config.checkpoint_prefix}_*"
        dirs = []
        for path in self.output_dir.glob(pattern):
            if not path.is_dir():
                continue
            suffix = path.name.removeprefix(f"{self.config.checkpoint_prefix}_")
            if suffix.isdigit():
                dirs.append((int(suffix), path))
        dirs.sort()
        for _, path in dirs[:-keep]:
            for f in path.iterdir():
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                path.rmdir()
            except OSError:
                pass

    def _append_event(self, event: SharedRLEvent) -> None:
        self.pending_events.append(event)
        if len(self.pending_events) > self.config.max_pending_events:
            self.pending_events = self.pending_events[-self.config.max_pending_events :]
        with self.monitor_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False, sort_keys=True) + "\n")

    def _should_monitor(self, event: SharedRLEvent) -> bool:
        if event.feedback and event.feedback.strip():
            return True
        if event.mean_entropy is not None and event.mean_entropy >= self.config.entropy_trigger:
            return True
        if (
            event.mean_confidence is not None
            and event.mean_confidence <= self.config.confidence_trigger
        ):
            return True
        return False

    def _ready_for_update(self) -> bool:
        return len(self.pending_events) >= self.config.min_events_before_update

    def _compute_reward(
        self,
        feedback: Optional[str],
        entropy: Optional[float],
        confidence: Optional[float],
    ) -> float:
        if feedback:
            text = feedback.lower()
            negative = ("wrong", "false", "bad", "non", "faux", "incorrect")
            positive = ("correct", "good", "yes", "oui", "exact")
            if any(word in text for word in negative):
                return self.config.negative_reward
            if any(word in text for word in positive):
                return self.config.positive_reward
        if entropy is not None and entropy >= self.config.entropy_trigger:
            return self.config.uncertainty_reward
        if confidence is not None and confidence <= self.config.confidence_trigger:
            return self.config.uncertainty_reward
        return 0.0

    def _encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if hasattr(ids, "ids"):
            ids = ids.ids
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _decode(self, ids: torch.Tensor) -> str:
        values = ids.detach().cpu().tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        try:
            return self.tokenizer.decode(values, skip_special_tokens=True)
        except TypeError:
            return self.tokenizer.decode(values)

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0 or top_k >= logits.size(-1):
            return logits
        cutoff = torch.topk(logits, top_k)[0][..., -1, None]
        return logits.masked_fill(logits < cutoff, float("-inf"))

    @staticmethod
    def _logits_from_output(output: Any) -> torch.Tensor:
        if hasattr(output, "logits"):
            return output.logits
        if isinstance(output, dict) and "logits" in output:
            return output["logits"]
        if isinstance(output, (tuple, list)):
            return output[0]
        raise TypeError("Model output must expose logits")

    def _infer_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _write_current_pointer(self, checkpoint_path: Path) -> None:
        pointer = self.output_dir / self.config.current_pointer_name
        tmp = pointer.with_suffix(".tmp")
        tmp.write_text(str(checkpoint_path), encoding="utf-8")
        os.replace(tmp, pointer)

    def _infer_next_step(self) -> int:
        existing = []
        pattern = f"{self.config.checkpoint_prefix}_*"
        for path in self.output_dir.glob(pattern):
            if not path.is_dir():
                continue
            suffix = path.name.removeprefix(f"{self.config.checkpoint_prefix}_")
            if suffix.isdigit():
                existing.append(int(suffix))
        return max(existing, default=0)
