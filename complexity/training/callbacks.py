"""Training callbacks."""

import logging
import math
from typing import Optional

import torch

from ..parallel.data_parallel import is_main_process

logger = logging.getLogger(__name__)


class EarlyStoppingCallback:
    """Early stopping callback."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, trainer, step: int, loss: float):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping at step {step}")
                trainer.global_step = trainer.config.max_steps


class WandBCallback:
    """Weights & Biases logging callback."""

    def __init__(self, project: str, name: Optional[str] = None):
        try:
            import wandb
            self.wandb = wandb
            if is_main_process():
                wandb.init(project=project, name=name)
        except ImportError:
            self.wandb = None
            logger.warning("wandb not installed, skipping W&B logging")

    def __call__(self, trainer, step: int, loss: float):
        if self.wandb and is_main_process():
            ppl = math.exp(min(loss, 20))
            self.wandb.log({
                "loss": loss,
                "ppl": ppl,
                "lr": trainer.scheduler.get_last_lr()[0],
                "step": step,
            })


class TqdmCallback:
    """tqdm progress bar callback with loss + PPL + MoE telemetry.

    Postfix auto-populated with (when available):
      - loss, ppl, lr
      - γ (routed_gate, TokenRoutedMLP)
      - E shares (expert utilization, reduced across ranks)
      - per-expert diagnostics from MuonTR/AdamTR

    **Must be registered on ALL ranks** — global_expert_shares() issues an
    all_reduce collective. The display itself is rank-0 only.
    """

    def __init__(self, total_steps: int, desc: str = "train"):
        from tqdm import tqdm
        self.pbar = tqdm(
            total=total_steps, desc=desc, unit="step", dynamic_ncols=True,
            disable=not is_main_process(),
        )

    def __call__(self, trainer, step: int, loss: float):
        # Collective reductions — ALL ranks must participate at the same step.
        from .moe_telemetry import gamma_mean, global_expert_shares
        shares, dead = global_expert_shares(trainer.model)
        gamma = gamma_mean(trainer.model) if shares else float("nan")

        # Cache so later callbacks (CSV loggers) can read without triggering
        # a second collective on already-reset counters.
        self.last_shares = shares
        self.last_dead = dead
        self.last_gamma = gamma

        if not is_main_process():
            return

        ppl = math.exp(min(loss, 20))
        lr = trainer.scheduler.get_last_lr()[0]
        postfix = {"loss": f"{loss:.4f}", "ppl": f"{ppl:.1f}", "lr": f"{lr:.2e}"}

        # MoE-specific telemetry (only if the model has TokenRoutedMLP layers)
        if shares:
            postfix["γ"] = f"{gamma:.3f}"
            postfix["E"] = "/".join(f"{s:.2f}" for s in shares)
            if dead > 0:
                postfix["dead"] = str(dead)

        # Per-expert MuonTR/AdamTR diagnostics — local view (no collective).
        # These reflect rank 0's shard only under FSDP, kept for live insight.
        opt = trainer.optimizer
        muon_inner = getattr(opt, "muon_tr", None) or (opt if hasattr(opt, "get_ns_diagnostics") else None)
        if muon_inner is not None and hasattr(muon_inner, "get_ns_diagnostics"):
            try:
                diags = muon_inner.get_ns_diagnostics()
                if diags:
                    num_experts = max(diags.keys()) + 1
                    gn_values = [diags[e]["grad_norm"] for e in range(num_experts)]
                    postfix["E-gn"] = "/".join(f"{v:.1e}" for v in gn_values)
                    postfix["E-lr×"] = "/".join(f"{diags[e]['lr_ratio']:.2f}" for e in sorted(diags))
            except Exception as e:
                if not getattr(self, "_diag_warned", False):
                    print(f"[TqdmCallback] diagnostics disabled: {type(e).__name__}: {e}", flush=True)
                    self._diag_warned = True

        self.pbar.set_postfix(**postfix, ordered=True)
        self.pbar.update(1)

    def close(self):
        self.pbar.close()


class TensorBoardCallback:
    """TensorBoard logging callback."""

    def __init__(self, log_dir: str = "runs"):
        try:
            from torch.utils.tensorboard import SummaryWriter
            if is_main_process():
                self.writer = SummaryWriter(log_dir)
            else:
                self.writer = None
        except ImportError:
            self.writer = None
            logger.warning("tensorboard not installed")

    def __call__(self, trainer, step: int, loss: float):
        if self.writer:
            ppl = math.exp(min(loss, 20))
            self.writer.add_scalar("Loss/train", loss, step)
            self.writer.add_scalar("PPL/train", ppl, step)
            self.writer.add_scalar("LR", trainer.scheduler.get_last_lr()[0], step)
