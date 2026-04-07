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
    """tqdm progress bar callback with loss + PPL + per-expert MuonTR diagnostics.

    If the optimizer exposes get_ns_diagnostics() (MuonTR), the per-expert
    grad-norms and adaptive LR ratios are added to the postfix every step
    so you can watch the routed experts train live without scrolling.
    """

    def __init__(self, total_steps: int, desc: str = "train"):
        from tqdm import tqdm
        self.pbar = tqdm(total=total_steps, desc=desc, unit="step", dynamic_ncols=True)

    def __call__(self, trainer, step: int, loss: float):
        # Compute the per-expert grad-norm sync on ALL ranks first (collective
        # ops cannot be called from a subset of ranks — that deadlocks NCCL).
        # Then only rank 0 actually updates the bar.
        gn_values = None
        lr_values = None
        opt = trainer.optimizer
        muon_inner = getattr(opt, "muon_tr", None) or (opt if hasattr(opt, "get_ns_diagnostics") else None)
        if muon_inner is not None and hasattr(muon_inner, "get_ns_diagnostics"):
            try:
                diags = muon_inner.get_ns_diagnostics()
                if diags:
                    num_experts = max(diags.keys()) + 1
                    gn_values = [diags[e]["grad_norm"] for e in range(num_experts)]
                    lr_values = [diags[e]["lr_ratio"] for e in range(num_experts)]
                    # All-reduce (MAX) so each rank's local-view gets merged
                    # into the global per-expert picture. Called from EVERY rank.
                    if (torch.distributed.is_initialized()
                            and torch.distributed.get_world_size() > 1):
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        gn_tensor = torch.tensor(gn_values, dtype=torch.float32, device=device)
                        torch.distributed.all_reduce(gn_tensor, op=torch.distributed.ReduceOp.MAX)
                        gn_values = gn_tensor.tolist()
            except Exception as e:
                if not getattr(self, "_diag_warned", False):
                    print(f"[TqdmCallback] diagnostics disabled: {type(e).__name__}: {e}", flush=True)
                    self._diag_warned = True

        if not is_main_process():
            return

        ppl = math.exp(min(loss, 20))
        lr = trainer.scheduler.get_last_lr()[0]
        postfix = {"loss": f"{loss:.4f}", "ppl": f"{ppl:.1f}", "lr": f"{lr:.2e}"}
        if gn_values is not None:
            postfix["E-gn"] = "/".join(f"{v:.1e}" for v in gn_values)
        if lr_values is not None:
            postfix["E-lr×"] = "/".join(f"{v:.2f}" for v in lr_values)

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
