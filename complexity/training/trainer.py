"""
Integrated Trainer for framework-complexity.

Combines FSDP, mixed precision, gradient accumulation, checkpointing,
and learning rate scheduling into a single training loop.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import logging
import time
import math
import warnings

warnings.filterwarnings("ignore", message=".*epoch parameter in.*scheduler.step.*")

from ..parallel.data_parallel import (
    wrap_model_fsdp,
    ShardingMode,
    PrecisionMode,
    init_distributed,
    get_rank,
    get_world_size,
    is_main_process,
)
from ..utils.checkpointing import CheckpointManager, TrainingState
from ..utils.security import AuditLogger, SecureTrainingContext

from .config import TrainingConfig
from .scheduler import get_lr_scheduler, resolve_scheduler_name
from .metrics import MetricsTracker

logger = logging.getLogger(__name__)


class Trainer:
    """
    Integrated trainer for framework-complexity models.

    Usage:
        trainer = Trainer(model=model, config=config, train_dataloader=loader)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        compute_loss: Optional[Callable] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.callbacks = callbacks or []

        # Initialize distributed
        self.distributed = init_distributed()
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_main = is_main_process()

        # Device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        else:
            self.device = torch.device("cpu")

        # Wrap model with FSDP if enabled
        if config.use_fsdp and self.distributed:
            precision = PrecisionMode.BF16 if config.precision == "bf16" else (
                PrecisionMode.FP16 if config.precision == "fp16" else PrecisionMode.FP32
            )
            sharding = ShardingMode(config.sharding_mode)
            gc_enabled = getattr(model, '_gradient_checkpointing', False)
            self.model = wrap_model_fsdp(
                model,
                sharding_mode=sharding,
                precision=precision,
                gradient_checkpointing=gc_enabled,
            )
        else:
            self.model = model.to(self.device)

        # LR auto-scaling REMOVED: the sqrt(batch) heuristic silently
        # multiplied the user's --lr by up to 3× on multi-GPU, causing
        # instability on MoE models. The user now controls the actual LR.
        if self.is_main:
            effective_batch = config.batch_size * self.world_size * config.gradient_accumulation_steps
            logger.info("=" * 70)
            logger.info("TRAINING CONFIG")
            logger.info("=" * 70)
            logger.info(f"  optimizer       : {config.optimizer_type}")
            logger.info(f"  learning_rate   : {config.learning_rate:.2e}  (NO auto-scaling — value is used as-is)")
            if config.optimizer_type in ("muon", "muon_tr"):
                logger.info(f"  muon_lr         : {config.muon_lr:.2e}")
            if config.optimizer_type in ("muon_tr", "adam_tr"):
                logger.info(f"  expert_lr_scale : ×{config.expert_lr_scale}  "
                            f"(effective expert LR: {config.learning_rate * config.expert_lr_scale:.2e})")
            logger.info(f"  weight_decay    : {config.weight_decay}")
            logger.info(f"  warmup_steps    : {config.warmup_steps}   max_steps: {config.max_steps}")
            logger.info(f"  scheduler       : {config.lr_scheduler}   min_lr_ratio: {config.min_lr_ratio}")
            logger.info(f"  precision       : {config.precision}    grad_clip: {config.grad_clip}")
            logger.info(f"  batch (per rank): {config.batch_size}    grad_accum: {config.gradient_accumulation_steps}")
            logger.info(f"  world_size      : {self.world_size}    effective_batch: {effective_batch}")
            logger.info(f"  FSDP            : {config.use_fsdp}    sharding: {config.sharding_mode}")
            logger.info(f"  save_steps      : {config.save_steps}   save_total_limit: {config.save_total_limit}")
            logger.info(f"  checkpoint_dir  : {config.checkpoint_dir}")
            model_cfg = getattr(self.model, "config", None) or getattr(getattr(self.model, "module", None), "config", None)
            if model_cfg is not None and getattr(model_cfg, "num_experts", 1) > 1:
                logger.info(f"  MoE top_k       : {getattr(model_cfg, 'top_k', 1)}   "
                            f"primary_weight: {getattr(model_cfg, 'top_k_primary_weight', 0.75)}")
            logger.info("=" * 70)

        # Optimizer
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer

        # Scheduler
        num_training_steps = config.max_steps
        if scheduler is None:
            self.scheduler = get_lr_scheduler(
                self.optimizer, config, num_training_steps, model=model,
            )
            resolved = resolve_scheduler_name(config, model)
            if config.lr_scheduler == "auto":
                logger.info(f"Auto scheduler: {resolved}")
        else:
            self.scheduler = scheduler

        # Loss function
        self.compute_loss = compute_loss or self._default_loss

        # Checkpointing
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            max_checkpoints=config.save_total_limit,
        )

        # Metrics
        self.metrics = MetricsTracker(config.log_dir)

        # Audit logging
        self.audit = AuditLogger(
            log_path=str(Path(config.log_dir) / "audit.log")
        )

        # Training state
        self.state = TrainingState()
        self.global_step = 0
        self.epoch = 0

        # Param-update assertion (catches silent zero-grad bugs from forward-only
        # custom kernels). At step 0 we snapshot a few canary params, then check
        # at step 1 that they actually changed. Cheap (one max() call per param).
        # Triggered by: any learnable param in the model that didn't move after
        # one optimizer step. Disable with config.skip_param_update_check = True.
        self._init_snapshot: Dict[str, torch.Tensor] = {}
        self._update_check_done = False

        # Mixed precision scaler (FP16 only, BF16 doesn't need it)
        self.scaler = None
        if config.precision == "fp16":
            self.scaler = torch.cuda.amp.GradScaler()

        if self.is_main:
            self.audit.log_training_start(config.to_dict())
            logger.info(f"Trainer initialized on {self.world_size} devices")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        config = self.config

        if config.optimizer_type == "muon":
            from .muon import MuonWithAdamW, split_params_for_muon
            muon_groups, adam_groups = split_params_for_muon(self.model)
            optimizer = MuonWithAdamW(
                muon_params=muon_groups,
                adam_params=adam_groups,
                lr=config.muon_lr,
                adam_lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            if self.is_main:
                muon_p = sum(p.numel() for g in muon_groups for p in g['params'])
                adam_p = sum(p.numel() for g in adam_groups for p in g['params'])
                logger.info(f"Muon optimizer: {muon_p/1e6:.0f}M orthogonalized, {adam_p/1e6:.0f}M AdamW")
            return optimizer

        if config.optimizer_type == "muon_tr":
            from .muon_tr import MuonTRWithAdamW, split_params_for_muon_tr
            num_experts = getattr(getattr(self.model, 'config', None), 'num_experts', 4)
            muon_groups, adam_groups = split_params_for_muon_tr(self.model, num_experts=num_experts)
            optimizer = MuonTRWithAdamW(
                muon_params=muon_groups,
                adam_params=adam_groups,
                lr=config.muon_lr,
                adam_lr=config.learning_rate,
                weight_decay=config.weight_decay,
                expert_lr_scale=config.expert_lr_scale,
                expert_weight_decay=config.expert_weight_decay,
                num_experts=num_experts,
            )
            if self.is_main:
                expert_p = sum(p.numel() for g in muon_groups for p in g['params'] if g.get('param_type') == 'expert')
                shared_p = sum(p.numel() for g in muon_groups for p in g['params'] if g.get('param_type') == 'shared')
                dense_p = sum(p.numel() for g in muon_groups for p in g['params'] if g.get('param_type') == 'dense')
                adam_p = sum(p.numel() for g in adam_groups for p in g['params'])
                logger.info(f"MuonTR optimizer: {expert_p/1e6:.0f}M expert, {shared_p/1e6:.0f}M shared, "
                            f"{dense_p/1e6:.0f}M dense (NS), {adam_p/1e6:.0f}M AdamW")
                logger.info(f"  expert_lr_scale={config.expert_lr_scale}, "
                            f"expert_wd={config.expert_weight_decay}, adaptive_ns={config.adaptive_ns}")
            return optimizer

        if config.optimizer_type == "adam_tr":
            from .adam_tr import AdamTR, adamtr_param_groups
            num_experts = getattr(getattr(self.model, 'config', None), 'num_experts', 4)
            param_groups = adamtr_param_groups(
                self.model,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                expert_lr_scale=config.expert_lr_scale,
                expert_weight_decay=config.expert_weight_decay,
            )
            optimizer = AdamTR(
                param_groups,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                expert_lr_scale=config.expert_lr_scale,
                expert_weight_decay=config.expert_weight_decay,
                num_experts=num_experts,
                spectral_conditioning=True,
            )
            if self.is_main:
                total_p = sum(p.numel() for g in param_groups for p in g['params'])
                logger.info(f"AdamTR optimizer: {total_p/1e6:.0f}M params, "
                            f"expert_lr_scale={config.expert_lr_scale}, spectral_conditioning=True")
            return optimizer

        # Split params: decay vs no-decay, and optionally muP scaling
        embed_params = []
        hidden_params = []
        no_decay_params = []

        base_width = getattr(config, 'mup_base_width', 256)
        model_width = getattr(self.model, 'config', None)
        model_width = getattr(model_width, 'hidden_size', base_width) if model_width else base_width
        width_ratio = model_width / base_width

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or '.mu' in name:
                no_decay_params.append(param)
            elif 'embed' in name or 'lm_head' in name:
                embed_params.append(param)
            else:
                hidden_params.append(param)

        if config.optimizer_type == "adamw_mup":
            # muP: scale LR by 1/width_ratio for hidden layers,
            # keep base LR for embeddings. Ref: Yang et al. 2022
            param_groups = [
                {"params": embed_params, "lr": config.learning_rate, "weight_decay": config.weight_decay},
                {"params": hidden_params, "lr": config.learning_rate / width_ratio, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "lr": config.learning_rate / width_ratio, "weight_decay": 0.0},
            ]
            if self.is_main:
                logger.info(f"muP AdamW: base_width={base_width}, model_width={model_width}, "
                            f"embed_lr={config.learning_rate:.2e}, hidden_lr={config.learning_rate/width_ratio:.2e}")
        else:
            # Standard AdamW
            param_groups = [
                {"params": embed_params + hidden_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]

        return torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            foreach=False,
        )

    def _default_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Default loss computation (cross-entropy for LM)."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids[:, 1:]).to(self.device)

        outputs = model(input_ids)

        if outputs.dim() == 3:
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = labels[:, :shift_logits.size(1)].contiguous()
        else:
            shift_logits = outputs
            shift_labels = labels

        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        self.model.train()

        if self.config.resume_from:
            self.state = self.checkpoint_manager.load(self.config.resume_from)
            self.global_step = self.state.step
            self.epoch = self.state.epoch
            logger.info(f"Resumed from step {self.global_step}")

            # PyTorch footgun: scheduler.load_state_dict() restores base_lrs from
            # the checkpoint, silently overriding the --lr passed on the CLI.
            # If the caller changed learning_rate for the resume (e.g. lowering
            # after divergence), we must re-apply it to the optimizer and to the
            # scheduler's base_lrs so the cosine/WSD decay recomputes correctly.
            new_lr = self.config.learning_rate
            current_base_lr = self._scheduler_base_lr()
            if current_base_lr is not None and abs(new_lr - current_base_lr) > 1e-12:
                if self.is_main:
                    logger.warning(
                        f"Overriding loaded scheduler base_lr {current_base_lr:.2e} → "
                        f"{new_lr:.2e} (from --lr). This is the intended behavior when "
                        f"resuming with a different LR."
                    )
                for group in self.optimizer.param_groups:
                    group["initial_lr"] = new_lr
                    group["lr"] = new_lr
                self._scheduler_set_base_lr(new_lr)

        accumulation_steps = self.config.gradient_accumulation_steps

        try:
            while self.global_step < self.config.max_steps:
                self.epoch += 1

                for batch_idx, batch in enumerate(self.train_dataloader):
                    step_start = time.time()

                    loss = self._training_step(batch)

                    if (batch_idx + 1) % accumulation_steps == 0:
                        self._update_expert_token_counts(batch)
                        self._optimizer_step()
                        self.global_step += 1

                        step_time = time.time() - step_start
                        self.metrics.log_step_time(step_time)

                        if self.eval_dataloader and self.global_step % self.config.eval_steps == 0:
                            eval_loss = self.evaluate()
                            if self.is_main:
                                logger.info(f"Step {self.global_step} - Eval Loss: {eval_loss:.4f}")

                        if self.global_step % self.config.save_steps == 0:
                            self._save_checkpoint()

                        for callback in self.callbacks:
                            callback(self, self.global_step, loss.item())

                        if self.global_step >= self.config.max_steps:
                            break

                if self.config.max_epochs and self.epoch >= self.config.max_epochs:
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint(tag="interrupted")

        self._save_checkpoint(tag="final")

        summary = self.metrics.get_summary()
        if self.is_main:
            self.audit.log_training_end(summary)
            self.metrics.save()

        return summary

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward + backward pass under autocast."""
        if self.config.precision in ["fp16", "bf16"]:
            dtype = torch.float16 if self.config.precision == "fp16" else torch.bfloat16
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss = self.compute_loss(self.model, batch)
                scaled_loss = loss / self.config.gradient_accumulation_steps
                if self.scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
        else:
            loss = self.compute_loss(self.model, batch)
            scaled_loss = loss / self.config.gradient_accumulation_steps
            scaled_loss.backward()

        return loss

    def _update_expert_token_counts(self, batch):
        """Update per-expert token counts for MuonTR/AdamTR optimizers."""
        if not hasattr(self.optimizer, 'update_token_counts'):
            return
        input_ids = batch.get("input_ids")
        if input_ids is None:
            return
        num_experts = getattr(getattr(self.model, 'config', None), 'num_experts', 1)
        if num_experts <= 1:
            return
        # Find the token_to_expert mapping from the model
        token_to_expert = None
        for module in self.model.modules():
            if hasattr(module, 'token_to_expert'):
                token_to_expert = module.token_to_expert
                break
        if token_to_expert is None:
            return
        flat_ids = input_ids.view(-1).clamp(0, len(token_to_expert) - 1)
        expert_ids = token_to_expert[flat_ids]
        counts = torch.bincount(expert_ids, minlength=num_experts)
        self.optimizer.update_token_counts(counts)

    def _snapshot_canary_params(self):
        """Snapshot a small set of canary parameters before the first step.

        We pick at most one param from each class (embed, expert, shared,
        attention, mu_guidance) so the assertion check is O(few) tensors,
        not O(num_params). Norms are intentionally skipped — they init to
        1.0 and the AdamW weight-decay update (1 - lr*wd)*1.0 rounds back
        to 1.0 in bf16, producing false-positive 'no update' warnings.

        This is the diagnostic that catches silent zero-grad bugs from
        forward-only custom kernels (per Whatsonyourmind on Muon#65).
        """
        if self._init_snapshot:
            return  # already snapshotted

        seen_classes = set()

        def classify(name: str) -> str:
            if "embed" in name or "lm_head" in name:
                return "embed"
            if "gate_proj_w" in name or "up_proj_w" in name or "down_proj_w" in name:
                return "expert"
            if "shared_" in name:
                return "shared"
            if "self_attn" in name and "norm" not in name:
                return "attention"
            if "mu_guidance" in name or "mu_init" in name or "mu_to_" in name:
                return "mu"
            if "norm" in name or "ln_" in name:
                return "norm_skipped"  # explicitly skipped
            return "dense"

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            cls = classify(name)
            # Skip norms — bf16 rounding makes the update invisible
            if cls == "norm_skipped":
                continue
            if cls in seen_classes:
                continue
            seen_classes.add(cls)
            self._init_snapshot[name] = p.detach().clone()

    def _check_params_updated(self):
        """Verify that all snapshotted canary params actually moved after step.

        Logs both the param delta (W_t - W_init) AND the grad norm at the
        time of the check. A param can have a non-zero delta from weight
        decay alone even if grad is zero — so we report both to disambiguate
        real gradient flow from weight-decay drift.

        Per Whatsonyourmind on KellerJordan/Muon#65.
        """
        if not self._init_snapshot or self._update_check_done:
            return

        results = []  # list of (name, delta, grad_norm)
        params_dict = dict(self.model.named_parameters())
        for name, p_init in self._init_snapshot.items():
            p_now = params_dict.get(name)
            if p_now is None:
                continue
            try:
                p_now_local = p_now.to_local() if hasattr(p_now, "to_local") else p_now
                p_init_local = p_init.to_local() if hasattr(p_init, "to_local") else p_init
                delta = (p_now_local.detach() - p_init_local).abs().max().item()
                # grad may have been zeroed by optimizer.zero_grad() already.
                # If so, we'll just report 0.0 for grad_norm — the delta is
                # what matters for confirming the update happened.
                grad = p_now.grad
                if grad is not None:
                    g_local = grad.to_local() if hasattr(grad, "to_local") else grad
                    grad_norm = g_local.detach().float().norm().item()
                else:
                    grad_norm = 0.0
            except Exception:
                continue
            results.append((name, delta, grad_norm))

        self._update_check_done = True

        if not self.is_main:
            return

        # Print full diagnostic table
        print("\n[param-update check] post-step-1 diagnostic:", flush=True)
        for name, delta, grad_norm in results:
            short = name.replace("_orig_mod.", "").replace("module.", "")
            tag = "OK " if delta > 1e-7 else "DEAD"
            print(f"  [{tag}] {short:60s}  delta={delta:.3e}  grad_norm={grad_norm:.3e}", flush=True)
        dead_count = sum(1 for _, d, _ in results if d <= 1e-7)
        if dead_count:
            print(f"  ⚠ {dead_count}/{len(results)} canary params did not update — likely silent zero-grad bug", flush=True)
        else:
            print(f"  ✓ all {len(results)} canary params updated", flush=True)

    def _optimizer_step(self):
        """Optimizer step with gradient clipping."""
        # Snapshot canary params on the first call (before any update)
        skip_check = getattr(self.config, "skip_param_update_check", False)
        if not skip_check and not self._init_snapshot:
            self._snapshot_canary_params()

        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()

        # Check params BEFORE zero_grad so we can read p.grad alongside the
        # post-update delta — both signals together let us distinguish a real
        # gradient flow from a weight-decay-only drift.
        if not skip_check and not self._update_check_done:
            self._check_params_updated()

        self.optimizer.zero_grad()

    def _iter_schedulers(self):
        """Yield the scheduler and any nested sub-schedulers (SequentialLR/ChainedLR)."""
        yield self.scheduler
        for sub in getattr(self.scheduler, "_schedulers", []) or []:
            yield sub

    def _scheduler_base_lr(self):
        """Read the first base_lr across the scheduler tree (all sub-LRs should match)."""
        for sch in self._iter_schedulers():
            base = getattr(sch, "base_lrs", None)
            if base:
                return float(base[0])
        return None

    def _scheduler_set_base_lr(self, new_lr: float) -> None:
        """Rewrite every base_lr in the scheduler tree (SequentialLR/ChainedLR safe)."""
        for sch in self._iter_schedulers():
            base = getattr(sch, "base_lrs", None)
            if base is not None:
                sch.base_lrs = [new_lr for _ in base]

    def _save_checkpoint(self, tag: str = "step"):
        """Save checkpoint."""
        self.state.step = self.global_step
        self.state.epoch = self.epoch
        self.state.learning_rate = self.scheduler.get_last_lr()[0]

        self.checkpoint_manager.save(
            step=self.global_step,
            training_state=self.state,
            tag=tag,
        )

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation."""
        if self.eval_dataloader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataloader:
            loss = self._training_step(batch)
            total_loss += loss.item()
            num_batches += 1

        self.model.train()

        avg_loss = total_loss / max(num_batches, 1)

        if self.distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor)
            avg_loss = loss_tensor.item() / self.world_size

        return avg_loss
