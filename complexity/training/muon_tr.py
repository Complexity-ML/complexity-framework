"""
MuonTR — Muon for Token-Routed MoE architectures.

Extends Muon (Momentum Orthogonalized by Newton-Schulz) with the same
four MoE-aware features as AdamTR:

1. **Per-expert LR**: Experts seeing fewer tokens get higher LR.
2. **Expert-aware weight decay**: Lighter decay for routed experts.
3. **Gradient scaling per expert**: Normalizes by token count per expert.
4. **Separate momentum**: Natural with [E, H, I] parameter tensors.

Uses MuonTR for 2D+ expert/shared/dense weights, AdamW for the rest
(embeddings, biases, norms, mu params).

Usage:
    muon_params, adam_params = split_params_for_muon_tr(model, num_experts=4)
    optimizer = MuonTRWithAdamW(muon_params, adam_params, num_experts=4)

    # Each step, update token counts from the batch
    optimizer.update_token_counts(token_counts)

INL / Complexity-ML — 2026
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.optim import Optimizer

from .adam_tr import _classify_param
from .muon import newton_schulz, newton_schulz_adaptive

logger = logging.getLogger("complexity.muon_tr")


def _to_local(t: torch.Tensor) -> torch.Tensor:
    """Convert DTensor to local tensor (FSDP v2 compat)."""
    if hasattr(t, "to_local"):
        return t.to_local()
    return t


class MuonTR(Optimizer):
    """
    MuonTR: Muon optimizer specialized for Token-Routed MoE.

    Same Newton-Schulz orthogonalization as Muon, plus per-expert
    LR scaling, expert-aware weight decay, and gradient normalization.

    Key difference from standard Muon: **per-expert orthogonalization**.
    Each expert's [H, I] slice is orthogonalized independently because
    experts see different token distributions under Zipf routing, giving
    each expert a different singular value spectrum. Joint orthogonalization
    across all experts would force a single scaling from the largest SV,
    shrinking tail expert gradients excessively.

    Tail experts (fewer tokens) get adaptive NS iterations because their
    noisier gradients have wider SV spread → slower convergence.

    Ref: KellerJordan/Muon#65

    Args:
        params: Parameter groups (with param_type metadata).
        lr: Base learning rate (default: 0.02).
        momentum: Momentum coefficient (default: 0.95).
        nesterov: Use Nesterov momentum (default: True).
        ns_steps: Min Newton-Schulz iterations (default: 5).
        ns_max_steps: Max Newton-Schulz iterations for adaptive mode (default: 10).
        ns_tol: Convergence tolerance for adaptive NS (default: 1e-2).
        adaptive_ns: Use adaptive NS iterations per expert (default: True).
        weight_decay: Base weight decay (default: 0.01).
        expert_lr_scale: LR multiplier for routed expert params (default: 1.5).
        shared_lr_scale: LR multiplier for shared expert params (default: 1.0).
        expert_weight_decay: Weight decay for expert params (default: 0.005).
        shared_weight_decay: Weight decay for shared params (default: 0.01).
        token_counts: Optional [E] tensor of tokens per expert.
        num_experts: Number of experts.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,                 # 0.02 diverges on ~100M bf16; 0.01 is safe
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_max_steps: int = 10,
        ns_tol: float = 1e-4,
        adaptive_ns: bool = False,        # fixed NS is ~20% faster and as accurate in practice
        weight_decay: float = 0.01,
        expert_lr_scale: float = 1.5,
        shared_lr_scale: float = 1.0,
        expert_weight_decay: float = 0.005,
        shared_weight_decay: float = 0.01,
        token_counts: Optional[torch.Tensor] = None,
        num_experts: int = 4,
        ema_decay: float = 0.998,         # ~500 step half-life for the LR ratio EMA
        max_lr_ratio: float = 2.0,        # cap tail-expert LR boost (was 4.0 → diverged at lr=0.02)
        lr_warmup_steps: int = 50,        # linear 0→lr ramp — required, orthogonalized updates
                                          # are full-norm and blow up without warmup
        skip_ns_warmup_steps: int = 0,    # N steps without ortho (plain momentum update)
        max_update_rms: Optional[float] = 1.0,  # trust-region clamp: max RMS of the update
                                                # tensor per param. None disables.
        sanitize_nan: bool = True,        # replace NaN/Inf in update by 0 before applying
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_max_steps=ns_max_steps,
            ns_tol=ns_tol,
            adaptive_ns=adaptive_ns,
            weight_decay=weight_decay,
            expert_lr_scale=expert_lr_scale,
            shared_lr_scale=shared_lr_scale,
            expert_weight_decay=expert_weight_decay,
            shared_weight_decay=shared_weight_decay,
        )
        super().__init__(params, defaults)
        self.num_experts = num_experts
        self.token_counts = token_counts
        self.ema_decay = ema_decay
        self.max_lr_ratio = max_lr_ratio
        self.lr_warmup_steps = max(0, int(lr_warmup_steps))
        self.skip_ns_warmup_steps = max(0, int(skip_ns_warmup_steps))
        self.max_update_rms = max_update_rms
        self.sanitize_nan = sanitize_nan
        # Per-expert diagnostics kept as device tensors to avoid MPS/CUDA syncs;
        # converted to Python floats only when get_ns_diagnostics() is called.
        self._ns_residual: float = 0.0    # scalar, last batched NS residual
        self._ns_steps_used: int = 0       # scalar, last batched NS iter count
        self._grad_norms: Optional[torch.Tensor] = None  # [E]
        self._lr_ratio_ema: Optional[torch.Tensor] = None  # [E]
        self._step_count = 0

        # Auto-disable MoE-specific features if there's nothing to route
        self._moe_enabled = num_experts > 1

    def update_token_counts(self, token_counts: torch.Tensor):
        """Update per-expert token counts for gradient scaling."""
        self.token_counts = token_counts

    def get_ns_diagnostics(self) -> Dict[int, Dict[str, float]]:
        """Per-expert NS convergence diagnostics + grad-norm + LR ratio.

        Only this call syncs the GPU tensors to Python floats.
        """
        gnorm = (self._grad_norms.detach().cpu().tolist()
                 if self._grad_norms is not None else [0.0] * self.num_experts)
        lr_ratio = (self._lr_ratio_ema.detach().cpu().tolist()
                    if self._lr_ratio_ema is not None else [1.0] * self.num_experts)
        return {
            e: {
                "residual":  self._ns_residual,
                "steps":     self._ns_steps_used,
                "grad_norm": float(gnorm[e]),
                "lr_ratio":  float(lr_ratio[e]),
            }
            for e in range(self.num_experts)
        }

    def _compute_lr_ratio(self, device: torch.device) -> torch.Tensor:
        """
        Per-expert adaptive LR controller (Whatsonyourmind suggestion).

        target_lr_ratio[i] = max(1.0, mean_count / count[i])
        EMA-smoothed over ~500 steps to avoid oscillation.
        Clamped to [1.0, max_lr_ratio].
        """
        if self.token_counts is None:
            return torch.ones(self.num_experts, device=device)

        counts = self.token_counts.float().to(device)
        mean_count = counts.mean()
        # Tail experts (count < mean) get ratio > 1, head experts get 1.0
        target = (mean_count / counts.clamp(min=1.0)).clamp(min=1.0, max=self.max_lr_ratio)

        if self._lr_ratio_ema is None or self._lr_ratio_ema.device != device:
            self._lr_ratio_ema = target.clone()
        else:
            self._lr_ratio_ema.mul_(self.ema_decay).add_(target, alpha=1.0 - self.ema_decay)

        return self._lr_ratio_ema

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        # Linear LR warmup from 0 → peak over lr_warmup_steps.
        # Applied as a global multiplier so the group's scheduler (cosine/WSD)
        # continues to work as-is on top.
        if self.lr_warmup_steps > 0 and self._step_count <= self.lr_warmup_steps:
            warmup_mult = self._step_count / self.lr_warmup_steps
        else:
            warmup_mult = 1.0

        # Skip Newton-Schulz during the very first steps to let momentum build
        # on a non-orthogonalized signal — avoids blowing up the residual
        # stream with full-norm updates before the model has settled.
        skip_ns = self._step_count <= self.skip_ns_warmup_steps

        for group in self.param_groups:
            lr = group["lr"] * warmup_mult
            beta = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            ns_max_steps = group.get("ns_max_steps", 10)
            ns_tol = group.get("ns_tol", 1e-2)
            adaptive_ns = group.get("adaptive_ns", False)
            param_type = group.get("param_type", "dense")

            # Resolve effective LR / WD once per group
            if param_type == "expert":
                effective_lr = lr * group["expert_lr_scale"]
                wd = group["expert_weight_decay"]
            elif param_type == "shared":
                effective_lr = lr * group["shared_lr_scale"]
                wd = group["shared_weight_decay"]
            else:
                effective_lr = lr
                wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = _to_local(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                p_local = _to_local(p)

                is_expert_3d = (
                    param_type == "expert"
                    and grad.dim() == 3
                    and grad.shape[0] == self.num_experts
                    and self._moe_enabled
                )

                # --- Feature 3: Token-count grad scaling (vectorized) ---
                if is_expert_3d and self.token_counts is not None:
                    tc = self.token_counts.to(grad.device, dtype=torch.float32).clamp(min=1.0)
                    mean_count = tc.mean().clamp(min=1.0)
                    scale = (mean_count / tc).clamp(0.5, 2.0).to(grad.dtype)
                    grad = grad * scale.view(-1, 1, 1)

                # Momentum update + Nesterov lookahead (dtype-preserving)
                buf.lerp_(grad, 1 - beta)
                update = grad.lerp(buf, beta) if nesterov else buf.clone()

                # --- Expert path: batched Newton-Schulz over [E, H, I] ---
                if is_expert_3d:
                    # Track per-expert grad norms (tensor, synced only in diagnostics)
                    self._grad_norms = grad.float().flatten(1).norm(dim=-1)

                    rows, cols = update.shape[1], update.shape[2]

                    if not skip_ns:
                        transposed = rows > cols
                        if transposed:
                            update = update.transpose(1, 2).contiguous()

                        # newton_schulz / newton_schulz_adaptive are batch-compatible:
                        # X @ X.mT and X.norm(dim=(-2,-1)) both operate per-slice on dim 0.
                        if adaptive_ns:
                            update, residual, steps_used = newton_schulz_adaptive(
                                update, min_steps=ns_steps, max_steps=ns_max_steps, tol=ns_tol,
                            )
                            self._ns_residual = float(residual)
                            self._ns_steps_used = int(steps_used)
                        else:
                            update = newton_schulz(update, steps=ns_steps)
                            self._ns_steps_used = int(ns_steps)

                        if transposed:
                            update = update.transpose(1, 2).contiguous()

                        # Shape scaling (same for all experts since H, I uniform)
                        update *= max(1.0, rows / cols) ** 0.5
                    # else: keep raw momentum update (no ortho) for the first N steps

                    # Per-expert adaptive LR ratio (EMA on token counts)
                    per_expert_lr = self._compute_lr_ratio(update.device)
                    update *= per_expert_lr.to(update.dtype).view(-1, 1, 1)
                elif not skip_ns:
                    # Standard Muon: reshape to 2D, orthogonalize
                    original_shape = update.shape
                    if update.ndim > 2:
                        update = update.view(update.shape[0], -1)

                    rows, cols = update.shape
                    transposed = rows > cols
                    if transposed:
                        update = update.T

                    update = newton_schulz(update, steps=ns_steps)

                    if transposed:
                        update = update.T

                    update *= max(1.0, rows / cols) ** 0.5
                    update = update.reshape(original_shape)
                # else: skip_ns path for non-expert — use raw momentum update as-is

                # --- Safety: NaN/Inf sanitization ---
                # Orthogonalization in bf16 with tiny grads can produce NaN/Inf.
                # In-place replace; no CPU sync, no branch on tensor state.
                if self.sanitize_nan:
                    torch.nan_to_num_(update, nan=0.0, posinf=0.0, neginf=0.0)

                # --- Safety: trust-region clamp on update RMS ---
                # RMS = sqrt(mean(update²)) — scalar per param tensor. If it
                # exceeds max_update_rms, rescale so RMS == max_update_rms.
                # Protects against the slow-drift divergence seen at lr=0.02
                # where tail-expert updates grow unbounded.
                if self.max_update_rms is not None:
                    rms = update.pow(2).mean().sqrt().clamp(min=1e-12)
                    scale = (self.max_update_rms / rms).clamp(max=1.0)
                    update.mul_(scale)

                # Decoupled weight decay
                if wd != 0:
                    p_local.mul_(1 - effective_lr * wd)

                # Parameter update
                p_local.add_(update, alpha=-effective_lr)

        return loss


class MuonTRWithAdamW(Optimizer):
    """
    Combined optimizer: MuonTR for 2D+ weights, AdamW for the rest.

    Args:
        muon_params: Parameter groups for MuonTR (with param_type metadata).
        adam_params: Parameter groups for AdamW.
        lr: MuonTR learning rate (default: 0.02).
        adam_lr: AdamW learning rate (default: 3e-4).
        momentum: MuonTR momentum (default: 0.95).
        nesterov: MuonTR Nesterov (default: True).
        ns_steps: Newton-Schulz iterations (default: 5).
        adam_betas: AdamW betas (default: (0.9, 0.95)).
        adam_eps: AdamW epsilon (default: 1e-8).
        weight_decay: Base weight decay (default: 0.01).
        expert_lr_scale: LR multiplier for experts (default: 1.5).
        expert_weight_decay: Weight decay for experts (default: 0.005).
        num_experts: Number of experts (default: 4).
    """

    def __init__(
        self,
        muon_params,
        adam_params,
        lr: float = 0.01,
        adam_lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adam_betas: Tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        weight_decay: float = 0.01,
        expert_lr_scale: float = 1.5,
        shared_lr_scale: float = 1.0,
        expert_weight_decay: float = 0.005,
        shared_weight_decay: float = 0.01,
        num_experts: int = 4,
        ema_decay: float = 0.998,
        max_lr_ratio: float = 2.0,
        lr_warmup_steps: int = 50,
        skip_ns_warmup_steps: int = 0,
        adaptive_ns: bool = False,
        max_update_rms: Optional[float] = 1.0,
        sanitize_nan: bool = True,
    ):
        self.muon_tr = MuonTR(
            muon_params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adaptive_ns=adaptive_ns,
            weight_decay=weight_decay,
            expert_lr_scale=expert_lr_scale,
            shared_lr_scale=shared_lr_scale,
            expert_weight_decay=expert_weight_decay,
            shared_weight_decay=shared_weight_decay,
            num_experts=num_experts,
            ema_decay=ema_decay,
            max_lr_ratio=max_lr_ratio,
            lr_warmup_steps=lr_warmup_steps,
            skip_ns_warmup_steps=skip_ns_warmup_steps,
            max_update_rms=max_update_rms,
            sanitize_nan=sanitize_nan,
        )
        self.adam = torch.optim.AdamW(
            adam_params,
            lr=adam_lr,
            betas=adam_betas,
            eps=adam_eps,
            weight_decay=weight_decay,
        )

        super().__init__(
            self.muon_tr.param_groups + self.adam.param_groups,
            defaults={},
        )
        self.param_groups = self.muon_tr.param_groups + self.adam.param_groups

    def update_token_counts(self, token_counts: torch.Tensor):
        """Update per-expert token counts for gradient scaling."""
        self.muon_tr.update_token_counts(token_counts)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = self.muon_tr.step(closure)
        self.adam.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.muon_tr.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            'muon_tr': self.muon_tr.state_dict(),
            'adam': self.adam.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.muon_tr.load_state_dict(state_dict['muon_tr'])
        self.adam.load_state_dict(state_dict['adam'])


def split_params_for_muon_tr(
    model: torch.nn.Module,
    num_experts: int = 4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split model parameters into MuonTR and AdamW groups with param_type.

    MuonTR handles: 2D+ weight matrices (expert, shared, dense)
    AdamW handles: embeddings, biases, norms, mu params

    Returns:
        (muon_params, adam_params): Parameter groups with metadata.
    """
    expert_params: List[torch.Tensor] = []
    shared_params: List[torch.Tensor] = []
    dense_params: List[torch.Tensor] = []
    adam_decay: List[torch.Tensor] = []
    adam_no_decay: List[torch.Tensor] = []

    # Params that stay on AdamW: 1D (norms/bias/α/mu scalars), embeddings, LM head
    adam_keywords = ("embed", "lm_head", "head", "bias", "norm", "ln_",
                     ".mu", "mu_", "dynamics", "alpha")
    no_decay_keywords = ("bias", "norm", ".mu", "alpha")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lowered = name.lower()
        is_adam = param.ndim < 2 or any(k in lowered for k in adam_keywords)

        if is_adam:
            if param.ndim < 2 or any(k in lowered for k in no_decay_keywords):
                adam_no_decay.append(param)
            else:
                adam_decay.append(param)
            continue

        # Reuse the shared classifier for the Muon groups
        ptype = _classify_param(name, param, num_experts)
        if ptype == "expert":
            expert_params.append(param)
        elif ptype == "shared":
            shared_params.append(param)
        else:
            dense_params.append(param)

    muon_groups: List[Dict[str, Any]] = []
    if expert_params:
        muon_groups.append({"params": expert_params, "param_type": "expert"})
    if shared_params:
        muon_groups.append({"params": shared_params, "param_type": "shared"})
    if dense_params:
        muon_groups.append({"params": dense_params, "param_type": "dense"})

    adam_groups: List[Dict[str, Any]] = []
    if adam_decay:
        adam_groups.append({"params": adam_decay})
    if adam_no_decay:
        adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})

    return muon_groups, adam_groups
