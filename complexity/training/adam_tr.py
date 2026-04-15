"""
AdamTR — Adam for Token-Routed MoE architectures.

MoE-aware optimizer that treats expert parameters differently from shared/dense
parameters. Based on AdamW with five innovations:

1. **Per-expert LR**: each expert group gets its own LR scale.
2. **Expert-aware weight decay**: different decay rates for routed / shared / dense.
3. **Gradient scaling per expert**: normalizes by tokens-per-expert (Zipf-aware).
4. **Separate momentum**: exp_avg / exp_avg_sq tracked per expert slice so
   cross-expert momentum doesn't dilute specialization.
5. **Per-expert spectral conditioning**: estimates κ = σ_max / σ_rms of each
   expert's gradient matrix and scales the LR by 1/κ (floored). This is the
   Adam-native equivalent of Muon's Newton-Schulz orthogonalization.

The core update is vectorized across the [E, H, I] expert dim — power iteration,
σ_rms, κ, EMA and the final addcdiv are done with tensor ops, avoiding the
per-expert Python loop + `.item()` syncs that kill MPS throughput.

Reference: KellerJordan/Muon#65, Complexity-ML (2026)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

logger = logging.getLogger("complexity.adam_tr")


# Param type tags used throughout
_EXPERT = "expert"
_SHARED = "shared"
_DENSE  = "dense"


def _to_local(t: Tensor) -> Tensor:
    """Convert DTensor to local tensor (FSDP v2 compat)."""
    if hasattr(t, "to_local"):
        return t.to_local()
    return t


def _classify_param(name: str, param: Tensor, num_experts: int) -> str:
    """Classify a parameter into expert / shared / dense.

    Resilient to naming: checks name first, falls back to shape signature
    ([E, *, *] tensor matching num_experts counts as expert).
    """
    lowered = name.lower()
    if any(k in lowered for k in ("gate_proj_w", "up_proj_w", "down_proj_w")):
        return _EXPERT
    if any(k in lowered for k in ("shared_gate", "shared_up", "shared_down")):
        return _SHARED
    if (
        num_experts > 1
        and param.dim() == 3
        and param.shape[0] == num_experts
    ):
        return _EXPERT
    return _DENSE


def _power_iteration_batched(
    G: Tensor,
    n_iters: int = 3,
    eps: float = 1e-7,
) -> Tensor:
    """Estimate σ_max(G[e]) for each e, in one vectorized pass.

    Args:
        G: [E, H, I] gradient tensor (float32 recommended).
        n_iters: number of power iteration steps.
        eps: floor on norms to avoid div-by-zero.

    Returns:
        sigma_max: [E] tensor of largest singular values (same dtype as G).
    """
    E = G.shape[0]
    I = G.shape[2]
    # Random init, stable in fp32
    v = torch.randn(E, I, device=G.device, dtype=G.dtype)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=eps)

    Gt = G.transpose(1, 2)  # [E, I, H]
    for _ in range(n_iters):
        u = torch.bmm(G, v.unsqueeze(-1)).squeeze(-1)        # [E, H]
        u = u / u.norm(dim=-1, keepdim=True).clamp(min=eps)
        v = torch.bmm(Gt, u.unsqueeze(-1)).squeeze(-1)       # [E, I]
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=eps)

    # σ_max ≈ ||G v||
    return torch.bmm(G, v.unsqueeze(-1)).squeeze(-1).norm(dim=-1)  # [E]


class AdamTR(Optimizer):
    """AdamTR: Adam optimizer specialized for Token-Routed MoE.

    Constructor is API-compatible with the previous implementation.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        expert_lr_scale: float = 1.5,
        shared_lr_scale: float = 1.0,
        expert_weight_decay: float = 0.05,
        shared_weight_decay: float = 0.1,
        token_counts: Optional[Tensor] = None,
        num_experts: int = 4,
        spectral_conditioning: bool = True,
        spectral_ema: float = 0.99,
        spectral_floor: float = 0.7,
        spectral_warmup_steps: int = 50,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            expert_lr_scale=expert_lr_scale,
            shared_lr_scale=shared_lr_scale,
            expert_weight_decay=expert_weight_decay,
            shared_weight_decay=shared_weight_decay,
        )
        super().__init__(params, defaults)

        self.num_experts = num_experts
        self.token_counts = token_counts
        # Auto-disable MoE features when there's nothing to route
        self._moe_enabled = num_experts > 1
        self.spectral_conditioning = spectral_conditioning and self._moe_enabled
        self.spectral_ema = spectral_ema
        self.spectral_floor = spectral_floor
        self.spectral_warmup_steps = spectral_warmup_steps

        # Kappa EMA as a device tensor — updated without forcing CPU syncs
        self._kappa_ema: Optional[Tensor] = None
        self._grad_norms: Optional[Tensor] = None
        self._step_count = 0

        # Ensure every group has a resolved `param_type` (idempotent if already set)
        self._resolve_param_types()

    # --- setup helpers ----------------------------------------------------

    def _resolve_param_types(self) -> None:
        """Tag each param group with its type once, so step() doesn't re-parse names."""
        for group in self.param_groups:
            if "param_type" in group:
                continue
            name = group.get("param_name", "")
            params = group["params"]
            # All params in a group share a type. Use the first to classify.
            if not params:
                group["param_type"] = _DENSE
                continue
            group["param_type"] = _classify_param(name, params[0], self.num_experts)

    def _ensure_diag_buffers(self, device: torch.device) -> None:
        if self._kappa_ema is None:
            self._kappa_ema = torch.ones(self.num_experts, device=device)
            self._grad_norms = torch.zeros(self.num_experts, device=device)

    # --- public diagnostics (sync only on demand) -------------------------

    def update_token_counts(self, token_counts: Tensor) -> None:
        """Update per-expert token counts for gradient scaling."""
        self.token_counts = token_counts

    def get_spectral_diagnostics(self) -> Dict[int, float]:
        if self._kappa_ema is None:
            return {e: 1.0 for e in range(self.num_experts)}
        vals = self._kappa_ema.detach().cpu().tolist()
        return {e: float(v) for e, v in enumerate(vals)}

    def get_ns_diagnostics(self) -> Dict[int, Dict[str, float]]:
        """Per-expert diagnostics in MuonTR-compatible format."""
        if self._kappa_ema is None:
            return {
                e: {"grad_norm": 0.0, "lr_ratio": 1.0, "kappa": 1.0,
                    "residual": 0.0, "steps": 0}
                for e in range(self.num_experts)
            }
        kappa = self._kappa_ema.detach().cpu().tolist()
        gnorm = self._grad_norms.detach().cpu().tolist()
        return {
            e: {
                "grad_norm": float(gnorm[e]),
                "lr_ratio":  1.0 / max(float(kappa[e]), self.spectral_floor),
                "kappa":     float(kappa[e]),
                "residual":  0.0,
                "steps":     0,
            }
            for e in range(self.num_experts)
        }

    # --- main step --------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        do_spectral = (
            self.spectral_conditioning
            and self._step_count > self.spectral_warmup_steps
        )

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            base_lr = group["lr"]
            eps = group["eps"]
            ptype = group["param_type"]

            # Resolve effective LR and weight decay once per group
            if ptype == _EXPERT:
                effective_lr = base_lr * group["expert_lr_scale"]
                wd = group["expert_weight_decay"]
            elif ptype == _SHARED:
                effective_lr = base_lr * group["shared_lr_scale"]
                wd = group["shared_weight_decay"]
            else:
                effective_lr = base_lr
                wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = _to_local(p.grad)
                if grad.is_sparse:
                    raise RuntimeError("AdamTR does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                p_local = _to_local(p)

                # Decoupled weight decay (skip if wd==0 for a tiny win)
                if wd != 0.0:
                    p_local.mul_(1 - effective_lr * wd)

                # Expert-specific path: 3D [E, H, I] params with MoE features
                is_expert_3d = (
                    ptype == _EXPERT
                    and grad.dim() == 3
                    and grad.shape[0] == self.num_experts
                    and self._moe_enabled
                )

                if is_expert_3d:
                    self._ensure_diag_buffers(grad.device)

                    # --- Feature 3: Token-count gradient scaling (vectorized) ---
                    if self.token_counts is not None:
                        tc = self.token_counts.to(grad.device, dtype=torch.float32)
                        mean_count = tc.mean().clamp(min=1.0)
                        scale = (mean_count / tc.clamp(min=1.0)).clamp(0.5, 2.0)
                        grad = grad * scale.to(grad.dtype).view(-1, 1, 1)

                    # --- Update momentum ---
                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    bc2_sqrt = math.sqrt(bias_correction2)
                    denom = (exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)

                    # --- Feature 5: Spectral conditioning (batched, fp32) ---
                    if do_spectral:
                        G32 = grad.float()
                        H, I = G32.shape[1], G32.shape[2]
                        # σ_rms per expert (Frobenius / sqrt(min(H,I)))
                        frob = G32.flatten(1).norm(dim=-1)                    # [E]
                        sigma_rms = frob / math.sqrt(min(H, I))               # [E]
                        sigma_max = _power_iteration_batched(G32, n_iters=3)  # [E]

                        # Guard against NaN / zero / tiny grads
                        valid = (sigma_rms > 1e-10) & torch.isfinite(sigma_max) & torch.isfinite(sigma_rms)
                        kappa = torch.where(
                            valid,
                            (sigma_max / sigma_rms.clamp(min=1e-10)).clamp(min=1.0),
                            torch.ones_like(sigma_rms),
                        )

                        # EMA update only on valid experts
                        a = self.spectral_ema
                        new_kappa = a * self._kappa_ema + (1 - a) * kappa
                        self._kappa_ema = torch.where(valid, new_kappa, self._kappa_ema)
                        self._grad_norms = frob  # fresh per-expert grad norms

                        spectral_scale = (1.0 / self._kappa_ema).clamp(min=self.spectral_floor)
                    else:
                        # Track grad norms even without spectral conditioning
                        self._grad_norms = grad.float().flatten(1).norm(dim=-1)
                        spectral_scale = torch.ones(
                            self.num_experts, device=grad.device, dtype=torch.float32
                        )

                    # Final per-expert update, vectorized:
                    # p -= (effective_lr / bc1) * spectral_scale[e] * exp_avg[e] / denom[e]
                    step_size = (effective_lr / bias_correction1) * spectral_scale  # [E]
                    step_size = step_size.to(exp_avg.dtype).view(-1, 1, 1)
                    p_local.sub_(step_size * exp_avg / denom)

                    # Periodic debug log (cheap, one sync per log interval)
                    if do_spectral and self._step_count % 100 == 0 and logger.isEnabledFor(logging.DEBUG):
                        kvals = self._kappa_ema.detach().cpu().tolist()
                        svals = spectral_scale.detach().cpu().tolist()
                        parts = [f"E{e}: κ={kvals[e]:.2f} scale={svals[e]:.3f}"
                                 for e in range(self.num_experts)]
                        logger.debug("Spectral: %s", " | ".join(parts))
                    continue

                # --- Standard Adam path (dense, shared, 1D params, etc.) ---
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = effective_lr / bias_correction1
                bc2_sqrt = math.sqrt(bias_correction2)
                denom = (exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)
                p_local.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def adamtr_param_groups(
    model,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    expert_lr_scale: float = 1.5,
    expert_weight_decay: float = 0.05,
    shared_weight_decay: float = 0.1,
    num_experts: int = 4,
    no_decay_patterns: Tuple[str, ...] = ("bias", "layernorm", "rmsnorm", "mu", "alpha"),
) -> List[Dict[str, Any]]:
    """Build parameter groups for AdamTR with pre-resolved param_type.

    Tagging happens here (once), so AdamTR.step() never has to re-classify.
    """
    groups: List[Dict[str, Any]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        no_decay = (param.dim() < 2) or any(nd in name.lower() for nd in no_decay_patterns)
        ptype = _classify_param(name, param, num_experts)

        groups.append({
            "params": [param],
            "param_name": name,
            "param_type": ptype,
            "lr": lr,
            "weight_decay": 0.0 if no_decay else weight_decay,
            "expert_lr_scale": expert_lr_scale,
            "shared_lr_scale": 1.0,
            "expert_weight_decay": 0.0 if no_decay else expert_weight_decay,
            "shared_weight_decay": 0.0 if no_decay else shared_weight_decay,
        })

    return groups
