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

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Optional, Callable, Tuple, List, Dict, Any

from .muon import newton_schulz, newton_schulz_adaptive

import logging
logger = logging.getLogger("complexity.muon_tr")


def _to_local(t: torch.Tensor) -> torch.Tensor:
    """Convert DTensor to local tensor (FSDP v2 compat)."""
    if hasattr(t, 'to_local'):
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
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_max_steps: int = 10,
        ns_tol: float = 1e-2,
        adaptive_ns: bool = True,
        weight_decay: float = 0.01,
        expert_lr_scale: float = 1.5,
        shared_lr_scale: float = 1.0,
        expert_weight_decay: float = 0.005,
        shared_weight_decay: float = 0.01,
        token_counts: Optional[torch.Tensor] = None,
        num_experts: int = 4,
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
        # Per-expert NS convergence monitoring
        self._ns_residuals: Dict[int, float] = {}
        self._ns_steps_used: Dict[int, int] = {}
        self._step_count = 0

    def update_token_counts(self, token_counts: torch.Tensor):
        """Update per-expert token counts for gradient scaling."""
        self.token_counts = token_counts

    def get_ns_diagnostics(self) -> Dict[int, Dict[str, float]]:
        """Return per-expert NS convergence diagnostics."""
        return {
            e: {"residual": self._ns_residuals.get(e, 0.0), "steps": self._ns_steps_used.get(e, 0)}
            for e in range(self.num_experts)
        }

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            lr = group['lr']
            beta = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            ns_max_steps = group.get('ns_max_steps', 10)
            ns_tol = group.get('ns_tol', 1e-2)
            adaptive_ns = group.get('adaptive_ns', True)
            param_type = group.get('param_type', 'dense')

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = _to_local(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']

                # === Feature 3: Gradient scaling per expert ===
                p_local = _to_local(p)
                if param_type == 'expert' and self.token_counts is not None:
                    if p_local.dim() == 3 and p_local.shape[0] == self.num_experts:
                        mean_count = self.token_counts.float().mean()
                        for e in range(self.num_experts):
                            if self.token_counts[e] > 0:
                                scale = (mean_count / self.token_counts[e].float()).clamp(0.5, 2.0)
                                grad[e] = grad[e] * scale

                # Momentum update
                buf.lerp_(grad, 1 - beta)

                # Nesterov lookahead
                if nesterov:
                    update = grad.lerp(buf, beta)
                else:
                    update = buf.clone()

                # === Feature 1: Per-expert LR ===
                if param_type == 'expert':
                    effective_lr = lr * group['expert_lr_scale']
                elif param_type == 'shared':
                    effective_lr = lr * group['shared_lr_scale']
                else:
                    effective_lr = lr

                # === Feature 2: Expert-aware weight decay ===
                if param_type == 'expert':
                    wd = group['expert_weight_decay']
                elif param_type == 'shared':
                    wd = group['shared_weight_decay']
                else:
                    wd = group['weight_decay']

                # For expert tensors [E, H, I], orthogonalize each expert slice independently
                # Per-expert NS is more correct because each expert sees different token
                # distributions under Zipf routing → different SV spectra
                if p_local.dim() == 3 and p_local.shape[0] == self.num_experts and param_type == 'expert':
                    for e in range(self.num_experts):
                        slice_update = update[e]
                        rows, cols = slice_update.shape
                        transposed = rows > cols
                        if transposed:
                            slice_update = slice_update.T

                        if adaptive_ns:
                            slice_update, residual, steps_used = newton_schulz_adaptive(
                                slice_update, min_steps=ns_steps, max_steps=ns_max_steps, tol=ns_tol,
                            )
                            self._ns_residuals[e] = residual
                            self._ns_steps_used[e] = steps_used
                        else:
                            slice_update = newton_schulz(slice_update, steps=ns_steps)

                        if transposed:
                            slice_update = slice_update.T
                        slice_update *= max(1, rows / cols) ** 0.5
                        update[e] = slice_update

                    # Log per-expert NS convergence periodically
                    if self._step_count % 100 == 0 and self._ns_residuals:
                        parts = [f"E{e}: {self._ns_residuals.get(e, 0):.4f} ({self._ns_steps_used.get(e, 0)} iters)"
                                 for e in range(self.num_experts)]
                        logger.info("NS convergence ||X^T X - I||_F: %s", " | ".join(parts))
                else:
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

                    update *= max(1, rows / cols) ** 0.5
                    update = update.reshape(original_shape)

                # Decoupled weight decay
                if wd != 0:
                    p.mul_(1 - effective_lr * wd)

                # Update parameters
                p.add_(update, alpha=-effective_lr)

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
        lr: float = 0.02,
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
    ):
        self.muon_tr = MuonTR(
            muon_params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            expert_lr_scale=expert_lr_scale,
            shared_lr_scale=shared_lr_scale,
            expert_weight_decay=expert_weight_decay,
            shared_weight_decay=shared_weight_decay,
            num_experts=num_experts,
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
    expert_params = []
    shared_params = []
    dense_params = []
    adam_decay = []
    adam_no_decay = []

    adam_keywords = {
        'embed', 'lm_head', 'head',
        'bias',
        'norm', 'ln_',
        '.mu', 'mu_', 'dynamics',
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_adam = param.ndim < 2 or any(k in name for k in adam_keywords)

        if is_adam:
            if param.ndim < 2 or 'bias' in name or 'norm' in name or '.mu' in name:
                adam_no_decay.append(param)
            else:
                adam_decay.append(param)
        elif any(k in name for k in ('gate_proj_w', 'up_proj_w', 'down_proj_w')):
            expert_params.append(param)
        elif any(k in name for k in ('shared_gate', 'shared_up', 'shared_down')):
            shared_params.append(param)
        else:
            dense_params.append(param)

    muon_groups = []
    if expert_params:
        muon_groups.append({"params": expert_params, "param_type": "expert"})
    if shared_params:
        muon_groups.append({"params": shared_params, "param_type": "shared"})
    if dense_params:
        muon_groups.append({"params": dense_params, "param_type": "dense"})

    adam_groups = []
    if adam_decay:
        adam_groups.append({"params": adam_decay})
    if adam_no_decay:
        adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})

    return muon_groups, adam_groups
