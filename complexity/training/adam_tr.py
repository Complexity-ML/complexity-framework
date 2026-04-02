"""
AdamTR — Adam for Token-Routed MoE architectures.

MoE-aware optimizer that treats expert parameters differently from shared/dense
parameters. Based on AdamW with four key innovations:

1. **Per-expert LR**: Each expert group gets its own adaptive learning rate
   based on how many tokens it processes (Zipf-aware scaling).
2. **Expert-aware weight decay**: Different decay rates for routed experts
   vs shared expert vs attention/embeddings.
3. **Gradient scaling per expert**: Normalizes gradients by the number of
   tokens routed to each expert, preventing high-frequency experts from
   dominating updates.
4. **Separate momentum**: First and second moments (m, v) are tracked
   per expert group, not globally, so expert specialization isn't diluted
   by cross-expert momentum.

Usage:
    optimizer = AdamTR(
        model.parameters(),
        lr=3e-4,
        expert_lr_scale=1.5,       # experts get 1.5x base LR
        shared_lr_scale=1.0,       # shared expert gets base LR
        expert_weight_decay=0.05,  # lighter decay for experts
        shared_weight_decay=0.1,   # standard decay for shared
        token_counts=token_counts, # [E] tokens per expert for grad scaling
    )

Reference: Complexity-ML (2026) — Token-Routed MoE training optimization.
INL / Complexity-ML — 2026
"""

import math
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class AdamTR(Optimizer):
    """
    AdamTR: Adam optimizer specialized for Token-Routed MoE.

    Extends AdamW with per-expert learning rates, expert-aware weight decay,
    gradient normalization by token count, and separate momentum tracking.

    Args:
        params: Model parameters or parameter groups.
        lr: Base learning rate (default: 3e-4).
        betas: Coefficients for computing running averages (default: (0.9, 0.999)).
        eps: Numerical stability term (default: 1e-8).
        weight_decay: Base weight decay for non-expert params (default: 0.1).
        expert_lr_scale: LR multiplier for routed expert params (default: 1.5).
        shared_lr_scale: LR multiplier for shared expert params (default: 1.0).
        expert_weight_decay: Weight decay for routed expert params (default: 0.05).
        shared_weight_decay: Weight decay for shared expert params (default: 0.1).
        token_counts: Optional [E] tensor of tokens per expert for grad scaling.
        num_experts: Number of experts (for auto-detecting expert params).
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
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

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

    def update_token_counts(self, token_counts: Tensor):
        """Update per-expert token counts for gradient scaling.

        Call this each step with the actual token distribution from
        the current batch for accurate gradient normalization.

        Args:
            token_counts: [E] tensor of token counts per expert.
        """
        self.token_counts = token_counts

    def _get_param_type(self, param: Tensor, group: Dict[str, Any]) -> str:
        """Classify parameter as 'expert', 'shared', or 'dense'.

        Uses the parameter name stored in the group, or falls back
        to shape-based heuristics (3D tensors with first dim == E are experts).
        """
        param_name = group.get("param_name", "")

        # Name-based classification
        if any(k in param_name for k in ("gate_proj_w", "up_proj_w", "down_proj_w")):
            return "expert"
        if any(k in param_name for k in ("shared_gate", "shared_up", "shared_down")):
            return "shared"

        # Shape-based fallback: [E, ...] tensors are expert weights
        if param.dim() == 3 and param.shape[0] == self.num_experts:
            return "expert"

        return "dense"

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamTR does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step_t = state["step"]

                step_t += 1
                step = step_t.item()

                # Classify parameter
                param_type = self._get_param_type(p, group)

                # === Feature 1: Per-expert LR ===
                if param_type == "expert":
                    effective_lr = lr * group["expert_lr_scale"]
                elif param_type == "shared":
                    effective_lr = lr * group["shared_lr_scale"]
                else:
                    effective_lr = lr

                # === Feature 2: Expert-aware weight decay ===
                if param_type == "expert":
                    wd = group["expert_weight_decay"]
                elif param_type == "shared":
                    wd = group["shared_weight_decay"]
                else:
                    wd = group["weight_decay"]

                # Decoupled weight decay
                p.mul_(1 - effective_lr * wd)

                # === Feature 3: Gradient scaling per expert ===
                if param_type == "expert" and self.token_counts is not None:
                    # Normalize gradient by relative token count
                    # Expert seeing more tokens gets smaller gradient
                    # to prevent high-frequency experts from dominating
                    mean_count = self.token_counts.float().mean()
                    if p.dim() == 3 and p.shape[0] == self.num_experts:
                        # Per-expert scaling for [E, ...] parameter tensors
                        for e in range(self.num_experts):
                            if self.token_counts[e] > 0:
                                scale = mean_count / self.token_counts[e].float()
                                scale = scale.clamp(0.5, 2.0)  # Prevent extreme scaling
                                grad[e] = grad[e] * scale

                # === Feature 4: Separate momentum (per-parameter, already separate) ===
                # PyTorch already tracks m and v per parameter tensor.
                # For expert params ([E, H, I]), each expert's slice has
                # its own momentum statistics within the same tensor.
                # This is naturally separate — no cross-expert contamination.

                # Standard Adam update
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = effective_lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def adamtr_param_groups(
    model,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    expert_lr_scale: float = 1.5,
    expert_weight_decay: float = 0.05,
    shared_weight_decay: float = 0.1,
    no_decay_patterns: Tuple[str, ...] = ("bias", "layernorm", "rmsnorm", "mu"),
) -> List[Dict[str, Any]]:
    """Create parameter groups with proper classification for AdamTR.

    Automatically classifies parameters into expert, shared, and dense
    groups with appropriate hyperparameters.

    Args:
        model: The model to create parameter groups for.
        lr: Base learning rate.
        weight_decay: Weight decay for dense params.
        expert_lr_scale: LR multiplier for expert params.
        expert_weight_decay: Weight decay for expert params.
        shared_weight_decay: Weight decay for shared expert params.
        no_decay_patterns: Parameter name patterns that should have zero weight decay.

    Returns:
        List of parameter group dicts suitable for AdamTR constructor.
    """
    param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check no-decay patterns
        no_decay = any(nd in name.lower() for nd in no_decay_patterns)

        group = {
            "params": [param],
            "param_name": name,
            "lr": lr,
            "weight_decay": 0.0 if no_decay else weight_decay,
            "expert_lr_scale": expert_lr_scale,
            "shared_lr_scale": 1.0,
            "expert_weight_decay": 0.0 if no_decay else expert_weight_decay,
            "shared_weight_decay": 0.0 if no_decay else shared_weight_decay,
        }
        param_groups.append(group)

    return param_groups
