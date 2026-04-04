"""
Muon: Momentum Orthogonalized by Newton-Schulz.

Applies orthogonalized momentum updates to 2D+ weight matrices,
with an auxiliary AdamW for all other parameters (embeddings, biases, norms).

Converges ~2x faster than AdamW on LLM pre-training, especially in
large-batch regimes. The Newton-Schulz iteration runs in bfloat16 for speed.

Reference: https://github.com/KellerJordan/Muon
Blog: https://kellerjordan.github.io/posts/muon/
"""

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Optional, Callable, Tuple


def newton_schulz(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Approximate the nearest orthogonal matrix via Newton-Schulz iteration.

    Given M with SVD = U @ diag(S) @ V.T, converges to U @ V.T.
    Each iteration applies a quintic polynomial to singular values:
        phi(s) = a*s + b*s^3 + c*s^5

    Runs entirely in bfloat16 for speed on modern GPUs.
    FLOP overhead: ~0.7% for GPT-2 scale models.

    Args:
        M: 2D tensor (rows, cols) with rows <= cols
        steps: Number of Newton-Schulz iterations (5 is standard)

    Returns:
        Orthogonalized matrix (same shape as M)
    """
    # Optimized coefficients (Keller Jordan)
    a, b, c = 3.4445, -4.7750, 2.0315

    X = M.bfloat16()
    X /= X.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    return X.to(M.dtype)


def newton_schulz_adaptive(
    M: torch.Tensor,
    min_steps: int = 5,
    max_steps: int = 10,
    tol: float = 1e-2,
) -> tuple:
    """
    Newton-Schulz with adaptive iterations and convergence monitoring.

    Iterates until ||X^T X - I||_F < tol or max_steps reached.
    Returns (orthogonalized_matrix, residual_norm, steps_used).

    For noisy gradients (tail experts with fewer tokens), the singular
    value spread is wider → more iterations needed for convergence.

    Ref: KellerJordan/Muon#65
    """
    a, b, c = 3.4445, -4.7750, 2.0315

    X = M.bfloat16()
    X /= X.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-7)

    residual = float('inf')
    steps_used = 0

    for i in range(max_steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        steps_used = i + 1

        # Check convergence after min_steps
        if i + 1 >= min_steps:
            identity = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
            residual = (X @ X.mT - identity).norm().item()
            if residual < tol:
                break

    return X.to(M.dtype), residual, steps_used


class Muon(Optimizer):
    """
    Muon optimizer for 2D+ weight matrices.

    Algorithm per step:
        1. Momentum:  M = beta * M + (1 - beta) * grad
        2. Nesterov:  update = beta * M + (1 - beta) * grad
        3. Reshape to 2D if needed (e.g. conv filters)
        4. Newton-Schulz orthogonalization (5 iterations in bf16)
        5. Scale by sqrt(max(1, rows/cols))
        6. Decoupled weight decay
        7. Parameter update

    Only use this for hidden-layer 2D+ weights. Use AdamW for
    embeddings, biases, norms, and LM heads.

    Args:
        params: Parameters to optimize (must be ndim >= 2)
        lr: Learning rate (default: 0.02, has built-in muP scaling)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iterations (default: 5)
        weight_decay: Decoupled weight decay (default: 0.01)

    Reference: https://github.com/KellerJordan/Muon
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']

                # Momentum update
                buf.lerp_(grad, 1 - beta)

                # Nesterov lookahead
                if nesterov:
                    update = grad.lerp(buf, beta)
                else:
                    update = buf.clone()

                # Reshape to 2D for orthogonalization
                original_shape = update.shape
                if update.ndim > 2:
                    update = update.view(update.shape[0], -1)

                rows, cols = update.shape

                # Transpose if tall matrix (Newton-Schulz needs rows <= cols)
                transposed = rows > cols
                if transposed:
                    update = update.T

                # Newton-Schulz orthogonalization
                update = newton_schulz(update, steps=ns_steps)

                # Transpose back
                if transposed:
                    update = update.T

                # Scale correction
                update *= max(1, rows / cols) ** 0.5

                # Reshape back
                update = update.reshape(original_shape)

                # Decoupled weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # Update parameters
                p.add_(update, alpha=-lr)

        return loss


class MuonWithAdamW(Optimizer):
    """
    Combined optimizer: Muon for 2D+ hidden weights, AdamW for the rest.

    Automatically routes parameters to the correct optimizer based on
    dimensionality and name. This is the recommended entry point.

    Parameter routing:
        - Muon: Hidden-layer weights with ndim >= 2
        - AdamW: Embeddings, biases, norms, LM head, mu params

    Args:
        muon_params: Parameters for Muon (2D+ hidden weights)
        adam_params: Parameters for AdamW (everything else)
        lr: Learning rate for Muon (default: 0.02)
        adam_lr: Learning rate for AdamW (default: 3e-4)
        momentum: Muon momentum (default: 0.95)
        nesterov: Muon Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iterations (default: 5)
        adam_betas: AdamW betas (default: (0.9, 0.95))
        adam_eps: AdamW epsilon (default: 1e-8)
        weight_decay: Weight decay for both (default: 0.01)

    Example:
        muon_params, adam_params = split_params_for_muon(model)
        optimizer = MuonWithAdamW(muon_params, adam_params)
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
    ):
        # We store both sub-optimizers but present a unified interface
        self.muon = Muon(
            muon_params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        self.adam = torch.optim.AdamW(
            adam_params,
            lr=adam_lr,
            betas=adam_betas,
            eps=adam_eps,
            weight_decay=weight_decay,
        )

        # Combine param_groups for compatibility with schedulers
        super().__init__(
            self.muon.param_groups + self.adam.param_groups,
            defaults={},
        )
        # Override param_groups to point to the live sub-optimizer groups
        self.param_groups = self.muon.param_groups + self.adam.param_groups

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = self.muon.step(closure)
        self.adam.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            'muon': self.muon.state_dict(),
            'adam': self.adam.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict['muon'])
        self.adam.load_state_dict(state_dict['adam'])


def split_params_for_muon(model: torch.nn.Module):
    """
    Split model parameters into Muon-eligible and AdamW-fallback groups.

    Muon handles: hidden-layer 2D+ weight matrices
    AdamW handles: embeddings, biases, norms, LM head, mu/dynamics params

    Args:
        model: The model to split parameters for

    Returns:
        (muon_params, adam_params): Two lists of parameter dicts
            muon_params has weight_decay applied
            adam_params is split into decay/no-decay groups

    Example:
        muon_params, adam_params = split_params_for_muon(model)
        optimizer = MuonWithAdamW(muon_params, adam_params)
    """
    muon_params = []
    adam_decay = []
    adam_no_decay = []

    # Names that should always go to AdamW
    adam_keywords = {
        'embed', 'lm_head', 'head',  # Embedding / output layers
        'bias',                        # Biases
        'norm', 'ln_',                 # Normalization
        '.mu', 'mu_', 'dynamics',      # INL dynamics / mu params
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Route to AdamW if: 1D param, or name matches adam_keywords
        is_adam = param.ndim < 2 or any(k in name for k in adam_keywords)

        if is_adam:
            # Further split AdamW params into decay / no-decay
            if param.ndim < 2 or 'bias' in name or 'norm' in name or '.mu' in name:
                adam_no_decay.append(param)
            else:
                adam_decay.append(param)
        else:
            muon_params.append(param)

    muon_groups = [{"params": muon_params}] if muon_params else []
    adam_groups = []
    if adam_decay:
        adam_groups.append({"params": adam_decay})
    if adam_no_decay:
        adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})

    return muon_groups, adam_groups
