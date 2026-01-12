"""
Advanced Optimizers for LLM Training.

Implements modern optimizers beyond Adam:
- LION: Evolved Sign Momentum
- Sophia: Second-order clipped stochastic optimization
- 8-bit Adam: Memory-efficient Adam
- CAME: Confidence-guided Adaptive Memory Efficient
- Adalomo: Adaptive Low-Memory Optimization

References:
- LION: https://arxiv.org/abs/2302.06675
- Sophia: https://arxiv.org/abs/2305.14342
- 8-bit Adam: https://arxiv.org/abs/2110.02861
- CAME: https://arxiv.org/abs/2307.02047
"""

import torch
from torch.optim import Optimizer
from typing import Callable, Optional, Tuple, List
import math


# =============================================================================
# LION Optimizer
# =============================================================================

class Lion(Optimizer):
    """
    LION: Evolved Sign Momentum optimizer.

    Key properties:
    - Uses sign of momentum (not magnitude) for updates
    - More memory efficient than Adam (no v state)
    - Often achieves better generalization
    - Requires lower learning rate than Adam (~10x smaller)

    Reference: https://arxiv.org/abs/2302.06675
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        """
        Args:
            params: Model parameters
            lr: Learning rate (typically 3e-4 to 1e-3 / 10)
            betas: Momentum coefficients (β1, β2)
            weight_decay: Weight decay coefficient
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get gradient
                grad = p.grad

                # Get state
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # LION update: sign(β1 * m + (1-β1) * g)
                update = exp_avg * beta1 + grad * (1 - beta1)

                # Apply sign update
                p.add_(torch.sign(update), alpha=-group['lr'])

                # Update momentum: m = β2 * m + (1-β2) * g
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


# =============================================================================
# Sophia Optimizer
# =============================================================================

class Sophia(Optimizer):
    """
    Sophia: Second-order Clipped Stochastic Optimization.

    Uses Hessian diagonal estimation for adaptive learning rates,
    with clipping to prevent instability.

    Key properties:
    - Uses second-order information (Hessian diagonal)
    - Clips updates for stability
    - ~2x faster convergence than Adam on LLMs
    - Requires periodic Hessian updates

    Reference: https://arxiv.org/abs/2305.14342
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.96, 0.99),
        rho: float = 0.04,  # Clipping threshold
        weight_decay: float = 0.1,
        hessian_update_interval: int = 10,
    ):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Momentum coefficients
            rho: Clipping threshold for updates
            weight_decay: Weight decay (AdamW style)
            hessian_update_interval: Steps between Hessian updates
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            hessian_update_interval=hessian_update_interval,
        )
        super().__init__(params, defaults)
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                hessian = state['hessian']
                beta1, beta2 = group['betas']
                rho = group['rho']

                # Weight decay (AdamW style)
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Sophia update with clipping
                # update = m / max(h, rho)
                # clipped_update = clip(update, -1, 1)
                update = exp_avg / (hessian.clamp(min=rho) + 1e-8)
                update = torch.clamp(update, -1.0, 1.0)

                p.add_(update, alpha=-group['lr'])

        return loss

    def update_hessian(self, model_output: torch.Tensor, create_graph: bool = False):
        """
        Update Hessian diagonal estimate using Gauss-Newton approximation.

        This should be called periodically (every k steps).

        Args:
            model_output: Model logits [batch, seq, vocab]
            create_graph: Whether to create computation graph
        """
        batch_size = model_output.size(0)

        # Sample from softmax distribution
        probs = torch.softmax(model_output, dim=-1)
        sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, -1)

        # Compute pseudo-loss with sampled labels
        loss = torch.nn.functional.cross_entropy(
            model_output.view(-1, model_output.size(-1)),
            sampled.view(-1),
            reduction='mean',
        )

        # Backward to get gradients
        loss.backward(create_graph=create_graph)

        # Update Hessian estimate with squared gradients
        for group in self.param_groups:
            beta2 = group['betas'][1]

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if 'hessian' not in state:
                    state['hessian'] = torch.zeros_like(p)

                # EMA of squared gradients (Gauss-Newton approximation)
                state['hessian'].mul_(beta2).addcmul_(
                    p.grad, p.grad, value=1 - beta2
                )


# =============================================================================
# 8-bit Adam
# =============================================================================

class Adam8bit(Optimizer):
    """
    8-bit Adam optimizer for memory efficiency.

    Quantizes optimizer states (m, v) to 8-bit,
    reducing memory by ~75% vs standard Adam.

    Uses block-wise quantization for accuracy.

    Reference: https://arxiv.org/abs/2110.02861
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        block_size: int = 2048,
    ):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Momentum coefficients
            eps: Epsilon for numerical stability
            weight_decay: Weight decay
            block_size: Block size for quantization
        """
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, block_size=block_size
        )
        super().__init__(params, defaults)

    def _quantize_to_8bit(
        self,
        tensor: torch.Tensor,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to 8-bit with block-wise scaling."""
        # Flatten and pad to block boundary
        flat = tensor.flatten()
        pad_size = (block_size - flat.numel() % block_size) % block_size
        if pad_size > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_size))

        # Reshape to blocks
        blocks = flat.view(-1, block_size)

        # Compute per-block scales (max absolute value)
        scales = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)

        # Normalize and quantize to [-127, 127]
        normalized = blocks / scales
        quantized = (normalized * 127).round().to(torch.int8)

        return quantized, scales.squeeze()

    def _dequantize_from_8bit(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        original_shape: torch.Size,
        block_size: int,
    ) -> torch.Tensor:
        """Dequantize 8-bit tensor back to float."""
        # Dequantize
        blocks = quantized.float() / 127.0
        blocks = blocks * scales.unsqueeze(1)

        # Flatten and remove padding
        flat = blocks.flatten()
        numel = torch.tensor(original_shape).prod().item()

        return flat[:numel].view(original_shape)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            block_size = group['block_size']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    # Store quantized states and scales
                    m_quant, m_scales = self._quantize_to_8bit(
                        torch.zeros_like(p), block_size
                    )
                    v_quant, v_scales = self._quantize_to_8bit(
                        torch.zeros_like(p), block_size
                    )
                    state['exp_avg_quant'] = m_quant
                    state['exp_avg_scales'] = m_scales
                    state['exp_avg_sq_quant'] = v_quant
                    state['exp_avg_sq_scales'] = v_scales

                state['step'] += 1

                # Dequantize states
                exp_avg = self._dequantize_from_8bit(
                    state['exp_avg_quant'],
                    state['exp_avg_scales'],
                    p.shape,
                    block_size,
                )
                exp_avg_sq = self._dequantize_from_8bit(
                    state['exp_avg_sq_quant'],
                    state['exp_avg_sq_scales'],
                    p.shape,
                    block_size,
                )

                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] / bias_correction1

                # Update
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Re-quantize states
                state['exp_avg_quant'], state['exp_avg_scales'] = \
                    self._quantize_to_8bit(exp_avg, block_size)
                state['exp_avg_sq_quant'], state['exp_avg_sq_scales'] = \
                    self._quantize_to_8bit(exp_avg_sq, block_size)

        return loss


# =============================================================================
# CAME Optimizer
# =============================================================================

class CAME(Optimizer):
    """
    CAME: Confidence-guided Adaptive Memory Efficient optimizer.

    Combines:
    - Confidence-guided update clipping
    - Memory-efficient state management
    - Better stability than Adam for large models

    Reference: https://arxiv.org/abs/2307.02047
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
        eps: Tuple[float, float] = (1e-30, 1e-16),
        weight_decay: float = 0.0,
        clip_threshold: float = 1.0,
    ):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            betas: (β1, β2, β3) for momentum, variance, confidence
            eps: (eps1, eps2) for numerical stability
            weight_decay: Weight decay
            clip_threshold: Gradient clipping threshold
        """
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, clip_threshold=clip_threshold
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            eps1, eps2 = group['eps']
            clip_threshold = group['clip_threshold']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients")

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['confidence'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                confidence = state['confidence']

                # Weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Compute denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps1)

                # Compute update direction
                update = exp_avg / bias_correction1 / denom

                # Update confidence (EMA of squared updates)
                update_sq = update ** 2
                confidence.mul_(beta3).add_(update_sq, alpha=1 - beta3)

                # Confidence-guided clipping
                conf_mask = (confidence > eps2).float()
                r = (update_sq / (confidence + eps2)).sqrt()
                r = torch.clamp(r, max=clip_threshold)

                # Apply clipped update
                clipped_update = update * (conf_mask + (1 - conf_mask) * r.clamp(max=1.0))
                p.add_(clipped_update, alpha=-group['lr'])

        return loss


# =============================================================================
# AdaLomo Optimizer
# =============================================================================

class AdaLomo(Optimizer):
    """
    Adalomo: Adaptive Low-Memory Optimization.

    Designed for fine-tuning large models with minimal memory:
    - No persistent optimizer states
    - Layer-wise learning rates
    - Gradient normalization for stability

    Memory usage: O(1) vs O(2N) for Adam

    Suitable for:
    - Fine-tuning 7B+ models on consumer GPUs
    - When memory is the bottleneck
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        grad_norm_clip: float = 1.0,
        use_layer_lr: bool = True,
    ):
        """
        Args:
            params: Model parameters
            lr: Base learning rate
            weight_decay: Weight decay
            grad_norm_clip: Maximum gradient norm
            use_layer_lr: Use layer-wise learning rates
        """
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            grad_norm_clip=grad_norm_clip,
            use_layer_lr=use_layer_lr,
        )
        super().__init__(params, defaults)

        # Compute layer-wise learning rate multipliers
        if use_layer_lr:
            self._init_layer_lrs()

    def _init_layer_lrs(self):
        """Initialize layer-wise learning rate multipliers."""
        total_params = sum(
            p.numel() for group in self.param_groups for p in group['params']
        )

        param_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # Later layers get higher LR (common for fine-tuning)
                depth_ratio = param_idx / max(len(list(self.param_groups[0]['params'])), 1)
                state['lr_mult'] = 0.5 + 0.5 * depth_ratio
                param_idx += 1

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Global gradient norm for clipping
        total_norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        for group in self.param_groups:
            clip_coef = group['grad_norm_clip'] / (total_norm + 1e-6)
            clip_coef = min(1.0, clip_coef)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad * clip_coef
                state = self.state[p]

                # Get layer-wise LR multiplier
                lr_mult = state.get('lr_mult', 1.0) if group['use_layer_lr'] else 1.0
                lr = group['lr'] * lr_mult

                # Weight decay (decoupled)
                if group['weight_decay'] != 0:
                    p.mul_(1 - lr * group['weight_decay'])

                # SGD update (no momentum states needed!)
                p.add_(grad, alpha=-lr)

        return loss


# =============================================================================
# Utility: Get Optimizer
# =============================================================================

def get_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs,
) -> Optimizer:
    """
    Get optimizer by name.

    Args:
        model: Model to optimize
        optimizer_type: One of "adamw", "lion", "sophia", "adam8bit", "came", "adalomo"
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Configured optimizer
    """
    # Separate weight decay for different parameter types
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'embedding' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer_map = {
        "adamw": lambda: torch.optim.AdamW(param_groups, lr=lr, **kwargs),
        "adam": lambda: torch.optim.Adam(param_groups, lr=lr, **kwargs),
        "sgd": lambda: torch.optim.SGD(param_groups, lr=lr, **kwargs),
        "lion": lambda: Lion(param_groups, lr=lr / 10, **kwargs),  # Lion needs lower LR
        "sophia": lambda: Sophia(param_groups, lr=lr / 2, **kwargs),
        "adam8bit": lambda: Adam8bit(param_groups, lr=lr, **kwargs),
        "came": lambda: CAME(param_groups, lr=lr, **kwargs),
        "adalomo": lambda: AdaLomo(param_groups, lr=lr * 10, **kwargs),  # AdaLomo needs higher LR
    }

    if optimizer_type.lower() not in optimizer_map:
        raise ValueError(
            f"Unknown optimizer: {optimizer_type}. "
            f"Available: {list(optimizer_map.keys())}"
        )

    return optimizer_map[optimizer_type.lower()]()
