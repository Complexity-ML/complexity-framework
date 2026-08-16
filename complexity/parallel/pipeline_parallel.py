"""
Pipeline Parallel (PP) - Split layers across GPUs.

Pipeline Parallelism splits the model by layers:
- GPU 0 holds layers 0-7
- GPU 1 holds layers 8-15
- etc.

Micro-batching is used to keep all GPUs busy:
- Split batch into micro-batches
- Pipeline them through the stages

References:
- GPipe: https://arxiv.org/abs/1811.06965
- PipeDream: https://arxiv.org/abs/1806.03377
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, List, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum


class PipelineSchedule(Enum):
    """Pipeline schedule types."""
    GPIPE = "gpipe"           # All-forward then all-backward
    ONE_F_ONE_B = "1f1b"      # Interleaved forward-backward
    INTERLEAVED = "interleaved"  # Virtual pipeline stages


@dataclass
class PipelineConfig:
    """Configuration for pipeline parallelism."""
    num_stages: int              # Number of pipeline stages (GPUs)
    num_micro_batches: int = 4   # Number of micro-batches per batch
    schedule: PipelineSchedule = PipelineSchedule.ONE_F_ONE_B


def get_pipeline_parallel_rank() -> int:
    """Get pipeline parallel rank (which stage this GPU is)."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_pipeline_parallel_world_size() -> int:
    """Get number of pipeline stages."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


# =============================================================================
# Pipeline Communication
# =============================================================================

def send_forward(tensor: torch.Tensor, next_rank: int):
    """Send tensor to the next pipeline stage."""
    if not dist.is_initialized():
        return

    dist.send(tensor, next_rank)


def recv_forward(tensor_shape: Tuple[int, ...], dtype: torch.dtype, prev_rank: int, device: torch.device) -> torch.Tensor:
    """Receive tensor from the previous pipeline stage."""
    if not dist.is_initialized():
        return torch.zeros(tensor_shape, dtype=dtype, device=device)

    tensor = torch.zeros(tensor_shape, dtype=dtype, device=device)
    dist.recv(tensor, prev_rank)
    return tensor


def send_backward(tensor: torch.Tensor, prev_rank: int):
    """Send gradient to the previous pipeline stage."""
    if not dist.is_initialized():
        return

    dist.send(tensor, prev_rank)


def recv_backward(tensor_shape: Tuple[int, ...], dtype: torch.dtype, next_rank: int, device: torch.device) -> torch.Tensor:
    """Receive gradient from the next pipeline stage."""
    if not dist.is_initialized():
        return torch.zeros(tensor_shape, dtype=dtype, device=device)

    tensor = torch.zeros(tensor_shape, dtype=dtype, device=device)
    dist.recv(tensor, next_rank)
    return tensor


# =============================================================================
# Pipeline Stage
# =============================================================================

class PipelineStage(nn.Module):
    """
    A single stage in the pipeline.

    Contains a subset of the model's layers.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        stage_id: int,
        num_stages: int,
    ):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.num_stages = num_stages

        self.is_first_stage = (stage_id == 0)
        self.is_last_stage = (stage_id == num_stages - 1)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward through this stage's layers."""
        for layer in self.layers:
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            hidden_states = layer(hidden_states, **kwargs)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        return hidden_states


# =============================================================================
# Pipeline Scheduler
# =============================================================================

class PipelineScheduler:
    """
    Manages the pipeline schedule for training.

    Implements different scheduling strategies:
    - GPipe: Simple all-forward then all-backward
    - 1F1B: Interleaved for better memory efficiency
    """

    def __init__(
        self,
        config: PipelineConfig,
        stage: PipelineStage,
    ):
        self.config = config
        self.stage = stage
        self.num_micro_batches = config.num_micro_batches

    def forward_backward(
        self,
        micro_batches: List[torch.Tensor],
        loss_fn: Callable,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Execute forward and backward passes according to schedule.

        Args:
            micro_batches: List of micro-batch inputs
            loss_fn: Function to compute loss (only on last stage)

        Returns:
            total_loss: Sum of losses across micro-batches
            gradients: Gradients to send backward
        """
        if self.config.schedule == PipelineSchedule.GPIPE:
            return self._gpipe_schedule(micro_batches, loss_fn)
        elif self.config.schedule == PipelineSchedule.ONE_F_ONE_B:
            return self._1f1b_schedule(micro_batches, loss_fn)
        else:
            raise ValueError(f"Unknown schedule: {self.config.schedule}")

    def _gpipe_schedule(
        self,
        micro_batches: List[torch.Tensor],
        loss_fn: Callable,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        GPipe schedule: all forward passes, then all backward passes.

        Simple but has high memory usage (stores all activations).
        """
        outputs = []
        total_loss = torch.tensor(0.0, device=micro_batches[0].device)

        # Forward passes
        for mb in micro_batches:
            out = self.stage(mb)
            outputs.append(out)

            if self.stage.is_last_stage:
                loss = loss_fn(out)
                total_loss = total_loss + loss

        # Backward passes (reverse order)
        for out in reversed(outputs):
            if self.stage.is_last_stage:
                out.backward(retain_graph=True)

        return total_loss, []

    def _1f1b_schedule(
        self,
        micro_batches: List[torch.Tensor],
        loss_fn: Callable,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        1F1B schedule: interleave forward and backward passes.

        Better memory efficiency than GPipe.

        Schedule for 4 micro-batches on 2 stages:
        Stage 0: F0 F1 F2 F3 B3 B2 B1 B0
        Stage 1:    F0 F1 F2 B3 F3 B2 B1 B0
        """
        num_warmup = self.config.num_stages - self.stage.stage_id - 1
        num_mb = len(micro_batches)

        outputs = []
        total_loss = torch.tensor(0.0, device=micro_batches[0].device)

        # Warmup: forward passes only
        for i in range(min(num_warmup, num_mb)):
            out = self.stage(micro_batches[i])
            outputs.append(out)
            if self.stage.is_last_stage:
                total_loss = total_loss + loss_fn(out)

        # Steady state: 1 forward, 1 backward
        for i in range(num_warmup, num_mb):
            # Forward
            out = self.stage(micro_batches[i])
            outputs.append(out)
            if self.stage.is_last_stage:
                total_loss = total_loss + loss_fn(out)

            # Backward for earliest output
            if outputs:
                out_to_backward = outputs.pop(0)
                if self.stage.is_last_stage:
                    out_to_backward.backward(retain_graph=True)

        # Cooldown: backward passes only
        for out in outputs:
            if self.stage.is_last_stage:
                out.backward(retain_graph=True)

        return total_loss, []


# =============================================================================
# Pipeline Model Wrapper
# =============================================================================

class PipelineModel(nn.Module):
    """
    Wrapper to enable pipeline parallelism for a model.

    Usage:
        model = ComplexityModel(config)
        pp_model = PipelineModel(
            model,
            num_stages=4,
            num_micro_batches=8,
        )

        # Training
        loss = pp_model(input_ids, labels=labels)
        loss.backward()
    """

    def __init__(
        self,
        model: nn.Module,
        num_stages: int,
        num_micro_batches: int = 4,
        schedule: PipelineSchedule = PipelineSchedule.ONE_F_ONE_B,
    ):
        super().__init__()

        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches

        # Get current stage
        self.stage_id = get_pipeline_parallel_rank() % num_stages

        # Split model into stages
        self.stage = self._create_stage(model)

        # Create scheduler
        self.config = PipelineConfig(
            num_stages=num_stages,
            num_micro_batches=num_micro_batches,
            schedule=schedule,
        )
        self.scheduler = PipelineScheduler(self.config, self.stage)

    def _create_stage(self, model: nn.Module) -> PipelineStage:
        """Split model layers into pipeline stages."""
        # Get all transformer layers
        if hasattr(model, 'layers'):
            all_layers = list(model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            all_layers = list(model.transformer.layers)
        else:
            raise ValueError("Cannot find layers in model")

        num_layers = len(all_layers)
        layers_per_stage = num_layers // self.num_stages

        # Assign layers to this stage
        start_idx = self.stage_id * layers_per_stage
        end_idx = start_idx + layers_per_stage
        if self.stage_id == self.num_stages - 1:
            end_idx = num_layers  # Last stage gets remaining layers

        stage_layers = nn.ModuleList(all_layers[start_idx:end_idx])

        # Also include embeddings on first stage, LM head on last stage
        if self.stage_id == 0 and hasattr(model, 'embed_tokens'):
            self.embed_tokens = model.embed_tokens
        if self.stage_id == self.num_stages - 1:
            if hasattr(model, 'norm'):
                self.norm = model.norm
            if hasattr(model, 'lm_head'):
                self.lm_head = model.lm_head

        return PipelineStage(
            layers=stage_layers,
            stage_id=self.stage_id,
            num_stages=self.num_stages,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with pipeline parallelism.

        Args:
            input_ids: Input token IDs [batch, seq]
            labels: Optional labels for loss computation
            **kwargs: Additional arguments

        Returns:
            loss if labels provided, else hidden states
        """
        batch_size = input_ids.size(0)
        micro_batch_size = batch_size // self.num_micro_batches

        # Split into micro-batches
        micro_batches = input_ids.split(micro_batch_size, dim=0)

        if labels is not None:
            micro_labels = labels.split(micro_batch_size, dim=0)

            def loss_fn(output):
                # Placeholder - actual implementation needs proper loss
                return output.mean()

            total_loss, _ = self.scheduler.forward_backward(
                list(micro_batches),
                loss_fn,
            )
            return total_loss / self.num_micro_batches

        else:
            # Inference mode
            outputs = []
            for mb in micro_batches:
                out = self.stage(mb, **kwargs)
                outputs.append(out)
            return torch.cat(outputs, dim=0)


def split_model_for_pipeline(
    model: nn.Module,
    num_stages: int,
) -> List[nn.Module]:
    """
    Split a model into pipeline stages.

    Args:
        model: The full model to split
        num_stages: Number of pipeline stages

    Returns:
        List of stage modules
    """
    stages = []

    # Get transformer layers
    if hasattr(model, 'layers'):
        all_layers = list(model.layers)
    else:
        raise ValueError("Model must have 'layers' attribute")

    num_layers = len(all_layers)
    layers_per_stage = num_layers // num_stages

    for stage_id in range(num_stages):
        start_idx = stage_id * layers_per_stage
        end_idx = start_idx + layers_per_stage
        if stage_id == num_stages - 1:
            end_idx = num_layers

        stage_layers = nn.ModuleList(all_layers[start_idx:end_idx])
        stages.append(PipelineStage(
            layers=stage_layers,
            stage_id=stage_id,
            num_stages=num_stages,
        ))

    return stages
