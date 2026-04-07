"""
Cluster-scale parallelism — 3D parallelism (TP × PP × DP).

Combines Tensor Parallelism, Pipeline Parallelism, and Data Parallelism
(FSDP) to train models that don't fit on a single GPU.

This is how GPT-3 175B was trained:
    - TP=8  (split attention/MLP within a node, 8 GPUs per node)
    - PP=16 (split layers across nodes, micro-batch pipeline)
    - DP=80 (replicate across 80 model copies, gradient all-reduce)
    - Total: 8 × 16 × 80 = 10,240 GPUs
    - Effective batch: ~1,536 sequences = 3.2M tokens/step

Usage:
    # 64 GPUs: TP=8, PP=2, DP=4
    config = ClusterConfig(
        tp_size=8,    # tensor parallel within node
        pp_size=2,    # pipeline stages across nodes
        dp_size=4,    # data parallel replicas
        micro_batch_size=64,
        num_micro_batches=4,  # pipeline micro-batches
    )
    # Effective batch = 64 × 4 (micro) × 4 (DP) = 1,024 sequences

    model = ClusterModel(model, config)
    model.train_step(batch)

    # Launch:
    torchrun --nnodes=8 --nproc_per_node=8 train.py

Architecture:
    ┌─── Node 0 (8 GPUs) ────────────────────┐
    │  TP group: GPU 0-7                      │
    │  Pipeline Stage 0 (layers 0-15)         │
    │  Each GPU holds 1/8 of each layer       │
    ├─── Node 1 (8 GPUs) ────────────────────┤
    │  TP group: GPU 8-15                     │
    │  Pipeline Stage 1 (layers 16-31)        │
    │  Each GPU holds 1/8 of each layer       │
    └─────────────────────────────────────────┘
    × DP replicas (gradient sync across replicas)

INL / Complexity-ML — 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist


@dataclass
class ClusterConfig:
    """
    3D parallelism configuration.

    tp_size × pp_size × dp_size must equal world_size.
    """
    # Parallelism dimensions
    tp_size: int = 8          # Tensor parallel (within node, NVLink)
    pp_size: int = 1          # Pipeline parallel (across nodes)
    dp_size: int = 1          # Data parallel (FSDP replicas)

    # Batch
    micro_batch_size: int = 64    # Sequences per micro-batch per DP rank
    num_micro_batches: int = 4    # Pipeline micro-batches (GPipe/1F1B)

    # Pipeline schedule
    pp_schedule: str = "1f1b"     # "gpipe" or "1f1b"

    # Precision
    precision: str = "bf16"

    # Gradient checkpointing
    gradient_checkpointing: bool = True

    # Use DDP instead of FSDP for the dp_size > 1 path. Required when using
    # torchao FP8 training: torchao Float8Linear does mixed Tensor/DTensor
    # matmuls that crash under FSDP shard but work fine under DDP.
    use_ddp: bool = False

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size * self.dp_size

    @property
    def effective_batch_size(self) -> int:
        """Total sequences per gradient step."""
        return self.micro_batch_size * self.num_micro_batches * self.dp_size

    @property
    def tokens_per_step(self, seq_len: int = 2048) -> int:
        return self.effective_batch_size * seq_len

    @classmethod
    def auto(cls, num_params: int, num_gpus: int, gpu_memory_gb: int = 80,
             micro_batch_size: int = 64, **kwargs) -> "ClusterConfig":
        """Auto-detect best TP/PP/DP split for given model + cluster."""
        return ClusterModel.estimate_config(
            num_params=num_params, num_gpus=num_gpus,
            gpu_memory_gb=gpu_memory_gb, batch_per_gpu=micro_batch_size,
        )

    def validate(self):
        actual = dist.get_world_size() if dist.is_initialized() else None
        if actual is not None and actual != self.world_size:
            raise ValueError(
                f"ClusterConfig expects {self.world_size} GPUs "
                f"(TP={self.tp_size} × PP={self.pp_size} × DP={self.dp_size}) "
                f"but world_size={actual}"
            )

    def estimate_memory_per_gpu(self, num_params: int, seq_len: int = 2048) -> Dict[str, float]:
        """Estimate memory usage per GPU in GB."""
        # Params sharded across TP × DP
        param_bytes = num_params * 2 / (self.tp_size * self.dp_size)  # bf16
        # Optimizer (Adam fp32: params + momentum + variance)
        optim_bytes = num_params * 12 / (self.tp_size * self.dp_size)
        # Gradients
        grad_bytes = num_params * 4 / (self.tp_size * self.dp_size)
        # Activations (per pipeline stage, TP-sharded)
        num_layers_per_stage = 1  # placeholder, depends on model
        act_bytes = (
            num_layers_per_stage * self.micro_batch_size * seq_len
            * (num_params / 12 / num_layers_per_stage) ** 0.5  # ~hidden_size
            * 2 / self.tp_size
        )

        to_gb = lambda b: b / 1e9
        return {
            "params_gb": to_gb(param_bytes),
            "optimizer_gb": to_gb(optim_bytes),
            "gradients_gb": to_gb(grad_bytes),
            "activations_gb": to_gb(act_bytes),
            "total_gb": to_gb(param_bytes + optim_bytes + grad_bytes + act_bytes),
        }


class ProcessGroupManager:
    """
    Creates and manages process groups for 3D parallelism.

    Given world_size = TP × PP × DP, assigns each rank to:
    - A TP group (ranks that share the same layers, same pipeline stage)
    - A PP group (ranks that form a pipeline, same TP position)
    - A DP group (ranks that are replicas, same TP pos + PP stage)
    """

    def __init__(self, config: ClusterConfig):
        self.config = config
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        config.validate()

        # Compute position in 3D grid
        # Layout: [DP, PP, TP] — TP is innermost (within node)
        self.tp_rank = self.rank % config.tp_size
        self.pp_rank = (self.rank // config.tp_size) % config.pp_size
        self.dp_rank = self.rank // (config.tp_size * config.pp_size)

        # Create process groups
        self.tp_group = self._build_tp_groups()
        self.pp_group = self._build_pp_groups()
        self.dp_group = self._build_dp_groups()

    def _build_tp_groups(self) -> dist.ProcessGroup:
        """TP group: same PP stage, same DP replica."""
        my_group = None
        for dp in range(self.config.dp_size):
            for pp in range(self.config.pp_size):
                ranks = [
                    dp * self.config.pp_size * self.config.tp_size
                    + pp * self.config.tp_size
                    + tp
                    for tp in range(self.config.tp_size)
                ]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    my_group = group
        return my_group

    def _build_pp_groups(self) -> dist.ProcessGroup:
        """PP group: same TP position, same DP replica."""
        my_group = None
        for dp in range(self.config.dp_size):
            for tp in range(self.config.tp_size):
                ranks = [
                    dp * self.config.pp_size * self.config.tp_size
                    + pp * self.config.tp_size
                    + tp
                    for pp in range(self.config.pp_size)
                ]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    my_group = group
        return my_group

    def _build_dp_groups(self) -> dist.ProcessGroup:
        """DP group: same TP position, same PP stage."""
        my_group = None
        for pp in range(self.config.pp_size):
            for tp in range(self.config.tp_size):
                ranks = [
                    dp * self.config.pp_size * self.config.tp_size
                    + pp * self.config.tp_size
                    + tp
                    for dp in range(self.config.dp_size)
                ]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    my_group = group
        return my_group

    def __repr__(self) -> str:
        return (
            f"ProcessGroupManager(rank={self.rank}, "
            f"tp={self.tp_rank}/{self.config.tp_size}, "
            f"pp={self.pp_rank}/{self.config.pp_size}, "
            f"dp={self.dp_rank}/{self.config.dp_size})"
        )


class ClusterModel(nn.Module):
    """
    Wraps a ComplexityModel with 3D parallelism for cluster-scale training.

    Orchestrates:
    1. Tensor Parallelism — split QKV/MLP projections within TP group
    2. Pipeline Parallelism — split layers across PP stages
    3. Data Parallelism — FSDP across DP replicas

    Usage:
        model = ComplexityModel(config)
        cluster = ClusterModel(model, ClusterConfig(tp_size=8, pp_size=2, dp_size=4))
        loss = cluster.train_step(batch)
    """

    def __init__(self, model: nn.Module, config: ClusterConfig):
        super().__init__()
        self.config = config

        # Create DP process groups (needed for FSDP when using TP+DP)
        self._dp_group = None
        if config.dp_size > 1 and config.tp_size > 1:
            self._dp_group = self._build_dp_groups(config)

        # Only create full PG manager if we need PP
        self.pg = ProcessGroupManager(config) if config.pp_size > 1 else None

        # 1. Apply Tensor Parallelism (split within TP group)
        if config.tp_size > 1:
            from .tensor_parallel import init_tensor_parallel_group, make_parallel
            init_tensor_parallel_group(config.tp_size)
            make_parallel(model, config.tp_size)

        # 2. Apply Pipeline Parallelism (split layers across PP stages)
        if config.pp_size > 1:
            from .pipeline_parallel import PipelineModel, PipelineConfig
            pp_config = PipelineConfig(
                num_stages=config.pp_size,
                num_micro_batches=config.num_micro_batches,
                schedule=config.pp_schedule,
            )
            model = PipelineModel(model, pp_config)

        # 3. Apply DP wrapping (across DP replicas only)
        if config.dp_size > 1:
            if config.use_ddp:
                # DDP path — replicate full model on every rank, all-reduce
                # gradients. Required for torchao FP8 because Float8Linear
                # cannot handle FSDP DTensor weights.
                from torch.nn.parallel import DistributedDataParallel as DDP
                # Move model to the local CUDA device before DDP wrap
                if torch.cuda.is_available():
                    local_rank = dist.get_rank() % torch.cuda.device_count()
                    model = model.to(f"cuda:{local_rank}")
                model = DDP(
                    model,
                    device_ids=[dist.get_rank() % torch.cuda.device_count()] if torch.cuda.is_available() else None,
                    process_group=self._dp_group,
                    find_unused_parameters=False,
                    gradient_as_bucket_view=True,
                )
            else:
                # FSDP path (default) — shard params + grads + optimizer state
                # across DP replicas. Lower memory but incompatible with FP8.
                from .data_parallel import wrap_model_fsdp, ShardingMode, PrecisionMode
                precision = PrecisionMode.BF16 if config.precision == "bf16" else PrecisionMode.FP32
                model = wrap_model_fsdp(
                    model,
                    sharding_mode=ShardingMode.FULL_SHARD,
                    precision=precision,
                    gradient_checkpointing=config.gradient_checkpointing,
                    process_group=self._dp_group,
                )

        # 4. Gradient checkpointing (if no PP — PP handles its own checkpointing)
        if config.gradient_checkpointing and config.pp_size == 1:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

        self.model = model

    @staticmethod
    def _build_dp_groups(config: ClusterConfig):
        """
        Build DP process groups for FSDP when using TP+DP.

        For TP=2, DP=2 on 4 GPUs: DP groups are [0,2] and [1,3].
        Each DP group contains ranks with the same TP position.
        """
        rank = dist.get_rank()
        my_group = None
        for tp_pos in range(config.tp_size):
            # Ranks with same TP position form a DP group
            ranks = [
                dp * config.tp_size + tp_pos
                for dp in range(config.dp_size)
            ]
            group = dist.new_group(ranks)
            if rank in ranks:
                my_group = group
        return my_group

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train_step(self, batch: Dict[str, torch.Tensor], loss_fn=None) -> torch.Tensor:
        """
        Single training step with 3D parallelism.

        Handles micro-batching for pipeline parallelism automatically.
        """
        if self.config.pp_size > 1:
            # Pipeline handles micro-batching internally
            return self.model(batch, loss_fn=loss_fn)

        # Standard forward/backward (TP + DP only)
        outputs = self.model(batch["input_ids"])
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        if loss_fn is not None:
            return loss_fn(logits, batch["labels"])
        return logits

    @staticmethod
    def estimate_config(
        num_params: int,
        num_gpus: int,
        gpu_memory_gb: int = 80,
        batch_per_gpu: int = 64,
        seq_len: int = 2048,
    ) -> ClusterConfig:
        """
        Auto-estimate a ClusterConfig for a given model + cluster.

        Heuristic:
        - TP = min(8, gpus_needed_for_model)  (NVLink within node)
        - PP = layers / max_layers_per_stage   (pipeline if model is deep)
        - DP = remaining GPUs                   (data parallel)

        Args:
            num_params: Total model parameters
            num_gpus: Total GPUs available
            gpu_memory_gb: Per-GPU memory in GB
            batch_per_gpu: Desired micro-batch size
            seq_len: Sequence length
        """
        param_gb = num_params * 18 / 1e9  # params + optimizer in mixed precision

        # TP: how many GPUs needed to fit params + activations
        tp_size = 1
        while param_gb / tp_size > gpu_memory_gb * 0.6 and tp_size < 8:
            tp_size *= 2
        tp_size = min(tp_size, num_gpus)

        # PP: use if model still doesn't fit after TP
        remaining_after_tp = num_gpus // tp_size
        pp_size = 1
        if param_gb / tp_size > gpu_memory_gb * 0.4:
            pp_size = min(4, remaining_after_tp)

        # DP: remaining GPUs
        dp_size = num_gpus // (tp_size * pp_size)
        dp_size = max(dp_size, 1)

        # Adjust if total doesn't match
        total = tp_size * pp_size * dp_size
        if total != num_gpus:
            # Fall back to simpler config
            dp_size = num_gpus // tp_size
            pp_size = 1

        return ClusterConfig(
            tp_size=tp_size,
            pp_size=pp_size,
            dp_size=dp_size,
            micro_batch_size=batch_per_gpu,
        )

    @staticmethod
    def print_config_table(num_params: int, seq_len: int = 2048):
        """Print example configs for common cluster sizes."""
        print(f"\n{'='*70}")
        print(f"  Cluster configs for {num_params/1e9:.0f}B model (seq={seq_len})")
        print(f"{'='*70}")
        print(f"{'GPUs':>6} {'TP':>4} {'PP':>4} {'DP':>4} {'Batch':>8} {'Tokens/step':>14}")
        print(f"{'-'*70}")

        for num_gpus in [8, 16, 32, 64, 128, 256, 512, 1024]:
            cfg = ClusterModel.estimate_config(num_params, num_gpus)
            tokens = cfg.effective_batch_size * seq_len
            print(
                f"{num_gpus:>6} "
                f"{cfg.tp_size:>4} "
                f"{cfg.pp_size:>4} "
                f"{cfg.dp_size:>4} "
                f"{cfg.effective_batch_size:>8} "
                f"{tokens:>14,}"
            )
        print()
