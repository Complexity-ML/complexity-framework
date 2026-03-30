"""
Parallel training module for framework-complexity.

Supports:
- Tensor Parallelism (TP): Split layers across GPUs
- Pipeline Parallelism (PP): Split model by layers
- Data Parallelism (DP/FSDP): Replicate model, split data

Usage:
    # FSDP (recommended for most cases)
    from complexity.parallel import wrap_model_fsdp, ShardingMode

    model = wrap_model_fsdp(
        model,
        sharding_mode=ShardingMode.FULL_SHARD,
        precision=PrecisionMode.BF16,
    )

    # Tensor Parallel
    from complexity.parallel import ColumnParallelLinear, RowParallelLinear

    # Pipeline Parallel
    from complexity.parallel import PipelineModel, PipelineSchedule
"""

from .data_parallel import (
    wrap_model_fsdp,
    simple_ddp,
    ShardingMode,
    PrecisionMode,
    DataParallelConfig,
    GradientAccumulator,
    init_distributed,
    get_world_size,
    get_rank,
    is_main_process,
    barrier,
    cleanup,
)

from .tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
    all_reduce,
    reduce_scatter,
    all_gather,
    get_tensor_parallel_rank,
    get_tensor_parallel_world_size,
    make_parallel,
    initialize_tensor_parallel,
)

from .pipeline_parallel import (
    PipelineModel,
    PipelineStage,
    PipelineScheduler,
    PipelineConfig,
    PipelineSchedule,
    split_model_for_pipeline,
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
)

__all__ = [
    # Data Parallel
    "wrap_model_fsdp",
    "simple_ddp",
    "ShardingMode",
    "PrecisionMode",
    "DataParallelConfig",
    "GradientAccumulator",
    "init_distributed",
    "get_world_size",
    "get_rank",
    "is_main_process",
    "barrier",
    "cleanup",
    # Tensor Parallel
    "ColumnParallelLinear",
    "RowParallelLinear",
    "ParallelEmbedding",
    "all_reduce",
    "reduce_scatter",
    "all_gather",
    "get_tensor_parallel_rank",
    "get_tensor_parallel_world_size",
    "make_parallel",
    "initialize_tensor_parallel",
    # Pipeline Parallel
    "PipelineModel",
    "PipelineStage",
    "PipelineScheduler",
    "PipelineConfig",
    "PipelineSchedule",
    "split_model_for_pipeline",
    "get_pipeline_parallel_rank",
    "get_pipeline_parallel_world_size",
]
