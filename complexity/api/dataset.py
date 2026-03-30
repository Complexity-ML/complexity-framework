"""
Dataset API - Re-exports from complexity.dataset.

For full dataset functionality, use:
    from complexity.dataset import Dataset, DataConfig
"""

from complexity.dataset import (
    Dataset,
    DataConfig,
    StreamingDataset,
    DataPipeline,
)

__all__ = [
    "Dataset",
    "DataConfig",
    "StreamingDataset",
    "DataPipeline",
]
