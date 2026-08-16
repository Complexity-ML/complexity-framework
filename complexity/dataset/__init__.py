"""
Complexity Dataset - Load, process, iterate over training data.

Usage:
    from complexity.dataset import Dataset, DataConfig

    # Load
    ds = Dataset.load("./train.jsonl")

    # Iterate
    for batch in ds.batches(batch_size=16):
        model(batch["input_ids"])

    # Pipeline
    from complexity.dataset import DataPipeline
    pipeline = DataPipeline(data_path, output_path, tokenizer)
    pipeline.run()
"""

from .config import DataConfig
from .dataset import Dataset, StreamingDataset
from .pipeline import DataPipeline

__all__ = [
    "DataConfig",
    "Dataset",
    "StreamingDataset",
    "DataPipeline",
]
