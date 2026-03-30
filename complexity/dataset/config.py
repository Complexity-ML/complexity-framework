"""
Dataset Configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class DataConfig:
    """Config dataset - defaults sensibles, tout overridable."""
    seq_length: int = 2048
    batch_size: int = 32
    text_column: str = "text"
    shuffle: bool = True
    num_workers: int = 0
    pack_sequences: bool = True
    # Extra
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for k, v in self.extra.items():
            setattr(self, k, v)
