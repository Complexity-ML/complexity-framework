"""Static-shape CUDA Graph buckets for TR-Hash inference."""

from __future__ import annotations

from typing import Iterable, Tuple

from .config import GraphBucket


class GraphBucketPlanner:
    """Choose the smallest static bucket that contains an input shape."""

    def __init__(self, buckets: Iterable[GraphBucket]):
        self.buckets: Tuple[GraphBucket, ...] = tuple(
            sorted(
                set(buckets),
                key=lambda bucket: (
                    bucket.token_capacity,
                    bucket.batch_size,
                    bucket.sequence_length,
                ),
            )
        )
        if not self.buckets:
            raise ValueError("at least one CUDA Graph bucket is required")

    def select(self, batch_size: int, sequence_length: int) -> GraphBucket:
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("input batch and sequence dimensions must be positive")
        candidates = [
            bucket
            for bucket in self.buckets
            if bucket.batch_size >= batch_size and bucket.sequence_length >= sequence_length
        ]
        if not candidates:
            raise ValueError(
                f"no CUDA Graph bucket can contain input shape ({batch_size}, {sequence_length})"
            )
        return min(
            candidates,
            key=lambda bucket: (
                bucket.token_capacity,
                bucket.batch_size,
                bucket.sequence_length,
            ),
        )
