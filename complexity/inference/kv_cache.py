"""
KV-Cache implementations for efficient inference.

KV-Cache stores the key and value tensors from attention layers,
avoiding recomputation during autoregressive generation.

Implementations:
- KVCache: Simple per-sequence cache
- PagedKVCache: Page-based for dynamic memory (vLLM-style)
- SlidingWindowCache: Fixed-window cache (Mistral-style)
"""

import torch
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class CacheConfig:
    """Configuration for KV cache."""
    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int
    dtype: torch.dtype = torch.float16
    device: str = "cuda"


class KVCache:
    """
    Simple KV-Cache for autoregressive generation.

    Stores key-value tensors for all past positions.

    Usage:
        cache = KVCache(num_layers=32, num_heads=32, head_dim=128, max_seq_len=2048)

        for token in generation:
            # Model forward returns new k, v
            output, (new_k, new_v) = model(token, cache.get(layer_idx))
            cache.update(layer_idx, new_k, new_v)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        """
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads (or KV heads for GQA)
            head_dim: Dimension per head
            max_seq_len: Maximum sequence length
            batch_size: Batch size
            dtype: Data type for cache
            device: Device for cache
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device

        # Pre-allocate cache: [batch, num_heads, max_seq, head_dim]
        self.k_cache = torch.zeros(
            num_layers, batch_size, num_heads, max_seq_len, head_dim,
            dtype=dtype, device=device
        )
        self.v_cache = torch.zeros(
            num_layers, batch_size, num_heads, max_seq_len, head_dim,
            dtype=dtype, device=device
        )

        # Current sequence length per batch item
        self.seq_lens = torch.zeros(batch_size, dtype=torch.long, device=device)

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value and return full cache.

        Args:
            layer_idx: Layer index
            key: New keys [batch, num_heads, new_len, head_dim]
            value: New values [batch, num_heads, new_len, head_dim]

        Returns:
            Full keys and values including history
        """
        batch_size, num_heads, new_len, head_dim = key.shape

        # Get current positions
        start_pos = self.seq_lens[0].item()  # Assume all same length
        end_pos = start_pos + new_len

        if end_pos > self.max_seq_len:
            raise ValueError(f"Sequence length {end_pos} exceeds max {self.max_seq_len}")

        # Update cache
        self.k_cache[layer_idx, :batch_size, :, start_pos:end_pos, :] = key
        self.v_cache[layer_idx, :batch_size, :, start_pos:end_pos, :] = value

        # Return full cache up to current position
        return (
            self.k_cache[layer_idx, :batch_size, :, :end_pos, :],
            self.v_cache[layer_idx, :batch_size, :, :end_pos, :],
        )

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached keys and values for a layer."""
        seq_len = self.seq_lens[0].item()
        return (
            self.k_cache[layer_idx, :, :, :seq_len, :],
            self.v_cache[layer_idx, :, :, :seq_len, :],
        )

    def advance(self, num_tokens: int = 1):
        """Advance sequence position after generation."""
        self.seq_lens += num_tokens

    def reset(self):
        """Reset cache for new sequence."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_lens.zero_()

    @property
    def seq_len(self) -> int:
        """Current sequence length."""
        return self.seq_lens[0].item()

    def memory_usage(self) -> float:
        """Memory usage in MB."""
        element_size = self.k_cache.element_size()
        total_elements = self.k_cache.numel() + self.v_cache.numel()
        return total_elements * element_size / (1024 ** 2)


class PagedKVCache:
    """
    Paged KV-Cache for dynamic memory allocation (vLLM-style).

    Instead of pre-allocating max_seq_len for each sequence,
    allocates memory in fixed-size pages as needed.

    Benefits:
    - More memory efficient for variable-length sequences
    - Better memory sharing across requests
    - Enables continuous batching

    Reference: vLLM paper (https://arxiv.org/abs/2309.06180)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,  # Tokens per page
        num_pages: int = 1024,  # Total pages in pool
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        """
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of KV heads
            head_dim: Dimension per head
            page_size: Number of tokens per page
            num_pages: Total pages in the memory pool
            dtype: Data type
            device: Device
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.num_pages = num_pages
        self.dtype = dtype
        self.device = device

        # Page pool: [num_pages, num_layers, 2, num_heads, page_size, head_dim]
        # 2 for key and value
        self.page_pool = torch.zeros(
            num_pages, num_layers, 2, num_heads, page_size, head_dim,
            dtype=dtype, device=device
        )

        # Free pages (indices into page_pool)
        self.free_pages = list(range(num_pages))

        # Page tables: maps sequence_id -> list of page indices per layer
        self.page_tables: Dict[int, List[List[int]]] = {}

        # Current position in last page per sequence
        self.page_positions: Dict[int, int] = {}

    def allocate_sequence(self, seq_id: int):
        """Allocate initial pages for a new sequence."""
        if seq_id in self.page_tables:
            raise ValueError(f"Sequence {seq_id} already exists")

        # Allocate one page per layer
        pages_per_layer = []
        for _ in range(self.num_layers):
            if not self.free_pages:
                raise RuntimeError("Out of pages")
            page_idx = self.free_pages.pop()
            pages_per_layer.append([page_idx])

        self.page_tables[seq_id] = pages_per_layer
        self.page_positions[seq_id] = 0

    def free_sequence(self, seq_id: int):
        """Free all pages for a sequence."""
        if seq_id not in self.page_tables:
            return

        for layer_pages in self.page_tables[seq_id]:
            self.free_pages.extend(layer_pages)

        del self.page_tables[seq_id]
        del self.page_positions[seq_id]

    def update(
        self,
        seq_id: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache and return full KV for attention.

        Args:
            seq_id: Sequence identifier
            layer_idx: Layer index
            key: New key [1, num_heads, 1, head_dim]
            value: New value [1, num_heads, 1, head_dim]

        Returns:
            Full key and value tensors for this sequence
        """
        pages = self.page_tables[seq_id][layer_idx]
        pos = self.page_positions[seq_id]

        # Current page and position within page
        page_idx = pos // self.page_size
        pos_in_page = pos % self.page_size

        # Allocate new page if needed
        if page_idx >= len(pages):
            if not self.free_pages:
                raise RuntimeError("Out of pages")
            new_page = self.free_pages.pop()
            pages.append(new_page)

        # Store in page
        current_page = pages[page_idx]
        self.page_pool[current_page, layer_idx, 0, :, pos_in_page, :] = key.squeeze(0).squeeze(1)
        self.page_pool[current_page, layer_idx, 1, :, pos_in_page, :] = value.squeeze(0).squeeze(1)

        # Gather all KV from pages
        total_len = pos + 1
        all_k = torch.zeros(1, self.num_heads, total_len, self.head_dim, device=self.device, dtype=self.dtype)
        all_v = torch.zeros(1, self.num_heads, total_len, self.head_dim, device=self.device, dtype=self.dtype)

        for p_idx, page_id in enumerate(pages):
            start = p_idx * self.page_size
            end = min(start + self.page_size, total_len)
            length = end - start

            all_k[0, :, start:end, :] = self.page_pool[page_id, layer_idx, 0, :, :length, :]
            all_v[0, :, start:end, :] = self.page_pool[page_id, layer_idx, 1, :, :length, :]

        return all_k, all_v

    def advance(self, seq_id: int, num_tokens: int = 1):
        """Advance sequence position."""
        self.page_positions[seq_id] += num_tokens

    def get_seq_len(self, seq_id: int) -> int:
        """Get current sequence length."""
        return self.page_positions.get(seq_id, 0)

    def memory_usage(self) -> Dict[str, float]:
        """Memory usage statistics."""
        used_pages = self.num_pages - len(self.free_pages)
        page_bytes = self.page_pool[0].numel() * self.page_pool.element_size()

        return {
            "total_mb": self.page_pool.numel() * self.page_pool.element_size() / (1024 ** 2),
            "used_mb": used_pages * page_bytes / (1024 ** 2),
            "free_pages": len(self.free_pages),
            "used_pages": used_pages,
        }


class SlidingWindowCache(KVCache):
    """
    Sliding Window KV-Cache (Mistral-style).

    Only keeps the most recent `window_size` tokens.
    Older tokens are discarded, enabling unlimited context
    with fixed memory.

    Used by:
    - Mistral (4096 window)
    - Streaming LLM applications
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        window_size: int = 4096,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        """
        Args:
            num_layers: Number of layers
            num_heads: Number of KV heads
            head_dim: Dimension per head
            window_size: Sliding window size
            batch_size: Batch size
            dtype: Data type
            device: Device
        """
        # Initialize with window_size as max_seq_len
        super().__init__(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=window_size,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        self.window_size = window_size
        self.total_tokens = 0  # Total tokens seen (can exceed window)

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update with sliding window behavior.

        When window is full, shifts contents left.
        """
        batch_size, num_heads, new_len, head_dim = key.shape
        current_len = min(self.seq_lens[0].item(), self.window_size)

        # If adding new tokens would exceed window, shift
        if current_len + new_len > self.window_size:
            shift = current_len + new_len - self.window_size

            # Shift cache contents left
            self.k_cache[:, :, :, :-shift, :] = self.k_cache[:, :, :, shift:, :].clone()
            self.v_cache[:, :, :, :-shift, :] = self.v_cache[:, :, :, shift:, :].clone()

            # Update position
            self.seq_lens -= shift
            current_len = self.seq_lens[0].item()

        # Now add new tokens
        start_pos = current_len
        end_pos = start_pos + new_len

        self.k_cache[layer_idx, :batch_size, :, start_pos:end_pos, :] = key
        self.v_cache[layer_idx, :batch_size, :, start_pos:end_pos, :] = value

        self.total_tokens += new_len

        return (
            self.k_cache[layer_idx, :batch_size, :, :end_pos, :],
            self.v_cache[layer_idx, :batch_size, :, :end_pos, :],
        )

    def get_attention_mask(self) -> torch.Tensor:
        """
        Get attention mask accounting for sliding window.

        Returns causal mask limited to window size.
        """
        seq_len = min(self.seq_lens[0].item(), self.window_size)
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=self.device),
            diagonal=1
        )
        return mask


def create_cache(
    config: CacheConfig,
    cache_type: str = "standard",
    **kwargs,
) -> KVCache:
    """
    Create KV cache based on type.

    Args:
        config: Cache configuration
        cache_type: "standard", "paged", or "sliding_window"
        **kwargs: Additional arguments for specific cache types

    Returns:
        KV cache instance
    """
    if cache_type == "standard":
        return KVCache(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            dtype=config.dtype,
            device=config.device,
            **kwargs,
        )
    elif cache_type == "paged":
        return PagedKVCache(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            dtype=config.dtype,
            device=config.device,
            **kwargs,
        )
    elif cache_type == "sliding_window":
        return SlidingWindowCache(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            window_size=kwargs.get("window_size", config.max_seq_len),
            dtype=config.dtype,
            device=config.device,
        )
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
