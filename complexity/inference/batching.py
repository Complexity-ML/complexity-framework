"""
Continuous Batching for high-throughput LLM serving.

Continuous batching allows dynamic addition/removal of requests
during generation, maximizing GPU utilization.

Key concepts:
- Requests can join mid-batch
- Completed requests are removed immediately
- New requests fill freed slots

Reference: Orca (https://www.usenix.org/conference/osdi22/presentation/yu)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time
import threading


class RequestStatus(Enum):
    """Status of a generation request."""
    PENDING = "pending"      # Waiting to be processed
    RUNNING = "running"      # Currently generating
    COMPLETED = "completed"  # Generation finished
    FAILED = "failed"        # Error occurred


@dataclass
class Request:
    """A generation request."""
    request_id: str
    input_ids: torch.Tensor
    max_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stop_tokens: Optional[List[int]] = None

    # Output
    output_ids: List[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.PENDING

    # Timing
    arrival_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

    # Internal state
    position: int = 0  # Position in batch
    tokens_generated: int = 0


@dataclass
class BatchConfig:
    """Configuration for continuous batching."""
    max_batch_size: int = 32
    max_sequence_length: int = 2048
    prefill_chunk_size: int = 512  # Tokens to process in prefill
    dynamic_batching: bool = True  # Allow mid-generation additions
    timeout_seconds: float = 60.0  # Request timeout


class ContinuousBatcher:
    """
    Continuous Batcher for high-throughput serving.

    Dynamically batches requests, adding new ones and removing
    completed ones without waiting for the entire batch.

    Usage:
        batcher = ContinuousBatcher(model, config)

        # Add requests
        batcher.add_request(Request("req1", input_ids1))
        batcher.add_request(Request("req2", input_ids2))

        # Process in loop
        while batcher.has_pending():
            completed = batcher.step()
            for req in completed:
                print(f"Request {req.request_id} done")
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[BatchConfig] = None,
        kv_cache_factory: Optional[Callable] = None,
    ):
        """
        Args:
            model: Language model
            config: Batching configuration
            kv_cache_factory: Factory function to create KV caches
        """
        self.model = model
        self.config = config or BatchConfig()
        self.kv_cache_factory = kv_cache_factory

        self.model.eval()

        # Request queues
        self.pending_queue: deque = deque()
        self.running: Dict[str, Request] = {}
        self.completed: Dict[str, Request] = {}

        # Batch state
        self.batch_input_ids: Optional[torch.Tensor] = None
        self.batch_positions: Dict[int, str] = {}  # position -> request_id

        # Device
        self.device = next(model.parameters()).device

        # Lock for thread safety
        self.lock = threading.Lock()

    def add_request(self, request: Request) -> str:
        """
        Add a new request to the queue.

        Args:
            request: Generation request

        Returns:
            Request ID
        """
        request.arrival_time = time.time()
        request.input_ids = request.input_ids.to(self.device)

        with self.lock:
            self.pending_queue.append(request)

        return request.request_id

    def has_pending(self) -> bool:
        """Check if there are pending or running requests."""
        return len(self.pending_queue) > 0 or len(self.running) > 0

    @torch.no_grad()
    def step(self) -> List[Request]:
        """
        Execute one generation step.

        Returns:
            List of completed requests in this step
        """
        with self.lock:
            # Add pending requests to batch
            self._fill_batch()

            if not self.running:
                return []

            # Prepare batch
            batch_ids, attention_mask = self._prepare_batch()

            # Forward pass
            outputs = self.model(batch_ids, attention_mask=attention_mask)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            # Sample next tokens
            next_tokens = self._sample_tokens(logits)

            # Update requests and check completion
            completed = self._update_requests(next_tokens)

            return completed

    def _fill_batch(self):
        """Fill batch with pending requests."""
        while self.pending_queue and len(self.running) < self.config.max_batch_size:
            request = self.pending_queue.popleft()

            # Check timeout
            if time.time() - request.arrival_time > self.config.timeout_seconds:
                request.status = RequestStatus.FAILED
                self.completed[request.request_id] = request
                continue

            # Find free position
            position = self._find_free_position()
            request.position = position
            request.status = RequestStatus.RUNNING
            request.start_time = time.time()

            self.running[request.request_id] = request
            self.batch_positions[position] = request.request_id

    def _find_free_position(self) -> int:
        """Find free position in batch."""
        for i in range(self.config.max_batch_size):
            if i not in self.batch_positions:
                return i
        raise RuntimeError("No free position in batch")

    def _prepare_batch(self) -> tuple:
        """Prepare batch tensors for forward pass."""
        batch_size = len(self.running)

        # Find max length
        max_len = max(
            req.input_ids.size(-1) + req.tokens_generated
            for req in self.running.values()
        )
        max_len = min(max_len, self.config.max_sequence_length)

        # Prepare tensors
        batch_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)

        # Fill batch
        for i, req in enumerate(self.running.values()):
            seq_len = req.input_ids.size(-1) + len(req.output_ids)
            seq_len = min(seq_len, max_len)

            # Original input
            input_len = min(req.input_ids.size(-1), max_len)
            batch_ids[i, :input_len] = req.input_ids[0, :input_len]

            # Generated tokens
            gen_len = min(len(req.output_ids), max_len - input_len)
            if gen_len > 0:
                batch_ids[i, input_len:input_len + gen_len] = torch.tensor(
                    req.output_ids[:gen_len], device=self.device
                )

            attention_mask[i, :seq_len] = 1

        return batch_ids, attention_mask

    def _sample_tokens(self, logits: torch.Tensor) -> List[int]:
        """Sample next tokens for each request."""
        next_tokens = []

        for i, (req_id, req) in enumerate(self.running.items()):
            # Get logits for last position
            seq_len = req.input_ids.size(-1) + len(req.output_ids)
            token_logits = logits[i, seq_len - 1, :] / req.temperature

            # Top-k
            if req.top_k is not None:
                v, _ = torch.topk(token_logits, min(req.top_k, token_logits.size(-1)))
                token_logits[token_logits < v[-1]] = float('-inf')

            # Sample
            probs = torch.softmax(token_logits, dim=-1)

            # Top-p
            if req.top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > req.top_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                sorted_probs[mask] = 0
                probs = sorted_probs.scatter(0, sorted_idx, sorted_probs)
                probs = probs / probs.sum()

            next_token = torch.multinomial(probs, num_samples=1).item()
            next_tokens.append(next_token)

        return next_tokens

    def _update_requests(self, next_tokens: List[int]) -> List[Request]:
        """Update requests with new tokens and check completion."""
        completed = []
        to_remove = []

        for (req_id, req), token in zip(list(self.running.items()), next_tokens):
            req.output_ids.append(token)
            req.tokens_generated += 1

            # Check completion conditions
            should_stop = False

            # Max tokens
            if req.tokens_generated >= req.max_tokens:
                should_stop = True

            # Stop tokens
            if req.stop_tokens and token in req.stop_tokens:
                should_stop = True

            # Max sequence length
            total_len = req.input_ids.size(-1) + req.tokens_generated
            if total_len >= self.config.max_sequence_length:
                should_stop = True

            if should_stop:
                req.status = RequestStatus.COMPLETED
                req.end_time = time.time()
                completed.append(req)
                to_remove.append(req_id)
                self.completed[req_id] = req

        # Remove completed
        for req_id in to_remove:
            req = self.running.pop(req_id)
            del self.batch_positions[req.position]

        return completed

    def get_request(self, request_id: str) -> Optional[Request]:
        """Get request by ID."""
        if request_id in self.running:
            return self.running[request_id]
        if request_id in self.completed:
            return self.completed[request_id]
        for req in self.pending_queue:
            if req.request_id == request_id:
                return req
        return None

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a request."""
        with self.lock:
            # Check running
            if request_id in self.running:
                req = self.running.pop(request_id)
                del self.batch_positions[req.position]
                req.status = RequestStatus.FAILED
                self.completed[request_id] = req
                return True

            # Check pending
            for i, req in enumerate(self.pending_queue):
                if req.request_id == request_id:
                    self.pending_queue.remove(req)
                    req.status = RequestStatus.FAILED
                    self.completed[request_id] = req
                    return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get serving statistics."""
        completed_times = [
            req.end_time - req.start_time
            for req in self.completed.values()
            if req.status == RequestStatus.COMPLETED
        ]

        return {
            "pending": len(self.pending_queue),
            "running": len(self.running),
            "completed": len(self.completed),
            "avg_latency": sum(completed_times) / max(len(completed_times), 1),
            "throughput": len(self.completed) / max(sum(completed_times), 1) if completed_times else 0,
        }


class AsyncBatcher(ContinuousBatcher):
    """
    Async version of continuous batcher.

    Runs generation in a background thread for non-blocking operation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._running = False
        self._thread = None
        self._callbacks: Dict[str, Callable] = {}

    def start(self):
        """Start the async generation loop."""
        self._running = True
        self._thread = threading.Thread(target=self._generation_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the async generation loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def add_request_async(
        self,
        request: Request,
        callback: Optional[Callable[[Request], None]] = None,
    ) -> str:
        """Add request with optional completion callback."""
        req_id = self.add_request(request)
        if callback:
            self._callbacks[req_id] = callback
        return req_id

    def _generation_loop(self):
        """Background generation loop."""
        while self._running:
            if self.has_pending():
                completed = self.step()

                # Trigger callbacks
                for req in completed:
                    if req.request_id in self._callbacks:
                        try:
                            self._callbacks[req.request_id](req)
                        except Exception as e:
                            print(f"Callback error: {e}")
                        del self._callbacks[req.request_id]
            else:
                time.sleep(0.001)  # Small sleep when idle
