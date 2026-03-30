"""
Inference Engine for high-performance text generation.

Provides a unified interface for:
- Single request generation
- Batch generation
- Streaming generation
- Various decoding strategies

Integrates KV caching, speculative decoding, and continuous batching.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Union, Iterator, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

from .kv_cache import KVCache, PagedKVCache, SlidingWindowCache
from .speculative import SpeculativeDecoder
from .batching import ContinuousBatcher, BatchConfig, Request, RequestStatus


class DecodingStrategy(Enum):
    """Decoding strategies for text generation."""
    GREEDY = "greedy"
    SAMPLING = "sampling"
    BEAM_SEARCH = "beam_search"
    TOP_K = "top_k"
    TOP_P = "top_p"
    CONTRASTIVE = "contrastive"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    # Length
    max_new_tokens: int = 100
    min_new_tokens: int = 0

    # Decoding strategy
    strategy: DecodingStrategy = DecodingStrategy.SAMPLING

    # Sampling parameters
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9

    # Beam search
    num_beams: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = False

    # Repetition control
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    # Stop conditions
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    stop_sequences: Optional[List[List[int]]] = None

    # Output control
    return_logits: bool = False
    return_hidden_states: bool = False

    # Contrastive search
    penalty_alpha: float = 0.6  # For contrastive search


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    # Cache settings
    use_kv_cache: bool = True
    cache_type: str = "standard"  # standard, paged, sliding
    max_cache_length: int = 2048
    page_size: int = 16  # For paged cache

    # Batching
    max_batch_size: int = 32
    dynamic_batching: bool = True
    timeout_seconds: float = 60.0

    # Speculative decoding
    use_speculative: bool = False
    draft_model: Optional[nn.Module] = None
    num_speculative_tokens: int = 4

    # Precision
    dtype: torch.dtype = torch.float16

    # Device
    device: Optional[torch.device] = None


class InferenceEngine:
    """
    High-performance inference engine.

    Supports multiple generation modes:
    - Single generation
    - Batch generation
    - Streaming generation
    - Continuous batching for serving

    Example:
        engine = InferenceEngine(model, config)

        # Single generation
        output = engine.generate(input_ids, GenerationConfig(max_new_tokens=50))

        # Streaming
        for tokens in engine.generate_stream(input_ids, config):
            print(tokenizer.decode(tokens))

        # Batch generation
        outputs = engine.generate_batch([input1, input2, input3], config)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[InferenceConfig] = None,
    ):
        """
        Args:
            model: Language model for generation
            config: Inference configuration
        """
        self.model = model
        self.config = config or InferenceConfig()

        # Set device
        if self.config.device is None:
            self.config.device = next(model.parameters()).device

        self.model.eval()

        # Initialize cache
        self._init_cache()

        # Initialize speculative decoder if enabled
        self.speculative_decoder = None
        if self.config.use_speculative and self.config.draft_model is not None:
            self.speculative_decoder = SpeculativeDecoder(
                target_model=model,
                draft_model=self.config.draft_model,
                k=self.config.num_speculative_tokens,
            )

        # Initialize batcher for serving
        self.batcher = None

    def _init_cache(self):
        """Initialize KV cache based on config."""
        self.cache = None
        if not self.config.use_kv_cache:
            return

        # Get model config for cache dimensions
        model_config = getattr(self.model, 'config', None)
        if model_config is None:
            return

        num_layers = getattr(model_config, 'num_hidden_layers', 12)
        num_heads = getattr(model_config, 'num_attention_heads', 12)
        head_dim = getattr(model_config, 'hidden_size', 768) // num_heads

        if self.config.cache_type == "paged":
            self.cache = PagedKVCache(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                page_size=self.config.page_size,
                max_pages=self.config.max_cache_length // self.config.page_size,
                dtype=self.config.dtype,
                device=self.config.device,
            )
        elif self.config.cache_type == "sliding":
            self.cache = SlidingWindowCache(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=self.config.max_cache_length,
                dtype=self.config.dtype,
                device=self.config.device,
            )
        else:
            self.cache = KVCache(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                max_length=self.config.max_cache_length,
                dtype=self.config.dtype,
                device=self.config.device,
            )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate tokens from input.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            generation_config: Generation parameters
            attention_mask: Attention mask

        Returns:
            Dictionary with generated tokens and optional outputs
        """
        config = generation_config or GenerationConfig()

        # Use speculative decoding if available
        if self.speculative_decoder is not None:
            output_ids = self.speculative_decoder.generate(
                input_ids=input_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                eos_token_id=config.eos_token_id,
            )
            return {"sequences": output_ids}

        # Use beam search
        if config.num_beams > 1:
            return self._generate_beam_search(input_ids, config, attention_mask)

        # Use sampling/greedy
        return self._generate_sampling(input_ids, config, attention_mask)

    def _generate_sampling(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate with sampling or greedy decoding."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize cache if available
        if self.cache is not None:
            self.cache.reset()

        # Track generated tokens
        generated = input_ids.clone()
        all_logits = [] if config.return_logits else None

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Generate tokens
        for _ in range(config.max_new_tokens):
            # Get model input (use cache if available)
            if self.cache is not None and generated.size(1) > seq_len:
                model_input = generated[:, -1:]
                position_ids = torch.tensor([[generated.size(1) - 1]], device=device)
            else:
                model_input = generated
                position_ids = None

            # Forward pass
            outputs = self.model(
                model_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=self.cache is not None,
            )

            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            # Get logits for last position
            next_logits = logits[:, -1, :] / config.temperature

            if all_logits is not None:
                all_logits.append(next_logits.clone())

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(
                    next_logits, generated, config.repetition_penalty
                )

            # Apply no_repeat_ngram
            if config.no_repeat_ngram_size > 0:
                next_logits = self._apply_no_repeat_ngram(
                    next_logits, generated, config.no_repeat_ngram_size
                )

            # Sample next token
            next_token = self._sample_token(next_logits, config)

            # Append to generated
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            # Update attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
            ], dim=-1)

            # Check stop conditions
            if self._should_stop(generated, next_token, config):
                break

        result = {"sequences": generated}
        if all_logits is not None:
            result["logits"] = torch.stack(all_logits, dim=1)

        return result

    def _generate_beam_search(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate with beam search."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        num_beams = config.num_beams

        # Expand input for beams
        input_ids = input_ids.unsqueeze(1).expand(-1, num_beams, -1)
        input_ids = input_ids.reshape(batch_size * num_beams, -1)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(-1, num_beams, -1)
            attention_mask = attention_mask.reshape(batch_size * num_beams, -1)

        # Initialize beam scores
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = float('-inf')  # Only first beam active initially
        beam_scores = beam_scores.reshape(-1)

        generated = input_ids.clone()
        done = [False] * batch_size

        for step in range(config.max_new_tokens):
            # Forward pass
            outputs = self.model(generated, attention_mask=attention_mask)

            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            next_logits = logits[:, -1, :] / config.temperature
            vocab_size = next_logits.size(-1)

            # Calculate scores
            next_scores = torch.log_softmax(next_logits, dim=-1)
            next_scores = next_scores + beam_scores.unsqueeze(-1)

            # Reshape for beam selection
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)

            # Select top beams
            next_scores, next_tokens = torch.topk(
                next_scores, 2 * num_beams, dim=-1, largest=True, sorted=True
            )

            # Extract beam and token indices
            next_beam_indices = next_tokens // vocab_size
            next_token_indices = next_tokens % vocab_size

            # Build next generation
            new_generated = []
            new_beam_scores = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # Keep best beam
                    new_generated.append(generated[batch_idx * num_beams])
                    new_beam_scores.append(beam_scores[batch_idx * num_beams])
                    continue

                beams_added = 0
                for beam_rank in range(2 * num_beams):
                    if beams_added >= num_beams:
                        break

                    beam_idx = next_beam_indices[batch_idx, beam_rank]
                    token_idx = next_token_indices[batch_idx, beam_rank]
                    score = next_scores[batch_idx, beam_rank]

                    # Get previous sequence
                    prev_seq = generated[batch_idx * num_beams + beam_idx]
                    new_seq = torch.cat([prev_seq, token_idx.unsqueeze(0)])

                    new_generated.append(new_seq)
                    new_beam_scores.append(score)
                    beams_added += 1

                    # Check EOS
                    if config.eos_token_id is not None and token_idx == config.eos_token_id:
                        if config.early_stopping:
                            done[batch_idx] = True

            generated = torch.stack(new_generated[:batch_size * num_beams])
            beam_scores = torch.tensor(new_beam_scores[:batch_size * num_beams], device=device)

            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(batch_size * num_beams, 1, device=device)
                ], dim=-1)

            if all(done):
                break

        # Select best beam for each batch
        beam_scores = beam_scores.view(batch_size, num_beams)
        generated = generated.view(batch_size, num_beams, -1)

        # Apply length penalty
        if config.length_penalty != 1.0:
            lengths = (generated != config.pad_token_id).sum(dim=-1).float()
            beam_scores = beam_scores / (lengths ** config.length_penalty)

        best_beams = beam_scores.argmax(dim=-1)
        best_sequences = generated[torch.arange(batch_size), best_beams]

        return {"sequences": best_sequences}

    def _sample_token(
        self,
        logits: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """Sample next token from logits."""
        if config.strategy == DecodingStrategy.GREEDY:
            return logits.argmax(dim=-1)

        # Apply top-k
        if config.top_k is not None and config.top_k > 0:
            v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float('-inf')

        # Apply top-p (nucleus sampling)
        if config.top_p is not None and config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens above threshold
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Sample
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        for i in range(logits.size(0)):
            for token_id in set(generated[i].tolist()):
                if logits[i, token_id] < 0:
                    logits[i, token_id] *= penalty
                else:
                    logits[i, token_id] /= penalty
        return logits

    def _apply_no_repeat_ngram(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        ngram_size: int,
    ) -> torch.Tensor:
        """Prevent repeated n-grams."""
        batch_size = generated.size(0)

        for batch_idx in range(batch_size):
            gen_tokens = generated[batch_idx].tolist()
            if len(gen_tokens) < ngram_size:
                continue

            # Get current n-1 gram
            current_ngram = tuple(gen_tokens[-(ngram_size - 1):])

            # Find all previous occurrences and their next tokens
            banned_tokens = set()
            for i in range(len(gen_tokens) - ngram_size + 1):
                if tuple(gen_tokens[i:i + ngram_size - 1]) == current_ngram:
                    banned_tokens.add(gen_tokens[i + ngram_size - 1])

            # Ban tokens
            for token_id in banned_tokens:
                logits[batch_idx, token_id] = float('-inf')

        return logits

    def _should_stop(
        self,
        generated: torch.Tensor,
        next_token: torch.Tensor,
        config: GenerationConfig,
    ) -> bool:
        """Check if generation should stop."""
        # Check EOS token
        if config.eos_token_id is not None:
            if (next_token == config.eos_token_id).all():
                return True

        # Check stop sequences
        if config.stop_sequences is not None:
            for stop_seq in config.stop_sequences:
                seq_len = len(stop_seq)
                if generated.size(1) >= seq_len:
                    if (generated[0, -seq_len:].tolist() == stop_seq):
                        return True

        return False

    def generate_stream(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Iterator[torch.Tensor]:
        """
        Stream generated tokens one at a time.

        Yields:
            Generated token IDs one at a time
        """
        config = generation_config or GenerationConfig()
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize cache
        if self.cache is not None:
            self.cache.reset()

        generated = input_ids.clone()

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        for _ in range(config.max_new_tokens):
            # Get model input
            if self.cache is not None and generated.size(1) > seq_len:
                model_input = generated[:, -1:]
            else:
                model_input = generated

            # Forward pass
            outputs = self.model(model_input, attention_mask=attention_mask)

            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            next_logits = logits[:, -1, :] / config.temperature

            # Sample
            next_token = self._sample_token(next_logits, config)

            # Yield token
            yield next_token

            # Update state
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
            ], dim=-1)

            # Check stop
            if self._should_stop(generated, next_token, config):
                break

    def generate_batch(
        self,
        input_ids_list: List[torch.Tensor],
        generation_config: Optional[GenerationConfig] = None,
    ) -> List[torch.Tensor]:
        """
        Generate for a list of inputs.

        Args:
            input_ids_list: List of input tensors
            generation_config: Generation parameters

        Returns:
            List of generated sequences
        """
        config = generation_config or GenerationConfig()
        results = []

        for input_ids in input_ids_list:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            output = self.generate(input_ids, config)
            results.append(output["sequences"])

        return results

    def start_serving(self):
        """Start continuous batching server."""
        batch_config = BatchConfig(
            max_batch_size=self.config.max_batch_size,
            max_sequence_length=self.config.max_cache_length,
            dynamic_batching=self.config.dynamic_batching,
            timeout_seconds=self.config.timeout_seconds,
        )
        self.batcher = ContinuousBatcher(self.model, batch_config)

    def add_request(
        self,
        request_id: str,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        """Add request to serving queue."""
        if self.batcher is None:
            self.start_serving()

        config = generation_config or GenerationConfig()

        request = Request(
            request_id=request_id,
            input_ids=input_ids,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            stop_tokens=[config.eos_token_id] if config.eos_token_id else None,
        )

        return self.batcher.add_request(request)

    def step_serving(self) -> List[Dict[str, Any]]:
        """Process one step of continuous batching."""
        if self.batcher is None:
            return []

        completed = self.batcher.step()

        return [
            {
                "request_id": req.request_id,
                "output_ids": req.output_ids,
                "status": req.status.value,
                "latency": req.end_time - req.start_time,
            }
            for req in completed
        ]

    def get_serving_stats(self) -> Dict[str, Any]:
        """Get serving statistics."""
        if self.batcher is None:
            return {}
        return self.batcher.get_statistics()


def create_engine(
    model: nn.Module,
    cache_type: str = "standard",
    use_speculative: bool = False,
    draft_model: Optional[nn.Module] = None,
    dtype: torch.dtype = torch.float16,
) -> InferenceEngine:
    """
    Factory function to create inference engine.

    Args:
        model: Target model
        cache_type: Cache type (standard, paged, sliding)
        use_speculative: Enable speculative decoding
        draft_model: Draft model for speculative decoding
        dtype: Data type for inference

    Returns:
        Configured InferenceEngine
    """
    config = InferenceConfig(
        cache_type=cache_type,
        use_speculative=use_speculative,
        draft_model=draft_model,
        dtype=dtype,
    )

    return InferenceEngine(model, config)
