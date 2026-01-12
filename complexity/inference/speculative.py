"""
Speculative Decoding for faster LLM inference.

Speculative decoding uses a smaller "draft" model to propose tokens,
then verifies them with the larger "target" model in parallel.

Key insight: Verification is faster than generation because
multiple tokens can be processed in a single forward pass.

Speedup: 2-4x for autoregressive generation

References:
- Fast Inference from Transformers via Speculative Decoding (https://arxiv.org/abs/2211.17192)
- Accelerating Large Language Model Decoding (https://arxiv.org/abs/2302.01318)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
import time


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    num_speculative_tokens: int = 5  # Tokens to draft before verification
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None


class SpeculativeDecoder:
    """
    Speculative Decoder for fast inference.

    Uses a smaller draft model to propose candidate tokens,
    then verifies them with the target model in a single forward pass.

    Usage:
        draft_model = load_model("draft_model")  # Small model
        target_model = load_model("target_model")  # Large model

        decoder = SpeculativeDecoder(
            target_model=target_model,
            draft_model=draft_model,
            config=SpeculativeConfig(num_speculative_tokens=5),
        )

        output = decoder.generate(input_ids, max_new_tokens=100)
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        config: Optional[SpeculativeConfig] = None,
        tokenizer: Optional[any] = None,
    ):
        """
        Args:
            target_model: Large target model
            draft_model: Small draft model
            config: Speculative decoding configuration
            tokenizer: Optional tokenizer for decoding
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.config = config or SpeculativeConfig()
        self.tokenizer = tokenizer

        # Put models in eval mode
        self.target_model.eval()
        self.draft_model.eval()

        # Statistics
        self.stats = {
            "total_tokens": 0,
            "accepted_tokens": 0,
            "draft_calls": 0,
            "target_calls": 0,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Generate tokens using speculative decoding.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            stop_tokens: Tokens that trigger early stopping

        Returns:
            Generated token IDs
        """
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p
        num_spec = self.config.num_speculative_tokens

        device = input_ids.device
        generated = input_ids.clone()
        num_generated = 0

        while num_generated < max_new_tokens:
            # Step 1: Draft tokens using small model
            draft_tokens, draft_probs = self._draft(
                generated, num_spec, temperature, top_k, top_p
            )

            # Step 2: Verify with target model
            accepted_tokens = self._verify(
                generated, draft_tokens, draft_probs, temperature, top_k, top_p
            )

            # Step 3: Append accepted tokens
            if len(accepted_tokens) > 0:
                generated = torch.cat([
                    generated,
                    torch.tensor([accepted_tokens], device=device)
                ], dim=1)
                num_generated += len(accepted_tokens)

                # Check for stop tokens
                if stop_tokens:
                    for token in accepted_tokens:
                        if token in stop_tokens:
                            return generated

            # If no tokens accepted, fall back to single target token
            if len(accepted_tokens) == 0:
                next_token = self._sample_target(generated, temperature, top_k, top_p)
                generated = torch.cat([
                    generated,
                    next_token.unsqueeze(0).unsqueeze(0)
                ], dim=1)
                num_generated += 1

        return generated

    def _draft(
        self,
        context: torch.Tensor,
        num_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Draft tokens using the small model.

        Returns:
            draft_tokens: List of drafted token IDs
            draft_probs: List of probability distributions
        """
        draft_tokens = []
        draft_probs = []
        current = context

        for _ in range(num_tokens):
            # Get draft model prediction
            outputs = self.draft_model(current)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            logits = logits[:, -1, :] / temperature

            # Apply top-k/top-p
            probs = self._apply_sampling(logits, top_k, top_p)
            draft_probs.append(probs)

            # Sample
            token = torch.multinomial(probs, num_samples=1).item()
            draft_tokens.append(token)

            # Append for next iteration
            current = torch.cat([
                current,
                torch.tensor([[token]], device=context.device)
            ], dim=1)

        self.stats["draft_calls"] += 1
        return draft_tokens, draft_probs

    def _verify(
        self,
        context: torch.Tensor,
        draft_tokens: List[int],
        draft_probs: List[torch.Tensor],
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> List[int]:
        """
        Verify draft tokens with target model.

        Uses rejection sampling to decide which tokens to accept.

        Returns:
            List of accepted tokens
        """
        if not draft_tokens:
            return []

        device = context.device

        # Prepare input with all draft tokens
        draft_tensor = torch.tensor([draft_tokens], device=device)
        full_input = torch.cat([context, draft_tensor], dim=1)

        # Get target model predictions for all positions
        outputs = self.target_model(full_input)
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs

        self.stats["target_calls"] += 1

        # Verify each position
        accepted = []
        context_len = context.size(1)

        for i, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
            # Target probability at this position
            target_logits = logits[:, context_len + i - 1, :] / temperature
            target_prob = self._apply_sampling(target_logits, top_k, top_p)

            # Rejection sampling
            # Accept with probability min(1, p_target / p_draft)
            p_draft = draft_prob[0, draft_token].item()
            p_target = target_prob[0, draft_token].item()

            accept_prob = min(1.0, p_target / (p_draft + 1e-8))

            if torch.rand(1).item() < accept_prob:
                accepted.append(draft_token)
                self.stats["accepted_tokens"] += 1
            else:
                # Rejection - sample from adjusted distribution
                # p_adjusted = max(0, p_target - p_draft)
                adjusted = F.relu(target_prob - draft_prob)
                adjusted = adjusted / (adjusted.sum() + 1e-8)

                if adjusted.sum() > 0:
                    new_token = torch.multinomial(adjusted, num_samples=1).item()
                    accepted.append(new_token)

                break  # Stop verification after first rejection

        self.stats["total_tokens"] += len(accepted)
        return accepted

    def _sample_target(
        self,
        context: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> torch.Tensor:
        """Sample single token from target model."""
        outputs = self.target_model(context)
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs

        logits = logits[:, -1, :] / temperature
        probs = self._apply_sampling(logits, top_k, top_p)

        self.stats["target_calls"] += 1
        self.stats["total_tokens"] += 1

        return torch.multinomial(probs, num_samples=1).squeeze()

    def _apply_sampling(
        self,
        logits: torch.Tensor,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> torch.Tensor:
        """Apply top-k and/or top-p sampling."""
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)

        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs[mask] = 0
            probs = sorted_probs.scatter(-1, sorted_indices, sorted_probs)
            probs = probs / probs.sum()

        return probs

    def get_statistics(self) -> dict:
        """Get decoding statistics."""
        acceptance_rate = (
            self.stats["accepted_tokens"] / max(self.stats["total_tokens"], 1)
        )
        return {
            **self.stats,
            "acceptance_rate": acceptance_rate,
            "speedup_estimate": 1 + acceptance_rate * (self.config.num_speculative_tokens - 1),
        }

    def reset_statistics(self):
        """Reset statistics."""
        self.stats = {
            "total_tokens": 0,
            "accepted_tokens": 0,
            "draft_calls": 0,
            "target_calls": 0,
        }


class SelfSpeculativeDecoder:
    """
    Self-speculative decoding using early exit.

    Instead of a separate draft model, uses early layers
    of the target model as the draft.

    Benefits:
    - No need for separate draft model
    - Lower memory overhead
    - Still achieves speedup

    Reference: Draft & Verify (https://arxiv.org/abs/2309.08168)
    """

    def __init__(
        self,
        model: nn.Module,
        exit_layer: int,
        config: Optional[SpeculativeConfig] = None,
    ):
        """
        Args:
            model: Target model
            exit_layer: Layer to use for early exit (draft)
            config: Configuration
        """
        self.model = model
        self.exit_layer = exit_layer
        self.config = config or SpeculativeConfig()

        # Find LM head for early exit
        self.lm_head = None
        if hasattr(model, 'lm_head'):
            self.lm_head = model.lm_head

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate with self-speculative decoding."""
        generated = input_ids.clone()
        num_spec = self.config.num_speculative_tokens

        for _ in range(0, max_new_tokens, num_spec + 1):
            # Draft using early exit
            draft_tokens = self._draft_early_exit(generated, num_spec, temperature)

            # Verify with full model
            verified = self._verify_full(generated, draft_tokens, temperature)

            # Append
            if verified:
                generated = torch.cat([
                    generated,
                    torch.tensor([verified], device=input_ids.device)
                ], dim=1)

        return generated

    def _draft_early_exit(
        self,
        context: torch.Tensor,
        num_tokens: int,
        temperature: float,
    ) -> List[int]:
        """Draft using early exit from target model."""
        # This requires model modification to support early exit
        # Simplified placeholder
        draft_tokens = []
        current = context

        for _ in range(num_tokens):
            # Run through early layers only
            hidden = self._forward_to_layer(current, self.exit_layer)

            if self.lm_head is not None:
                logits = self.lm_head(hidden[:, -1:, :])
            else:
                logits = hidden

            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            token = torch.multinomial(probs, num_samples=1).item()
            draft_tokens.append(token)

            current = torch.cat([
                current,
                torch.tensor([[token]], device=context.device)
            ], dim=1)

        return draft_tokens

    def _forward_to_layer(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Forward through model up to specified layer."""
        # Get embeddings
        if hasattr(self.model, 'embed_tokens'):
            hidden = self.model.embed_tokens(input_ids)
        elif hasattr(self.model, 'embedding'):
            hidden = self.model.embedding(input_ids)
        else:
            raise ValueError("Cannot find embedding layer")

        # Run through layers up to exit
        if hasattr(self.model, 'layers'):
            for i, layer in enumerate(self.model.layers):
                if i >= layer_idx:
                    break
                hidden = layer(hidden)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]

        return hidden

    def _verify_full(
        self,
        context: torch.Tensor,
        draft_tokens: List[int],
        temperature: float,
    ) -> List[int]:
        """Verify with full model."""
        # Simplified - just accept all for now
        # Full implementation would do proper verification
        return draft_tokens
