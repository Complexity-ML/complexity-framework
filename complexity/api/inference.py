"""
External text generation and historical compatibility names.

    from complexity.api import create_external_backend, ExternalGenerationConfig
    backend = create_external_backend("vllm", base_url="http://localhost:8000",
                                      model="your-served-model")
    # Requires an already running, compatible serving endpoint:
    # response = backend.complete("Hello", ExternalGenerationConfig(max_tokens=100))

Generate/GenerationConfig remain importable for compatibility. Native Generate
methods consistently direct callers to the external serving interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Union

import torch

# Historical native-generation types remain importable.
from complexity.inference.engine import (
    DecodingStrategy,
    InferenceConfig,
    InferenceEngine,
    create_engine,
)
from complexity.inference.engine import (
    GenerationConfig as _GenerationConfig,
)
from complexity.inference.external import ExternalGenerationConfig, create_external_backend


@dataclass
class GenerationConfig:
    """
    Config génération - defaults sensibles, tout overridable via kwargs.

    Usage:
        config = GenerationConfig()  # Defaults
        config = GenerationConfig(temperature=0.7, top_p=0.9, top_k=50)
        config = GenerationConfig(strategy="greedy")
        config = GenerationConfig(strategy="beam", num_beams=4)
    """

    # Length
    max_tokens: int = 100
    min_tokens: int = 0

    # Sampling
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

    # Strategy
    strategy: str = "sampling"  # greedy, sampling, beam, top_k, top_p

    # Beam search
    num_beams: int = 1
    length_penalty: float = 1.0

    # Répétition
    repetition_penalty: float = 1.0
    no_repeat_ngram: int = 0

    # Stop
    stop_tokens: List[int] = field(default_factory=list)
    stop_sequences: List[str] = field(default_factory=list)

    # Output
    return_logits: bool = False

    def to_internal(self) -> _GenerationConfig:
        """Convertit vers le format interne."""
        strategy_map = {
            "greedy": DecodingStrategy.GREEDY,
            "sampling": DecodingStrategy.SAMPLING,
            "beam": DecodingStrategy.BEAM_SEARCH,
            "top_k": DecodingStrategy.TOP_K,
            "top_p": DecodingStrategy.TOP_P,
        }

        return _GenerationConfig(
            max_new_tokens=self.max_tokens,
            min_new_tokens=self.min_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            strategy=strategy_map.get(self.strategy, DecodingStrategy.SAMPLING),
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram,
            eos_token_id=self.stop_tokens[0] if self.stop_tokens else None,
            return_logits=self.return_logits,
        )


class Generate:
    """
    Historical compatibility namespace for disabled native generation.

    Calls fail before tokenization or configuration mutation. For generation
    use create_external_backend against an existing vLLM or SGLang server.
    """

    @classmethod
    def text(
        cls, model, prompt: str, tokenizer=None, config: GenerationConfig = None, **kwargs
    ) -> str:
        """Compatibility entry point; generation uses an external serving endpoint."""
        raise RuntimeError(
            "Generate is disabled for native text generation: use "
            "complexity.inference.create_external_backend with a vLLM or SGLang endpoint."
        )

    @classmethod
    def tokens(
        cls, model, input_ids: torch.Tensor, config: GenerationConfig = None, **kwargs
    ) -> torch.Tensor:
        """Compatibility entry point; generation uses an external serving endpoint."""
        raise RuntimeError(
            "Generate is disabled for native text generation: use "
            "complexity.inference.create_external_backend with a vLLM or SGLang endpoint."
        )

    @classmethod
    def stream(
        cls,
        model,
        prompt: Union[str, torch.Tensor],
        tokenizer=None,
        config: GenerationConfig = None,
        **kwargs,
    ) -> Iterator[str]:
        """Compatibility entry point; generation uses an external serving endpoint."""
        raise RuntimeError(
            "Generate is disabled for native text generation: use "
            "complexity.inference.create_external_backend with a vLLM or SGLang endpoint."
        )

    @classmethod
    def engine(
        cls,
        model,
        cache_type: str = "standard",
        use_speculative: bool = False,
        draft_model=None,
        **kwargs,
    ) -> InferenceEngine:
        """Compatibility entry point; generation uses an external serving endpoint."""
        raise RuntimeError(
            "Generate is disabled for native text generation: use "
            "complexity.inference.create_external_backend with a vLLM or SGLang endpoint."
        )

    @classmethod
    def chat(
        cls,
        model,
        messages: List[Dict[str, str]],
        tokenizer=None,
        config: GenerationConfig = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Compatibility entry point; generation uses an external serving endpoint."""
        raise RuntimeError(
            "Generate is disabled for native text generation: use "
            "complexity.inference.create_external_backend with a vLLM or SGLang endpoint."
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Generate",
    "GenerationConfig",
    "InferenceEngine",
    "InferenceConfig",
    "DecodingStrategy",
    "create_engine",
    "create_external_backend",
    "ExternalGenerationConfig",
]
