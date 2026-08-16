"""
Inference API - Génération facile.
==================================

Usage:
    from complexity.api import Generate, GenerationConfig

    # Production generation must use vLLM/SGLang OpenAI-compatible serving.

    # Streaming
    for token in Generate.stream(model, "Hello", max_tokens=100):
        print(token, end="")

    # The PyTorch framework layer does not own a native generate loop.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Iterator, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# Import depuis inference module
from complexity.inference.engine import (
    InferenceEngine,
    InferenceConfig,
    GenerationConfig as _GenerationConfig,
    DecodingStrategy,
    create_engine,
)


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
    API de génération flexible.

    Usage:
        # Deprecated compatibility wrapper. Use vLLM/SGLang instead.
    """

    @classmethod
    def text(
        cls,
        model,
        prompt: str,
        tokenizer=None,
        config: GenerationConfig = None,
        **kwargs
    ) -> str:
        """
        Génère du texte depuis un prompt.

        Args:
            model: Model API ou nn.Module
            prompt: Texte de prompt
            tokenizer: Tokenizer (auto-détecté si possible)
            config: GenerationConfig ou None
            **kwargs: Override (max_tokens, temperature, ...)

        Returns:
            Texte généré
        """
        # Get tokenizer
        tok = tokenizer or getattr(model, '_tokenizer', None)
        if tok is None:
            raise ValueError("Tokenizer required for text generation")

        # Build config
        if config:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        else:
            config = GenerationConfig(**kwargs)

        # Encode
        input_ids = tok.encode(prompt)
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # Get device
        device = getattr(model, 'device', None) or getattr(model, '_device', 'cpu')
        input_ids = input_ids.to(device)

        # Generate
        output_ids = cls.tokens(model, input_ids, config=config)

        # Decode new tokens only
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return tok.decode(new_tokens.tolist())

    @classmethod
    def tokens(
        cls,
        model,
        input_ids: torch.Tensor,
        config: GenerationConfig = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Génère des tokens depuis input_ids.

        Args:
            model: Model
            input_ids: Input token IDs [batch, seq]
            config: GenerationConfig
            **kwargs: Override

        Returns:
            Generated token IDs
        """
        raise RuntimeError(
            "Generate.tokens is disabled: generation must run through vLLM or "
            "SGLang, not Complexity's PyTorch layer."
        )

    @classmethod
    def stream(
        cls,
        model,
        prompt: Union[str, torch.Tensor],
        tokenizer=None,
        config: GenerationConfig = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream génération token par token.

        Args:
            model: Model
            prompt: Texte ou token IDs
            tokenizer: Tokenizer (requis pour texte)
            config: GenerationConfig
            **kwargs: Override

        Yields:
            Texte décodé pour chaque token
        """
        # Build config
        if config:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        else:
            config = GenerationConfig(**kwargs)

        # Get tokenizer
        tok = tokenizer or getattr(model, '_tokenizer', None)

        # Prepare input
        if isinstance(prompt, str):
            if tok is None:
                raise ValueError("Tokenizer required for text input")
            input_ids = tok.encode(prompt)
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids], dtype=torch.long)
        else:
            input_ids = prompt
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

        # Get device
        device = getattr(model, 'device', None) or getattr(model, '_device', 'cpu')
        input_ids = input_ids.to(device)

        # Create engine and stream
        engine = cls.engine(model)
        internal_config = config.to_internal()

        for token in engine.generate_stream(input_ids, internal_config):
            if tok is not None:
                yield tok.decode([token.item()])
            else:
                yield token

    @classmethod
    def engine(
        cls,
        model,
        cache_type: str = "standard",
        use_speculative: bool = False,
        draft_model=None,
        **kwargs
    ) -> InferenceEngine:
        """
        Crée un engine d'inférence.

        Args:
            model: Model
            cache_type: "standard", "paged", "sliding"
            use_speculative: Active speculative decoding
            draft_model: Draft model pour speculative
            **kwargs: Override config

        Returns:
            InferenceEngine configuré
        """
        raise RuntimeError(
            "Generate.engine is disabled for native text generation: use vLLM "
            "or SGLang OpenAI-compatible serving."
        )

    @classmethod
    def chat(
        cls,
        model,
        messages: List[Dict[str, str]],
        tokenizer=None,
        config: GenerationConfig = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Génère une réponse chat.

        Args:
            model: Model
            messages: Liste de messages [{"role": "user", "content": "..."}]
            tokenizer: Tokenizer
            config: GenerationConfig
            **kwargs: Override

        Returns:
            Message assistant {"role": "assistant", "content": "..."}
        """
        # Get tokenizer
        tok = tokenizer or getattr(model, '_tokenizer', None)
        if tok is None:
            raise ValueError("Tokenizer required for chat")

        # Build config
        if config:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        else:
            config = GenerationConfig(**kwargs)

        # Encode chat
        if hasattr(tok, 'encode_chat'):
            input_ids = tok.encode_chat(messages)
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids], dtype=torch.long)
        else:
            # Fallback: concat messages
            text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            text += "\nassistant: "
            input_ids = tok.encode(text)
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids], dtype=torch.long)

        # Get device
        device = getattr(model, 'device', None) or getattr(model, '_device', 'cpu')
        input_ids = input_ids.to(device)

        # Generate
        output_ids = cls.tokens(model, input_ids, config=config)

        # Decode
        new_tokens = output_ids[0, input_ids.shape[1]:]
        content = tok.decode(new_tokens.tolist())

        return {"role": "assistant", "content": content}


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
]
