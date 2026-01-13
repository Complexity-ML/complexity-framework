"""
Model Builder - constructs complete models from configuration.

This is the main entry point for creating models in the framework.
"""

import os
import json
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path

from ..config import ModelConfig, get_preset
from ..core.registry import NORMALIZATION_REGISTRY, MODEL_REGISTRY, register_model
from .block import TransformerBlock


@register_model("complexity")
@register_model("decoder")
@register_model("causal_lm")
class ComplexityModel(nn.Module):
    """
    Complete Transformer model built from configuration.

    This model supports any architecture defined by ModelConfig:
    - Llama-style (GQA, SwiGLU, RMSNorm)
    - Mistral-style (sliding window)
    - GPT-style (MHA, GELU, LayerNorm)
    - Complexity custom (Token-Routed MoE)

    Usage:
        from complexity.config import ModelConfig
        from complexity.models import ComplexityModel

        # From config
        config = ModelConfig(hidden_size=768, num_hidden_layers=12)
        model = ComplexityModel(config)

        # From preset
        model = ComplexityModel.from_preset("llama-7b")

        # Forward pass
        logits = model(input_ids)

        # Generation
        output_ids = model.generate(input_ids, max_new_tokens=100)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Final normalization
        self.norm = NORMALIZATION_REGISTRY.build(
            config.norm_type,
            config.hidden_size,
            eps=config.norm_eps,
        )

        # Output projection (LM head)
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        if self.lm_head is not None:
            return self.lm_head
        return self.embed_tokens  # Tied weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_hidden_states: bool = False,
        velocity_states: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: Optional attention mask
            past_key_values: Optional list of KV caches per layer
            use_cache: Whether to return updated KV caches
            return_hidden_states: Whether to return all hidden states
            velocity_states: Optional list of velocity states per layer (for INL Dynamics)

        Returns:
            Dictionary with:
                - logits: [batch, seq_len, vocab_size]
                - past_key_values: Optional list of KV caches
                - hidden_states: Optional list of hidden states
                - velocity_states: Optional list of velocity states (if INL Dynamics enabled)
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Store hidden states if requested
        all_hidden_states = [hidden_states] if return_hidden_states else None

        # Initialize KV cache list
        new_past_key_values = [] if use_cache else None

        # Initialize velocity states list (for INL Dynamics)
        new_velocity_states = [] if self.config.use_inl_dynamics else None

        # Process through layers
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            velocity_state = velocity_states[i] if velocity_states is not None else None

            hidden_states, new_kv, new_velocity = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
                token_ids=input_ids,  # For MoE routing
                velocity_state=velocity_state,
            )

            if use_cache:
                new_past_key_values.append(new_kv)

            if new_velocity_states is not None:
                new_velocity_states.append(new_velocity)

            if return_hidden_states:
                all_hidden_states.append(hidden_states)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Compute logits
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied weights
            logits = torch.matmul(hidden_states, self.embed_tokens.weight.T)

        return {
            "logits": logits,
            "past_key_values": new_past_key_values,
            "hidden_states": all_hidden_states,
            "last_hidden_state": hidden_states,
            "velocity_states": new_velocity_states,
        }

    def set_tokenizer(self, tokenizer):
        """Set tokenizer for text-based generation."""
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        velocity_states: Optional[List[torch.Tensor]] = None,
        return_text: bool = False,
    ) -> Union[torch.Tensor, str]:
        """
        Generate text autoregressively with velocity tracking.

        Args:
            input_ids: [batch, seq_len] initial token IDs (or use text=)
            text: Input text (requires tokenizer to be set)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = neutral)
            top_k: Top-k sampling (0 = disabled)
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample (False = greedy)
            eos_token_id: Stop token ID
            velocity_states: Optional initial velocity states (for INL Dynamics)
            return_text: Return decoded text instead of IDs (requires tokenizer)

        Returns:
            output_ids: [batch, seq_len + new_tokens] or decoded text string
        """
        # Handle text input
        if text is not None:
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                raise ValueError("Tokenizer not set. Use model.set_tokenizer(tokenizer) first.")
            input_ids = torch.tensor([self.tokenizer.encode(text, add_bos=True)], device=next(self.parameters()).device)
            if eos_token_id is None:
                eos_token_id = self.tokenizer.eos_token_id

        if input_ids is None:
            raise ValueError("Either input_ids or text must be provided.")
        self.eval()
        device = input_ids.device

        # Use KV cache for efficient generation
        past_key_values = None

        for _ in range(max_new_tokens):
            # Get model output
            if past_key_values is None:
                # First pass: process full sequence
                outputs = self.forward(input_ids, use_cache=True, velocity_states=velocity_states)
            else:
                # Subsequent passes: only process last token
                # For INL Dynamics: extract only the last position's velocity
                if velocity_states is not None:
                    velocity_states_last = [v[:, -1:, :] for v in velocity_states]
                else:
                    velocity_states_last = None
                outputs = self.forward(
                    input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    velocity_states=velocity_states_last,
                )

            past_key_values = outputs["past_key_values"]
            # Update velocity states (concatenate new velocity to existing)
            if outputs["velocity_states"] is not None:
                if velocity_states is None:
                    velocity_states = outputs["velocity_states"]
                else:
                    velocity_states = [
                        torch.cat([v_old, v_new], dim=1)
                        for v_old, v_new in zip(velocity_states, outputs["velocity_states"])
                    ]
            logits = outputs["logits"][:, -1, :]  # [batch, vocab]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Sampling
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")

                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        # Return text if requested
        if return_text:
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                raise ValueError("Tokenizer not set. Use model.set_tokenizer(tokenizer) first.")
            return self.tokenizer.decode(input_ids[0].tolist())

        return input_ids

    @classmethod
    def from_preset(cls, name: str) -> "ComplexityModel":
        """Create model from preset configuration."""
        config = get_preset(name)
        return cls(config)

    @classmethod
    def from_config(cls, config_path: str) -> "ComplexityModel":
        """Create model from config file."""
        config = ModelConfig.load(config_path)
        return cls(config)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, save_directory: Union[str, Path], safe_serialization: bool = True):
        """
        Save model and config to directory (HuggingFace-compatible format).

        Args:
            save_directory: Path to save the model
            safe_serialization: If True, save using safetensors format (requires safetensors)
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = save_directory / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save model weights
        if safe_serialization:
            try:
                from safetensors.torch import save_file
                weights_path = save_directory / "model.safetensors"
                save_file(self.state_dict(), str(weights_path))
            except ImportError:
                # Fall back to PyTorch format
                weights_path = save_directory / "pytorch_model.bin"
                torch.save(self.state_dict(), weights_path)
        else:
            weights_path = save_directory / "pytorch_model.bin"
            torch.save(self.state_dict(), weights_path)

        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Union[str, Path],
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "ComplexityModel":
        """
        Load model from pretrained directory.

        Args:
            pretrained_model_path: Path to saved model directory
            device: Device to load model on (default: cpu)
            dtype: Data type for model (default: float32)

        Returns:
            Loaded ComplexityModel instance
        """
        pretrained_model_path = Path(pretrained_model_path)

        # Load config
        config_path = pretrained_model_path / "config.json"
        if not config_path.exists():
            raise ValueError(f"Config not found at {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = ModelConfig.from_dict(config_dict)

        # Create model
        model = cls(config)

        # Load weights
        safetensors_path = pretrained_model_path / "model.safetensors"
        pytorch_path = pretrained_model_path / "pytorch_model.bin"

        if safetensors_path.exists():
            try:
                from safetensors.torch import load_file
                state_dict = load_file(str(safetensors_path))
            except ImportError:
                raise ImportError(
                    "safetensors is required to load .safetensors files. "
                    "Install with: pip install safetensors"
                )
        elif pytorch_path.exists():
            state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=True)
        else:
            raise ValueError(
                f"No model weights found. Expected {safetensors_path} or {pytorch_path}"
            )

        model.load_state_dict(state_dict)

        # Move to device/dtype if specified
        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype)

        return model

    def __repr__(self) -> str:
        params = self.num_parameters() / 1e6
        return f"ComplexityModel({params:.1f}M params, {self.config.num_hidden_layers} layers)"
