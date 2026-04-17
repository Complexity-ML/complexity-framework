"""
Model Builder - constructs complete models from configuration.

This is the main entry point for creating models in the framework.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
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
    - Complexity custom (Token-Routed MoE + Mu-Guidance)

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

        # Learnable mu_init: gives layer 0 a mu_prev instead of None
        self._has_mu = getattr(config, 'use_mu_guidance', False) or getattr(config, 'use_mu_projection', False)
        if self._has_mu and not getattr(config, 'disable_mu_guidance', False):
            self.mu_init = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # Gradient checkpointing (disabled by default)
        self._gradient_checkpointing = False

        # Initialize weights (GPT-style: residual projections scaled by 1/√(2N))
        self.apply(self._init_weights)
        self._init_residual_scaling()

    def _init_weights(self, module: nn.Module):
        """
        GPT-style weight initialization.

        Standard layers: normal_(std=0.02)
        Residual projections (o_proj, down_proj_w): scaled by 1/√(2*num_layers)
        to prevent residual stream from growing with depth.
        Ref: Radford et al. (GPT-2), "Language Models are Unsupervised Multitask Learners"
        """
        std = self.config.initializer_range  # 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _init_residual_scaling(self):
        """
        GPT-style residual scaling: init output projections with std=0.02/√(2N).

        Targets: o_proj (attention output) and down_proj/down_proj_w (MLP output).
        Re-initializes with normal_ then scales, overriding any prior init
        (e.g. kaiming_uniform_ in TokenRoutedMLP).
        Called once after self.apply(_init_weights).
        """
        n = self.config.num_hidden_layers
        residual_std = self.config.initializer_range / (2 * n) ** 0.5
        for layer in self.layers:
            # Attention output projection
            attn = layer.self_attn
            if hasattr(attn, 'o_proj') and isinstance(attn.o_proj, nn.Linear):
                nn.init.normal_(attn.o_proj.weight, mean=0.0, std=residual_std)
            # MLP down projection (nn.Linear or nn.Parameter) — routed experts and dense SwiGLU
            mlp = layer.mlp
            if hasattr(mlp, 'down_proj'):
                if isinstance(mlp.down_proj, nn.Linear):
                    nn.init.normal_(mlp.down_proj.weight, mean=0.0, std=residual_std)
                elif isinstance(mlp.down_proj, nn.Parameter):
                    nn.init.normal_(mlp.down_proj, mean=0.0, std=residual_std)
            if hasattr(mlp, 'down_proj_w') and isinstance(mlp.down_proj_w, nn.Parameter):
                nn.init.normal_(mlp.down_proj_w, mean=0.0, std=residual_std)
            # Shared expert down projection (TokenRoutedMLP with shared=True)
            if hasattr(mlp, 'shared_down') and isinstance(mlp.shared_down, nn.Linear):
                nn.init.normal_(mlp.shared_down.weight, mean=0.0, std=residual_std)

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

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
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: Optional attention mask
            past_key_values: Optional list of KV caches per layer
            use_cache: Whether to return updated KV caches
            return_hidden_states: Whether to return all hidden states

        Returns:
            Dictionary with:
                - logits: [batch, seq_len, vocab_size]
                - past_key_values: Optional list of KV caches
                - hidden_states: Optional list of hidden states
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Store hidden states if requested
        all_hidden_states = [hidden_states] if return_hidden_states else None

        # Initialize KV cache list
        new_past_key_values = [] if use_cache else None

        # Process through layers (mu flows from layer to layer)
        # mu_init: learnable starting mu so layer 0 gets guidance too
        mu_prev = None
        if self._has_mu and not getattr(self.config, 'disable_mu_guidance', False):
            mu_prev = self.mu_init.expand(batch_size, seq_len, -1)
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if self._gradient_checkpointing and self.training:
                hidden_states, new_kv, _, mu_contextual = activation_checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    past_kv,
                    use_cache,
                    input_ids,
                    None,  # velocity_state (unused, kept for compat)
                    mu_prev,
                    None,  # sort_idx (computed internally by token_routed)
                    use_reentrant=False,
                )
            else:
                hidden_states, new_kv, _, mu_contextual = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    token_ids=input_ids,
                    mu_prev=mu_prev,
                )

            # mu from this layer guides next layer's attention — free (no clamp).
            if mu_contextual is not None and not getattr(self.config, 'disable_mu_guidance', False):
                mu_prev = mu_contextual

            if use_cache:
                new_past_key_values.append(new_kv)

            if return_hidden_states:
                all_hidden_states.append(hidden_states)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Skip logits computation when training with fused cross-entropy
        # (logits are computed inside fused_cross_entropy from last_hidden_state)
        if self.training:
            return {
                "logits": None,
                "past_key_values": new_past_key_values,
                "hidden_states": all_hidden_states,
                "last_hidden_state": hidden_states,
            }

        # Compute logits only during inference/generation
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.embed_tokens.weight)

        return {
            "logits": logits,
            "past_key_values": new_past_key_values,
            "hidden_states": all_hidden_states,
            "last_hidden_state": hidden_states,
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
        return_text: bool = False,
    ) -> Union[torch.Tensor, str]:
        """
        Generate text autoregressively.

        Args:
            input_ids: [batch, seq_len] initial token IDs (or use text=)
            text: Input text (requires tokenizer to be set)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = neutral)
            top_k: Top-k sampling (0 = disabled)
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample (False = greedy)
            eos_token_id: Stop token ID
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
                outputs = self.forward(input_ids, use_cache=True)
            else:
                outputs = self.forward(
                    input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs["past_key_values"]
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

    def quantize_all(self):
        """
        Quantize all I64 components to INT8 for inference.

        Converts:
        - I64Attention: QKV + mu + O projections -> fused INT8
        - I64SwiGLUMLP: gate+up+down -> fused INT8 with LUT SiLU
        - I64TokenRoutedMLP: expert weights -> INT8
        - I64RMSNorm: weights -> Q12 INT16

        Call this after training, before inference.
        """
        from complexity.core.attention.i64_attention import I64Attention
        from complexity.core.mlp.i64_mlp import I64SwiGLUMLP, I64TokenRoutedMLP
        from complexity.core.normalization.i64_norm import I64RMSNorm

        quantized_count = 0
        for module in self.modules():
            if isinstance(module, (I64Attention, I64SwiGLUMLP, I64TokenRoutedMLP)):
                module.quantize()
                quantized_count += 1
            elif isinstance(module, I64RMSNorm):
                module.quantize_weight()
                quantized_count += 1

        print(f"Quantized {quantized_count} I64 modules to INT8")
        return self

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

        # Save model weights — handles FSDP v2 (composable) DTensor params.
        # CRITICAL: full_tensor() is a collective op (all-gather), so it MUST
        # be called on every rank — even though only rank 0 writes the file.
        # Calling it from inside an "if is_main" block deadlocks the others.
        import torch.distributed as dist
        is_distributed = dist.is_initialized() and dist.get_world_size() > 1
        is_main = (not is_distributed) or dist.get_rank() == 0

        # Collective barrier at entry: if a caller forgot to invoke this on
        # every rank (e.g. wrapped it in `if rank == 0`), the missing ranks
        # will fail the barrier with a clear NCCL timeout instead of hanging
        # silently inside full_tensor() for 30 minutes.
        if is_distributed:
            try:
                dist.barrier()
            except Exception as e:
                print(f"[save_pretrained] barrier failed on rank {dist.get_rank()}: {e}", flush=True)
                raise
            if is_main:
                print(f"[save_pretrained] collecting weights across {dist.get_world_size()} ranks → {save_directory}", flush=True)

        raw_sd = self.state_dict()
        cpu_sd = {}
        n_gathered = 0
        n_local_only = 0
        first_error: Optional[Exception] = None
        for k, v in raw_sd.items():
            is_dtensor = hasattr(v, "full_tensor")
            if is_dtensor:
                # Collective: every rank participates in the all-gather.
                # We MUST NOT swallow errors silently here — falling back to
                # .to_local() would write rank 0's shard only, producing a
                # corrupted checkpoint with ~1/world_size of the weights.
                try:
                    v = v.full_tensor()
                    n_gathered += 1
                except Exception as e:
                    if first_error is None:
                        first_error = e
                    if is_main:
                        print(f"[save_pretrained] full_tensor() failed on '{k}': "
                              f"{type(e).__name__}: {e}", flush=True)
                    if hasattr(v, "to_local"):
                        v = v.to_local()
                        n_local_only += 1
            elif hasattr(v, "to_local"):
                # Non-sharded DTensor (replicated); to_local is safe.
                v = v.to_local()
            if is_main:
                cpu_sd[k] = v.detach().cpu().contiguous() if hasattr(v, "detach") else v

        if is_main and n_local_only > 0:
            print(
                f"[save_pretrained] WARNING: {n_local_only}/{len(raw_sd)} params "
                f"fell back to to_local() after full_tensor() failed "
                f"(first error: {type(first_error).__name__}: {first_error}). "
                f"The saved checkpoint is INCOMPLETE — it only contains rank 0's "
                f"shard for those params. Investigate before using this file.",
                flush=True,
            )

        if not is_main:
            return

        if safe_serialization:
            try:
                from safetensors.torch import save_file
                weights_path = save_directory / "model.safetensors"
                save_file(cpu_sd, str(weights_path))
            except ImportError:
                weights_path = save_directory / "pytorch_model.bin"
                torch.save(cpu_sd, weights_path)
        else:
            weights_path = save_directory / "pytorch_model.bin"
            torch.save(cpu_sd, weights_path)

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
