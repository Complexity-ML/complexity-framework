"""Standalone matched language model used by the review artifact."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .attention import ContextualWRVAttention, GroupedQueryAttention, RMSNorm


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_read_heads: int
    num_write_heads: int
    intermediate_size: int
    lexical_object_rank: int = 16
    micro_num_experts: int = 4
    micro_expert_width: int = 32
    max_sequence_length: int = 2048
    attention_type: str = "wrv"
    lexical_write_residual: bool = False
    use_read_write_norm: bool = True
    attention_dropout: float = 0.0
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.attention_type not in {"gqa", "wrv"}:
            raise ValueError("attention_type must be 'gqa' or 'wrv'")
        if self.hidden_size % self.num_read_heads:
            raise ValueError("hidden_size must be divisible by num_read_heads")
        if self.num_read_heads % self.num_write_heads:
            raise ValueError("num_read_heads must be divisible by num_write_heads")

    @classmethod
    def paper(cls, *, attention_type: str, **overrides: Any) -> "ModelConfig":
        values: dict[str, Any] = dict(
            vocab_size=200_019,
            hidden_size=384,
            num_layers=10,
            num_read_heads=8,
            num_write_heads=2,
            intermediate_size=1118,
            lexical_object_rank=16,
            micro_num_experts=4,
            micro_expert_width=32,
            max_sequence_length=2048,
            attention_type=attention_type,
            lexical_write_residual=False,
            use_read_write_norm=True,
        )
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LexicalObjectMicroExpertMLP(nn.Module):
    """Shared SwiGLU plus lexical low-rank and deterministic micro-expert residuals."""

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        rank = config.lexical_object_rank
        micro_width = config.micro_num_experts * config.micro_expert_width
        self.hidden_size = hidden
        self.object_rank = rank
        self.micro_num_experts = config.micro_num_experts
        self.micro_expert_width = config.micro_expert_width
        self.shared_gate = nn.Linear(hidden, intermediate, bias=False)
        self.shared_up = nn.Linear(hidden, intermediate, bias=False)
        self.shared_down = nn.Linear(intermediate, hidden, bias=False)
        self.object_up = nn.Linear(hidden, rank, bias=False)
        self.object_down = nn.Linear(rank, hidden, bias=False)
        self.object_output_gate = nn.Parameter(torch.tensor(0.1))
        self.micro_gate = nn.Linear(hidden, micro_width, bias=False)
        self.micro_up = nn.Linear(hidden, micro_width, bias=False)
        self.micro_down = nn.Parameter(
            torch.empty(config.micro_num_experts, config.micro_expert_width, hidden)
        )
        self.micro_output_gate = nn.Parameter(torch.tensor(0.1))
        mapping = (torch.arange(config.vocab_size, dtype=torch.long) + layer_index) % config.micro_num_experts
        self.register_buffer("token_to_micro_expert", mapping, persistent=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        lexical_scale: torch.Tensor,
    ) -> torch.Tensor:
        shared = self.shared_down(F.silu(self.shared_gate(hidden_states)) * self.shared_up(hidden_states))
        object_hidden = F.silu(self.object_up(hidden_states)) * (1.0 + lexical_scale)
        object_residual = self.object_down(object_hidden)
        shape = hidden_states.shape[:-1] + (self.micro_num_experts, self.micro_expert_width)
        micro_hidden = (
            F.silu(self.micro_gate(hidden_states)) * self.micro_up(hidden_states)
        ).view(shape)
        all_outputs = torch.einsum("...ew,ewd->...ed", micro_hidden, self.micro_down)
        expert_ids = self.token_to_micro_expert[token_ids]
        selected = torch.gather(
            all_outputs,
            -2,
            expert_ids[..., None, None].expand(*expert_ids.shape, 1, self.hidden_size),
        ).squeeze(-2)
        return shared + self.object_output_gate * object_residual + self.micro_output_gate * selected


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.norm_eps)
        attention_cls = GroupedQueryAttention if config.attention_type == "gqa" else ContextualWRVAttention
        self.attention = attention_cls(config)
        self.mlp = LexicalObjectMicroExpertMLP(config, layer_index)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        lexical_scale: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attention_output, cache = self.attention(
            self.input_norm(hidden_states),
            token_ids=token_ids,
            lexical_scale=lexical_scale,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(
            self.post_attention_norm(hidden_states), token_ids, lexical_scale
        )
        return hidden_states, cache


class TinyLanguageModel(nn.Module):
    """Self-contained causal LM matching the paper's realized architecture."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lexical_scale = nn.Embedding(config.vocab_size, config.lexical_object_rank)
        self.blocks = nn.ModuleList(
            DecoderBlock(config, layer_index) for layer_index in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.apply(self._initialize)
        nn.init.zeros_(self.lexical_scale.weight)
        residual_std = config.initializer_range / math_sqrt(2 * config.num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.output_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.shared_down.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.object_down.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.micro_down, mean=0.0, std=residual_std)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        use_cache: bool = False,
        return_logits: bool = True,
    ) -> dict[str, Any]:
        hidden_states = self.token_embedding(input_ids)
        lexical_scale = self.lexical_scale(input_ids)
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = [] if use_cache else None
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            hidden_states, cache = block(hidden_states, input_ids, lexical_scale, past, use_cache)
            if caches is not None:
                caches.append(cache)
        hidden_states = self.final_norm(hidden_states)
        logits = F.linear(hidden_states, self.token_embedding.weight) if return_logits else None
        return {"logits": logits, "last_hidden_state": hidden_states, "past_key_values": caches}

    def loss(self, token_ids: torch.Tensor, *, chunk_tokens: int = 512) -> torch.Tensor:
        inputs, labels = token_ids[:, :-1], token_ids[:, 1:]
        hidden = self(inputs, return_logits=False)["last_hidden_state"].reshape(-1, self.config.hidden_size)
        labels = labels.reshape(-1)
        total = hidden.new_zeros((), dtype=torch.float32)
        count = 0
        for start in range(0, labels.numel(), chunk_tokens):
            end = min(start + chunk_tokens, labels.numel())
            logits = F.linear(hidden[start:end], self.token_embedding.weight)
            total = total + F.cross_entropy(logits.float(), labels[start:end], reduction="sum")
            count += end - start
        return total / count


def math_sqrt(value: int) -> float:
    return float(value) ** 0.5
