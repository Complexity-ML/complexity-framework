"""Standalone associative context mechanism.

The public contract is contextual read/write with fixed-size caller-owned state.
Its internal state representation is deliberately not part of the API.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedAssociativeContext(nn.Module):
    """Content-addressable context branch whose parameters can be shared."""

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        vocab_size: int | None = None,
        output_gate_init: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.value_proj = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.output_proj = nn.Linear(self.rank, self.hidden_size, bias=False)
        self.output_gate = nn.Parameter(torch.tensor(float(output_gate_init)))
        self.write_logit = nn.Parameter(torch.tensor(2.197225))
        self.decay_logit = nn.Parameter(torch.tensor(6.906755))
        if vocab_size is None:
            address_table = torch.empty(0, self.rank)
        else:
            address_table = self._compute_lexical_keys(
                torch.arange(vocab_size, dtype=torch.int64)
            )
        self.register_buffer("address_table", address_table, persistent=False)

    def _compute_lexical_keys(self, token_ids: torch.Tensor) -> torch.Tensor:
        dimensions = torch.arange(
            1, self.rank + 1, device=token_ids.device, dtype=torch.float64
        )
        phases = (
            token_ids.to(torch.float64)[..., None]
            * torch.pi
            * torch.sqrt(dimensions)
        )
        return F.normalize(torch.sin(phases).to(torch.float32), dim=-1)

    def _lexical_keys(self, token_ids: torch.Tensor) -> torch.Tensor:
        address_table = self.address_table
        assert isinstance(address_table, torch.Tensor)
        if address_table.numel() == 0:
            return self._compute_lexical_keys(token_ids)
        return F.embedding(token_ids, address_table)

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        association_state = torch.zeros(
            batch_size, self.rank, self.rank, device=device, dtype=dtype
        )
        previous_address = torch.zeros(
            batch_size, self.rank, device=device, dtype=dtype
        )
        return association_state, previous_address

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        association_state, previous_address = state
        addresses = self._lexical_keys(token_ids).to(hidden_states.dtype)
        values = self.value_proj(hidden_states)
        outputs = []
        write_rate = torch.sigmoid(self.write_logit)
        decay = torch.sigmoid(self.decay_logit)
        for address_chunk, value_chunk in zip(
            addresses.split(256, dim=1), values.split(256, dim=1)
        ):
            chunk_size = address_chunk.shape[1]
            write_addresses = torch.cat(
                (previous_address[:, None], address_chunk[:, :-1]), dim=1
            )
            positions = torch.arange(
                chunk_size, device=hidden_states.device, dtype=torch.float32
            )
            initial_powers = decay.float() ** positions
            initial_reads = torch.bmm(address_chunk, association_state)
            initial_reads = initial_reads * initial_powers[None, :, None]

            similarities = torch.bmm(
                address_chunk, write_addresses.transpose(1, 2)
            )
            distance = positions[:, None] - positions[None, :] - 1
            causal_weights = torch.where(
                distance >= 0,
                decay.float() ** distance.clamp_min(0),
                torch.zeros_like(distance),
            )
            local_reads = torch.bmm(
                similarities * causal_weights[None].to(similarities.dtype),
                value_chunk,
            )
            reads = initial_reads + write_rate * local_reads
            outputs.append(self.output_proj(reads))

            final_powers = decay.float() ** torch.arange(
                chunk_size - 1,
                -1,
                -1,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            weighted_values = value_chunk * final_powers[None, :, None]
            association_state = (
                decay**chunk_size * association_state
                + write_rate
                * torch.bmm(write_addresses.transpose(1, 2), weighted_values)
            )
            previous_address = address_chunk[:, -1]
        residual = self.output_gate * torch.cat(outputs, dim=1)
        return residual, (association_state, previous_address)
