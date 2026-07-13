"""Shared associative attention with learned contextual addresses."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from complexity.core.context.associative import SharedAssociativeContext


class SharedAssociativeAttention(SharedAssociativeContext):
    """Standalone shared attention experiment with learned addresses."""

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        vocab_size: int | None = None,
        output_gate_init: float = 1.0,
        contextual_mix_init: float = 0.0,
    ):
        super().__init__(hidden_size, rank, vocab_size, output_gate_init)
        self.address_proj = nn.Linear(hidden_size, rank, bias=False)
        self.contextual_mix = nn.Parameter(torch.tensor(float(contextual_mix_init)))
        self.contextual_enabled = contextual_mix_init != 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not self.contextual_enabled:
            return super().forward(hidden_states, token_ids, state)
        reads, next_state = self._contextual_reads(hidden_states, token_ids, state)
        return self.output_gate * self.output_proj(reads), next_state

    def _contextual_reads(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        association_state, previous_address = state
        lexical = self._lexical_keys(token_ids).to(hidden_states.dtype)
        mix = torch.tanh(self.contextual_mix)
        addresses = F.normalize(
            lexical + mix * F.normalize(self.address_proj(hidden_states), dim=-1),
            dim=-1,
        )
        values = self.value_proj(hidden_states)
        outputs = []
        write_rate = torch.sigmoid(self.write_logit)
        decay = torch.sigmoid(self.decay_logit)
        for address_chunk, value_chunk in zip(
            addresses.split(256, dim=1), values.split(256, dim=1)
        ):
            chunk_size = address_chunk.shape[1]
            write_keys = torch.cat(
                (previous_address[:, None], address_chunk[:, :-1]), dim=1
            )
            positions = torch.arange(
                chunk_size, device=hidden_states.device, dtype=torch.float32
            )
            initial_powers = decay.float() ** positions
            initial_reads = torch.bmm(address_chunk, association_state)
            initial_reads = initial_reads * initial_powers[None, :, None]

            similarities = torch.bmm(
                address_chunk, write_keys.transpose(1, 2)
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
            outputs.append(reads)

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
                * torch.bmm(write_keys.transpose(1, 2), weighted_values)
            )
            previous_address = address_chunk[:, -1]
        return torch.cat(outputs, dim=1), (association_state, previous_address)


class SharedContextFusion(SharedAssociativeAttention):
    """Tokenwise shared fusion of retrieved context without a scalar output gate."""

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        fusion_size: int,
        vocab_size: int | None = None,
        contextual_mix_init: float = 0.1,
        fusion_scale: float = 0.1,
    ):
        super().__init__(
            hidden_size,
            rank,
            vocab_size=vocab_size,
            output_gate_init=1.0,
            contextual_mix_init=contextual_mix_init,
        )
        del self.output_gate
        del self.output_proj
        self.fusion_gate = nn.Linear(hidden_size, fusion_size, bias=False)
        self.fusion_value = nn.Linear(rank, fusion_size, bias=False)
        self.fusion_out = nn.Linear(fusion_size, hidden_size, bias=False)
        self.fusion_scale = float(fusion_scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        reads, next_state = self._contextual_reads(hidden_states, token_ids, state)
        fused = F.silu(self.fusion_gate(hidden_states)) * self.fusion_value(reads)
        return self.fusion_scale * self.fusion_out(fused), next_state


class StableDeltaAssociativeAttention(SharedAssociativeAttention):
    """Reference stable fast-weight attention using residual delta writes."""

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        vocab_size: int | None = None,
        contextual_mix_init: float = 0.1,
        residual_scale: float = 0.1,
        delta_chunk_size: int = 512,
    ):
        super().__init__(
            hidden_size,
            rank,
            vocab_size=vocab_size,
            output_gate_init=1.0,
            contextual_mix_init=contextual_mix_init,
        )
        del self.output_gate
        self.residual_scale = float(residual_scale)
        self.delta_chunk_size = int(delta_chunk_size)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del dtype
        return (
            torch.zeros(batch_size, self.rank, self.rank, device=device),
            torch.zeros(batch_size, self.rank, device=device),
        )

    def delta_scan(
        self,
        write_keys: torch.Tensor,
        values: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = state.float()
        write_keys = write_keys.float()
        values = values.float()
        write_rate = torch.sigmoid(self.write_logit.float())
        decay = torch.sigmoid(self.decay_logit.float())
        reads = []
        for position in range(write_keys.shape[1]):
            key = write_keys[:, position]
            value = values[:, position]
            prediction = torch.bmm(key[:, None], state).squeeze(1)
            reads.append(prediction)
            error = value - prediction
            state = decay * state + write_rate * torch.bmm(
                key[:, :, None], error[:, None, :]
            )
        return state, torch.stack(reads, dim=1)

    def vectorized_delta_scan(
        self,
        write_keys: torch.Tensor,
        values: torch.Tensor,
        state: torch.Tensor,
        queries: torch.Tensor | None = None,
        decay_override: torch.Tensor | None = None,
        write_rate_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact chunkwise delta recurrence via a causal triangular solve."""
        state = state.float()
        write_keys = write_keys.float()
        values = values.float()
        queries = write_keys if queries is None else queries.float()
        length = write_keys.shape[1]
        write_rate = (
            torch.sigmoid(self.write_logit.float())
            if write_rate_override is None
            else write_rate_override.float()
        )
        decay = (
            torch.sigmoid(self.decay_logit.float())
            if decay_override is None
            else decay_override.float()
        )
        batch_size = state.shape[0]
        write_rate = write_rate.expand(batch_size)
        decay = decay.expand(batch_size)
        positions = torch.arange(length, device=state.device, dtype=torch.float32)
        distance = positions[:, None] - positions[None, :] - 1
        causal_decay = torch.where(
            distance[None] >= 0,
            decay[:, None, None] ** distance.clamp_min(0)[None],
            torch.zeros_like(distance)[None],
        )

        key_gram = torch.bmm(write_keys, write_keys.transpose(1, 2))
        identity = torch.eye(length, device=state.device, dtype=torch.float32)
        system = (
            identity[None]
            + write_rate[:, None, None] * key_gram * causal_decay
        )
        initial_prediction = torch.bmm(write_keys, state)
        initial_powers = decay[:, None] ** positions[None]
        right_hand_side = write_rate[:, None, None] * (
            values - initial_powers[:, :, None] * initial_prediction
        )
        errors = torch.linalg.solve_triangular(
            system, right_hand_side, upper=False, unitriangular=True
        )

        initial_reads = initial_powers[:, :, None] * torch.bmm(queries, state)
        query_key = torch.bmm(queries, write_keys.transpose(1, 2))
        reads = initial_reads + torch.bmm(query_key * causal_decay, errors)

        reverse_positions = torch.arange(
            length - 1, -1, -1, device=state.device, dtype=torch.float32
        )
        final_powers = decay[:, None] ** reverse_positions[None]
        next_state = decay[:, None, None] ** length * state + torch.bmm(
            write_keys.transpose(1, 2), errors * final_powers[:, :, None]
        )
        return next_state, reads

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        association_state, previous_address = state
        lexical = self._lexical_keys(token_ids).float()
        contextual = F.normalize(self.address_proj(hidden_states).float(), dim=-1)
        addresses = F.normalize(
            lexical + torch.tanh(self.contextual_mix.float()) * contextual,
            dim=-1,
        )
        values = self.value_proj(hidden_states).float()
        outputs = []
        for address_chunk, value_chunk in zip(
            addresses.split(self.delta_chunk_size, dim=1),
            values.split(self.delta_chunk_size, dim=1),
        ):
            write_keys = torch.cat(
                (previous_address[:, None], address_chunk[:, :-1]), dim=1
            )
            association_state, reads = self.vectorized_delta_scan(
                write_keys,
                value_chunk,
                association_state,
                queries=address_chunk,
            )
            outputs.append(reads)
            previous_address = address_chunk[:, -1]

        reads = torch.cat(outputs, dim=1).to(self.output_proj.weight.dtype)
        residual = self.residual_scale * self.output_proj(reads)
        return residual, (association_state, previous_address)


class MultiOrderLexicalDeltaAttention(nn.Module):
    """Stable Delta memories addressed by deterministic lexical n-grams."""

    def __init__(
        self,
        hidden_size: int,
        head_rank: int = 32,
        orders: tuple[int, ...] = (1, 2, 4, 8),
        vocab_size: int | None = None,
        delta_chunk_size: int = 512,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        del vocab_size
        self.hidden_size = int(hidden_size)
        self.head_rank = int(head_rank)
        self.orders = tuple(int(order) for order in orders)
        self.max_order = max(self.orders)
        self.delta_chunk_size = int(delta_chunk_size)
        self.residual_scale = float(residual_scale)
        self.heads = nn.ModuleList()
        for _ in self.orders:
            head = StableDeltaAssociativeAttention(
                hidden_size,
                head_rank,
                vocab_size=None,
                contextual_mix_init=0.0,
                residual_scale=1.0,
                delta_chunk_size=delta_chunk_size,
            )
            del head.address_proj
            del head.contextual_mix
            self.heads.append(head)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del dtype
        matrices = torch.zeros(
            batch_size,
            len(self.orders),
            self.head_rank,
            self.head_rank,
            device=device,
        )
        previous = torch.zeros(
            batch_size,
            len(self.orders),
            self.head_rank,
            device=device,
        )
        history = torch.full(
            (batch_size, self.max_order - 1),
            -1,
            device=device,
            dtype=torch.int64,
        )
        return matrices, previous, history

    def _addresses(
        self,
        token_ids: torch.Tensor,
        history: torch.Tensor,
        head: StableDeltaAssociativeAttention,
        order: int,
    ) -> torch.Tensor:
        full_ids = torch.cat((history, token_ids), dim=1)
        end = self.max_order - 1 + torch.arange(
            token_ids.shape[1], device=token_ids.device
        )
        address = torch.zeros(
            *token_ids.shape,
            self.head_rank,
            device=token_ids.device,
            dtype=torch.float32,
        )
        for lag in range(order):
            ids = full_ids[:, end - lag]
            valid = ids >= 0
            keys = head._compute_lexical_keys(ids.clamp_min(0))
            keys = torch.roll(keys, shifts=(lag * 7) % self.head_rank, dims=-1)
            address = address + keys * valid[..., None]
        return F.normalize(address, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        matrices, previous, history = state
        outputs = []
        next_matrices = []
        next_previous = []
        for index, (order, head) in enumerate(zip(self.orders, self.heads)):
            addresses = self._addresses(token_ids, history, head, order)
            values = head.value_proj(hidden_states).float()
            matrix = matrices[:, index]
            prior = previous[:, index]
            reads_out = []
            for address_chunk, value_chunk in zip(
                addresses.split(self.delta_chunk_size, dim=1),
                values.split(self.delta_chunk_size, dim=1),
            ):
                write_keys = torch.cat(
                    (prior[:, None], address_chunk[:, :-1]), dim=1
                )
                matrix, reads = head.vectorized_delta_scan(
                    write_keys, value_chunk, matrix, queries=address_chunk
                )
                reads_out.append(reads)
                prior = address_chunk[:, -1]
            reads = torch.cat(reads_out, dim=1).to(head.output_proj.weight.dtype)
            outputs.append(head.output_proj(reads))
            next_matrices.append(matrix)
            next_previous.append(prior)

        full_ids = torch.cat((history, token_ids), dim=1)
        next_history = full_ids[:, -(self.max_order - 1) :]
        residual = self.residual_scale / math.sqrt(len(self.orders)) * sum(outputs)
        return residual, (
            torch.stack(next_matrices, dim=1),
            torch.stack(next_previous, dim=1),
            next_history,
        )


class MultiTimescaleDeltaAttention(StableDeltaAssociativeAttention):
    """Stable Delta with value channels assigned to learned timescales."""

    def __init__(
        self,
        hidden_size: int,
        state_rank: int = 128,
        num_timescales: int = 4,
        vocab_size: int | None = None,
        contextual_mix_init: float = 0.1,
        residual_scale: float = 0.1,
        delta_chunk_size: int = 512,
    ):
        if state_rank % num_timescales:
            raise ValueError("state_rank must be divisible by num_timescales")
        super().__init__(
            hidden_size,
            state_rank,
            vocab_size=vocab_size,
            contextual_mix_init=contextual_mix_init,
            residual_scale=residual_scale,
            delta_chunk_size=delta_chunk_size,
        )
        del self.write_logit
        del self.decay_logit
        initial_decays = torch.linspace(0.95, 0.9995, num_timescales)
        self.decay_logits = nn.Parameter(torch.logit(initial_decays))
        self.write_logits = nn.Parameter(
            torch.full((num_timescales,), math.log(4.0))
        )
        self.num_timescales = int(num_timescales)
        self.group_size = state_rank // num_timescales

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        association_state, previous_address = state
        lexical = self._lexical_keys(token_ids).float()
        contextual = F.normalize(self.address_proj(hidden_states).float(), dim=-1)
        addresses = F.normalize(
            lexical + torch.tanh(self.contextual_mix.float()) * contextual,
            dim=-1,
        )
        values = self.value_proj(hidden_states).float()
        decays = torch.sigmoid(self.decay_logits.float())
        write_rates = torch.sigmoid(self.write_logits.float())
        outputs = []
        for address_chunk, value_chunk in zip(
            addresses.split(self.delta_chunk_size, dim=1),
            values.split(self.delta_chunk_size, dim=1),
        ):
            write_keys = torch.cat(
                (previous_address[:, None], address_chunk[:, :-1]), dim=1
            )
            batch_size, length, rank = write_keys.shape
            grouped_keys = (
                write_keys[:, None]
                .expand(-1, self.num_timescales, -1, -1)
                .reshape(batch_size * self.num_timescales, length, rank)
            )
            grouped_queries = (
                address_chunk[:, None]
                .expand(-1, self.num_timescales, -1, -1)
                .reshape(batch_size * self.num_timescales, length, rank)
            )
            grouped_values = (
                value_chunk.reshape(
                    batch_size, length, self.num_timescales, self.group_size
                )
                .permute(0, 2, 1, 3)
                .reshape(
                    batch_size * self.num_timescales, length, self.group_size
                )
            )
            grouped_state = (
                association_state.reshape(
                    batch_size, rank, self.num_timescales, self.group_size
                )
                .permute(0, 2, 1, 3)
                .reshape(
                    batch_size * self.num_timescales, rank, self.group_size
                )
            )
            grouped_state, grouped_reads = self.vectorized_delta_scan(
                grouped_keys,
                grouped_values,
                grouped_state,
                queries=grouped_queries,
                decay_override=decays[None]
                .expand(batch_size, -1)
                .reshape(-1),
                write_rate_override=write_rates[None]
                .expand(batch_size, -1)
                .reshape(-1),
            )
            association_state = (
                grouped_state.reshape(
                    batch_size, self.num_timescales, rank, self.group_size
                )
                .permute(0, 2, 1, 3)
                .reshape(batch_size, rank, rank)
            )
            outputs.append(
                grouped_reads.reshape(
                    batch_size,
                    self.num_timescales,
                    length,
                    self.group_size,
                )
                .permute(0, 2, 1, 3)
                .reshape(batch_size, length, rank)
            )
            previous_address = address_chunk[:, -1]
        reads = torch.cat(outputs, dim=1).to(self.output_proj.weight.dtype)
        residual = self.residual_scale * self.output_proj(reads)
        return residual, (association_state, previous_address)


class CollisionNormalizedDeltaAttention(StableDeltaAssociativeAttention):
    """Stable Delta with diagonal normalization of lexical-address load."""

    def __init__(
        self,
        hidden_size: int,
        state_rank: int = 128,
        vocab_size: int | None = None,
        contextual_mix_init: float = 0.1,
        residual_scale: float = 0.1,
        delta_chunk_size: int = 512,
    ):
        super().__init__(
            hidden_size,
            state_rank,
            vocab_size=vocab_size,
            contextual_mix_init=contextual_mix_init,
            residual_scale=residual_scale,
            delta_chunk_size=delta_chunk_size,
        )
        self.state_rank = int(state_rank)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        matrix, previous = super().initial_state(
            batch_size, device=device, dtype=dtype
        )
        load = torch.zeros(
            batch_size, self.state_rank, device=device, dtype=torch.float32
        )
        return matrix, previous, load

    def _normalize_collisions(
        self,
        write_keys: torch.Tensor,
        queries: torch.Tensor,
        load: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = write_keys.shape[1]
        decay = torch.sigmoid(self.decay_logit.float())
        write_rate = torch.sigmoid(self.write_logit.float())
        positions = torch.arange(
            length, device=write_keys.device, dtype=torch.float32
        )
        distance = positions[:, None] - positions[None, :] - 1
        causal_decay = torch.where(
            distance >= 0,
            decay ** distance.clamp_min(0),
            torch.zeros_like(distance),
        )
        initial_powers = decay**positions
        load_before = initial_powers[None, :, None] * load[:, None]
        load_before = load_before + write_rate * torch.bmm(
            causal_decay[None].expand(write_keys.shape[0], -1, -1),
            write_keys.float().square(),
        )
        scale = torch.rsqrt(load_before + 1e-4)
        normalized_keys = F.normalize(write_keys.float() * scale, dim=-1)
        normalized_queries = F.normalize(queries.float() * scale, dim=-1)
        final_powers = decay ** torch.arange(
            length - 1,
            -1,
            -1,
            device=write_keys.device,
            dtype=torch.float32,
        )
        next_load = decay**length * load + write_rate * torch.sum(
            write_keys.float().square() * final_powers[None, :, None], dim=1
        )
        return normalized_keys, normalized_queries, next_load

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        association_state, previous_address, load = state
        lexical = self._lexical_keys(token_ids).float()
        contextual = F.normalize(self.address_proj(hidden_states).float(), dim=-1)
        addresses = F.normalize(
            lexical + torch.tanh(self.contextual_mix.float()) * contextual,
            dim=-1,
        )
        values = self.value_proj(hidden_states).float()
        outputs = []
        for address_chunk, value_chunk in zip(
            addresses.split(self.delta_chunk_size, dim=1),
            values.split(self.delta_chunk_size, dim=1),
        ):
            write_keys = torch.cat(
                (previous_address[:, None], address_chunk[:, :-1]), dim=1
            )
            write_keys, queries, load = self._normalize_collisions(
                write_keys, address_chunk, load
            )
            association_state, reads = self.vectorized_delta_scan(
                write_keys,
                value_chunk,
                association_state,
                queries=queries,
            )
            outputs.append(reads)
            previous_address = address_chunk[:, -1]
        reads = torch.cat(outputs, dim=1).to(self.output_proj.weight.dtype)
        residual = self.residual_scale * self.output_proj(reads)
        return residual, (association_state, previous_address, load)


class LexicalValueDeltaAttention(StableDeltaAssociativeAttention):
    """Stable Delta writing deterministic token codes instead of hidden states."""

    def __init__(
        self,
        hidden_size: int,
        state_rank: int = 128,
        vocab_size: int | None = None,
        contextual_mix_init: float = 0.1,
        residual_scale: float = 0.1,
        delta_chunk_size: int = 512,
    ):
        super().__init__(
            hidden_size,
            state_rank,
            vocab_size=vocab_size,
            contextual_mix_init=contextual_mix_init,
            residual_scale=residual_scale,
            delta_chunk_size=delta_chunk_size,
        )
        del self.value_proj
        self.lexical_value_rank = int(state_rank)
        if vocab_size is None:
            value_table = torch.empty(0, state_rank)
        else:
            value_table = self._compute_lexical_values(
                torch.arange(vocab_size, dtype=torch.int64)
            )
        self.register_buffer("value_table", value_table, persistent=False)

    def _compute_lexical_values(self, token_ids: torch.Tensor) -> torch.Tensor:
        dimensions = torch.arange(
            self.lexical_value_rank + 1,
            2 * self.lexical_value_rank + 1,
            device=token_ids.device,
            dtype=torch.float64,
        )
        phases = (
            (token_ids.to(torch.float64)[..., None] + 0.5)
            * torch.pi
            * torch.sqrt(dimensions)
        )
        return F.normalize(torch.cos(phases).to(torch.float32), dim=-1)

    def _lexical_values(self, token_ids: torch.Tensor) -> torch.Tensor:
        value_table = self.value_table
        assert isinstance(value_table, torch.Tensor)
        if value_table.numel() == 0:
            return self._compute_lexical_values(token_ids)
        return F.embedding(token_ids, value_table)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        association_state, previous_address = state
        lexical = self._lexical_keys(token_ids).float()
        contextual = F.normalize(self.address_proj(hidden_states).float(), dim=-1)
        addresses = F.normalize(
            lexical + torch.tanh(self.contextual_mix.float()) * contextual,
            dim=-1,
        )
        values = self._lexical_values(token_ids).float()
        outputs = []
        for address_chunk, value_chunk in zip(
            addresses.split(self.delta_chunk_size, dim=1),
            values.split(self.delta_chunk_size, dim=1),
        ):
            write_keys = torch.cat(
                (previous_address[:, None], address_chunk[:, :-1]), dim=1
            )
            association_state, reads = self.vectorized_delta_scan(
                write_keys,
                value_chunk,
                association_state,
                queries=address_chunk,
            )
            outputs.append(reads)
            previous_address = address_chunk[:, -1]
        reads = torch.cat(outputs, dim=1).to(self.output_proj.weight.dtype)
        residual = self.residual_scale * self.output_proj(reads)
        return residual, (association_state, previous_address)


class LexicalForgeDeltaAttention(StableDeltaAssociativeAttention):
    """Token-only forge for read/write addresses with contextual values."""

    def __init__(
        self,
        hidden_size: int,
        state_rank: int = 128,
        vocab_size: int | None = None,
        contextual_mix_init: float = 0.1,
        lexical_object_rank: int = 16,
        forge_bottleneck: int = 4,
        residual_scale: float = 0.1,
        delta_chunk_size: int = 512,
    ):
        super().__init__(
            hidden_size,
            state_rank,
            vocab_size=vocab_size,
            contextual_mix_init=contextual_mix_init,
            residual_scale=residual_scale,
            delta_chunk_size=delta_chunk_size,
        )
        self.read_log_scale = nn.Parameter(torch.zeros(state_rank))
        self.write_log_scale = nn.Parameter(torch.zeros(state_rank))
        self.lexical_token_scale: nn.Embedding | None = None
        self.forge_down = nn.Linear(lexical_object_rank, forge_bottleneck, bias=False)
        self.forge_up = nn.Linear(forge_bottleneck, state_rank, bias=False)

    def attach_token_scale(self, token_scale: nn.Embedding) -> None:
        if token_scale.embedding_dim != self.forge_down.in_features:
            raise ValueError(
                f"token_scale rank {token_scale.embedding_dim} does not match "
                f"forge rank {self.forge_down.in_features}"
            )
        self.lexical_token_scale = token_scale

    @staticmethod
    def _forge(code: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(log_scale.float().clamp(-2.0, 2.0))
        return F.normalize(code.float() * scale, dim=-1)

    def forge_lexical_codes(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lexical = self._lexical_keys(token_ids)
        if self.lexical_token_scale is not None:
            modulation = self.forge_up(
                F.silu(self.forge_down(self.lexical_token_scale(token_ids)))
            )
            lexical = F.normalize(lexical + modulation.float(), dim=-1)
        return (
            self._forge(lexical, self.read_log_scale),
            self._forge(lexical, self.write_log_scale),
        )

    def forge_addresses(
        self, hidden_states: torch.Tensor, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lexical_read, lexical_write = self.forge_lexical_codes(token_ids)
        contextual = F.normalize(self.address_proj(hidden_states).float(), dim=-1)
        contextual_mix = torch.tanh(self.contextual_mix.float())
        return (
            F.normalize(lexical_read + contextual_mix * contextual, dim=-1),
            F.normalize(lexical_write + contextual_mix * contextual, dim=-1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        association_state, previous_write = state
        read_codes, write_codes = self.forge_addresses(hidden_states, token_ids)
        values = self.value_proj(hidden_states).float()
        outputs = []
        for read_chunk, write_chunk, value_chunk in zip(
            read_codes.split(self.delta_chunk_size, dim=1),
            write_codes.split(self.delta_chunk_size, dim=1),
            values.split(self.delta_chunk_size, dim=1),
        ):
            write_keys = torch.cat(
                (previous_write[:, None], write_chunk[:, :-1]), dim=1
            )
            association_state, reads = self.vectorized_delta_scan(
                write_keys,
                value_chunk,
                association_state,
                queries=read_chunk,
            )
            outputs.append(reads)
            previous_write = write_chunk[:, -1]
        reads = torch.cat(outputs, dim=1).to(self.output_proj.weight.dtype)
        residual = self.residual_scale * self.output_proj(reads)
        return residual, (association_state, previous_write)








class CollisionNormalizedLexicalForgeDeltaAttention(CollisionNormalizedDeltaAttention):
    """Collision-normalized Delta with the compressed V3 lexical forge."""

    def __init__(self, hidden_size: int, state_rank: int = 128,
                 vocab_size: int | None = None, contextual_mix_init: float = 0.1,
                 lexical_object_rank: int = 16, forge_bottleneck: int = 4,
                 residual_scale: float = 0.1, delta_chunk_size: int = 512):
        super().__init__(hidden_size, state_rank, vocab_size=vocab_size,
                         contextual_mix_init=contextual_mix_init,
                         residual_scale=residual_scale,
                         delta_chunk_size=delta_chunk_size)
        self.lexical_token_scale: nn.Embedding | None = None
        self.forge_down = nn.Linear(lexical_object_rank, forge_bottleneck, bias=False)
        self.forge_up = nn.Linear(forge_bottleneck, state_rank, bias=False)
        self.read_log_scale = nn.Parameter(torch.zeros(state_rank))
        self.write_log_scale = nn.Parameter(torch.zeros(state_rank))

    def attach_token_scale(self, token_scale: nn.Embedding) -> None:
        if token_scale.embedding_dim != self.forge_down.in_features:
            raise ValueError("token_scale rank does not match forge rank")
        self.lexical_token_scale = token_scale

    @staticmethod
    def _forge(code: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(log_scale.float().clamp(-2.0, 2.0))
        return F.normalize(code.float() * scale, dim=-1)

    def forward(self, hidden_states: torch.Tensor, token_ids: torch.Tensor,
                state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        association_state, previous_write, load = state
        lexical = self._lexical_keys(token_ids)
        if self.lexical_token_scale is not None:
            modulation = self.forge_up(
                F.silu(self.forge_down(self.lexical_token_scale(token_ids)))
            )
            lexical = F.normalize(lexical + modulation.float(), dim=-1)
        read_codes = self._forge(lexical, self.read_log_scale)
        write_codes = self._forge(lexical, self.write_log_scale)
        contextual = F.normalize(self.address_proj(hidden_states).float(), dim=-1)
        contextual_mix = torch.tanh(self.contextual_mix.float())
        read_codes = F.normalize(read_codes + contextual_mix * contextual, dim=-1)
        write_codes = F.normalize(write_codes + contextual_mix * contextual, dim=-1)
        values = self.value_proj(hidden_states).float()
        outputs = []
        for read_chunk, write_chunk, value_chunk in zip(
            read_codes.split(self.delta_chunk_size, dim=1),
            write_codes.split(self.delta_chunk_size, dim=1),
            values.split(self.delta_chunk_size, dim=1),
        ):
            write_keys = torch.cat((previous_write[:, None], write_chunk[:, :-1]), dim=1)
            normalized_keys, queries, load = self._normalize_collisions(
                write_keys, read_chunk, load
            )
            association_state, reads = self.vectorized_delta_scan(
                normalized_keys, value_chunk, association_state, queries=queries
            )
            outputs.append(reads)
            previous_write = write_chunk[:, -1]
        reads = torch.cat(outputs, dim=1).to(self.output_proj.weight.dtype)
        residual = self.residual_scale * self.output_proj(reads)
        return residual, (association_state, previous_write, load)


class PositionAwareCollisionForgeDeltaAttention(
    CollisionNormalizedLexicalForgeDeltaAttention
):
    """Collision + Forge V3 with compact context and position signatures."""

    def __init__(self, *args, context_signature_rank: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = self.address_proj.in_features
        state_rank = self.address_proj.out_features
        del self.address_proj
        self.context_down = nn.Linear(
            hidden_size, context_signature_rank, bias=False
        )
        self.context_up = nn.Linear(
            context_signature_rank, state_rank, bias=False
        )
        self.position_gate = nn.Parameter(torch.tensor(0.0))

    def initial_state(self, batch_size: int, *, device: torch.device,
                      dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
        matrix, previous, load = super().initial_state(
            batch_size, device=device, dtype=dtype
        )
        position = torch.zeros(batch_size, device=device, dtype=torch.int64)
        return matrix, previous, load, position

    def _position_signature(self, positions: torch.Tensor) -> torch.Tensor:
        dimensions = torch.arange(
            1, self.state_rank + 1, device=positions.device, dtype=torch.float32
        )
        phases = torch.pi * (positions.float()[..., None] + 1.0) * torch.sqrt(dimensions)
        return F.normalize(torch.sin(phases), dim=-1)

    def forward(self, hidden_states: torch.Tensor, token_ids: torch.Tensor,
                state: tuple[torch.Tensor, ...]
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        association_state, previous_write, load, position = state
        lexical = self._lexical_keys(token_ids)
        if self.lexical_token_scale is not None:
            modulation = self.forge_up(
                F.silu(self.forge_down(self.lexical_token_scale(token_ids)))
            )
            lexical = F.normalize(lexical + modulation.float(), dim=-1)
        read_codes = self._forge(lexical, self.read_log_scale)
        write_codes = self._forge(lexical, self.write_log_scale)
        contextual = F.normalize(
            self.context_up(F.silu(self.context_down(hidden_states))).float(),
            dim=-1,
        )
        contextual_mix = torch.tanh(self.contextual_mix.float())
        local_positions = position[:, None] + torch.arange(
            token_ids.shape[1], device=token_ids.device, dtype=torch.int64
        )[None]
        positional = self._position_signature(local_positions)
        position_mix = torch.tanh(self.position_gate.float())
        occurrence = contextual_mix * contextual + position_mix * positional
        read_codes = F.normalize(read_codes + occurrence, dim=-1)
        write_codes = F.normalize(write_codes + occurrence, dim=-1)
        values = self.value_proj(hidden_states).float()
        outputs = []
        for read_chunk, write_chunk, value_chunk in zip(
            read_codes.split(self.delta_chunk_size, dim=1),
            write_codes.split(self.delta_chunk_size, dim=1),
            values.split(self.delta_chunk_size, dim=1),
        ):
            write_keys = torch.cat((previous_write[:, None], write_chunk[:, :-1]), dim=1)
            normalized_keys, queries, load = self._normalize_collisions(
                write_keys, read_chunk, load
            )
            association_state, reads = self.vectorized_delta_scan(
                normalized_keys, value_chunk, association_state, queries=queries
            )
            outputs.append(reads)
            previous_write = write_chunk[:, -1]
        position = position + token_ids.shape[1]
        reads = torch.cat(outputs, dim=1).to(self.output_proj.weight.dtype)
        residual = self.residual_scale * self.output_proj(reads)
        return residual, (association_state, previous_write, load, position)


class RoutedWriteValueDeltaAttention(StableDeltaAssociativeAttention):
    """Fixed-capacity bank of top-1-routed Write/Value associative slots.

    Each slot owns one fixed-rank WVR matrix. A straight-through top-1 router
    combines a deterministic lexical partition with a learned contextual
    correction. Values remain contextual projections of the hidden state.
    """

    def __init__(
        self,
        hidden_size: int,
        state_rank: int = 128,
        num_slots: int = 4,
        vocab_size: int | None = None,
        contextual_mix_init: float = 0.1,
        residual_scale: float = 0.1,
        delta_chunk_size: int = 512,
    ):
        if num_slots < 2:
            raise ValueError("num_slots must be at least 2")
        super().__init__(
            hidden_size,
            state_rank,
            vocab_size=vocab_size,
            contextual_mix_init=contextual_mix_init,
            residual_scale=residual_scale,
            delta_chunk_size=delta_chunk_size,
        )
        self.num_slots = int(num_slots)
        self.state_rank = int(state_rank)
        self.router_proj = nn.Linear(hidden_size, num_slots, bias=False)
        self.context_route_gate = nn.Parameter(torch.tensor(0.0))

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del dtype
        matrices = torch.zeros(
            batch_size,
            self.num_slots,
            self.state_rank,
            self.state_rank,
            device=device,
            dtype=torch.float32,
        )
        previous_address = torch.zeros(
            batch_size, self.state_rank, device=device, dtype=torch.float32
        )
        previous_route = torch.zeros(
            batch_size, self.num_slots, device=device, dtype=torch.float32
        )
        previous_route[:, 0] = 1.0
        return matrices, previous_address, previous_route

    def route_slots(
        self, hidden_states: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        lexical_slots = torch.remainder(token_ids, self.num_slots)
        lexical_logits = 2.0 * F.one_hot(
            lexical_slots, num_classes=self.num_slots
        ).float()
        contextual_logits = self.router_proj(hidden_states).float()
        logits = lexical_logits + torch.tanh(
            self.context_route_gate.float()
        ) * contextual_logits
        soft_routes = torch.softmax(logits, dim=-1)
        hard_routes = F.one_hot(
            logits.argmax(dim=-1), num_classes=self.num_slots
        ).float()
        return hard_routes + soft_routes - soft_routes.detach()

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        matrices, previous_address, previous_route = state
        lexical = self._lexical_keys(token_ids).float()
        contextual = F.normalize(self.address_proj(hidden_states).float(), dim=-1)
        addresses = F.normalize(
            lexical + torch.tanh(self.contextual_mix.float()) * contextual,
            dim=-1,
        )
        values = self.value_proj(hidden_states).float()
        routes = self.route_slots(hidden_states, token_ids)
        decay = torch.sigmoid(self.decay_logit.float())
        write_rate = torch.sigmoid(self.write_logit.float())
        reads = []

        for position in range(hidden_states.shape[1]):
            query = addresses[:, position]
            route = routes[:, position]
            slot_reads = torch.einsum("br,bsrv->bsv", query, matrices)
            read = torch.sum(route[..., None] * slot_reads, dim=1)
            reads.append(read)

            slot_predictions = torch.einsum(
                "br,bsrv->bsv", previous_address, matrices
            )
            prediction = torch.sum(
                previous_route[..., None] * slot_predictions, dim=1
            )
            error = values[:, position] - prediction
            update = torch.einsum("br,bv->brv", previous_address, error)
            matrices = decay * matrices + write_rate * (
                previous_route[..., None, None] * update[:, None]
            )
            previous_address = query
            previous_route = route

        read_tensor = torch.stack(reads, dim=1).to(self.output_proj.weight.dtype)
        residual = self.residual_scale * self.output_proj(read_tensor)
        return residual, (matrices, previous_address, previous_route)


class SharedRoutedWriteValueMemory(StableDeltaAssociativeAttention):
    """Stable shared WVR plus a zero-gated low-rank routed residual memory."""

    def __init__(
        self,
        hidden_size: int,
        state_rank: int = 128,
        routed_rank: int = 32,
        num_slots: int = 4,
        vocab_size: int | None = None,
        contextual_mix_init: float = 0.1,
        residual_scale: float = 0.1,
        delta_chunk_size: int = 512,
    ):
        if num_slots < 2:
            raise ValueError("num_slots must be at least 2")
        if routed_rank <= 0 or routed_rank > state_rank:
            raise ValueError("routed_rank must be in [1, state_rank]")
        super().__init__(
            hidden_size,
            state_rank,
            vocab_size=vocab_size,
            contextual_mix_init=contextual_mix_init,
            residual_scale=residual_scale,
            delta_chunk_size=delta_chunk_size,
        )
        self.state_rank = int(state_rank)
        self.routed_rank = int(routed_rank)
        self.num_slots = int(num_slots)
        self.routed_value_proj = nn.Linear(hidden_size, routed_rank, bias=False)
        self.routed_output_proj = nn.Linear(routed_rank, hidden_size, bias=False)
        self.route_proj = nn.Linear(hidden_size, num_slots, bias=False)
        self.context_route_gate = nn.Parameter(torch.tensor(0.0))
        self.routed_gate = nn.Parameter(torch.tensor(0.0))

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        common_matrix, common_previous = super().initial_state(
            batch_size, device=device, dtype=dtype
        )
        routed_matrices = torch.zeros(
            batch_size,
            self.num_slots,
            self.routed_rank,
            self.routed_rank,
            device=device,
            dtype=torch.float32,
        )
        routed_previous = torch.zeros(
            batch_size, self.routed_rank, device=device, dtype=torch.float32
        )
        previous_route = torch.zeros(
            batch_size, self.num_slots, device=device, dtype=torch.float32
        )
        previous_route[:, 0] = 1.0
        return (
            common_matrix,
            common_previous,
            routed_matrices,
            routed_previous,
            previous_route,
        )

    def route_slots(
        self, hidden_states: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        lexical_slots = torch.remainder(token_ids, self.num_slots)
        lexical_logits = 2.0 * F.one_hot(
            lexical_slots, num_classes=self.num_slots
        ).float()
        logits = lexical_logits + torch.tanh(
            self.context_route_gate.float()
        ) * self.route_proj(hidden_states).float()
        soft = torch.softmax(logits, dim=-1)
        hard = F.one_hot(
            logits.argmax(dim=-1), num_classes=self.num_slots
        ).float()
        return hard + soft - soft.detach()

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        (
            common_matrix,
            common_previous,
            routed_matrices,
            routed_previous,
            previous_route,
        ) = state
        common_output, (common_matrix, common_previous) = super().forward(
            hidden_states,
            token_ids,
            (common_matrix, common_previous),
        )

        lexical = self._lexical_keys(token_ids).float()
        contextual = F.normalize(self.address_proj(hidden_states).float(), dim=-1)
        common_addresses = F.normalize(
            lexical + torch.tanh(self.contextual_mix.float()) * contextual,
            dim=-1,
        )
        routed_addresses = F.normalize(
            common_addresses[..., : self.routed_rank], dim=-1
        )
        routed_values = self.routed_value_proj(hidden_states).float()
        routes = self.route_slots(hidden_states, token_ids)
        routed_reads = []
        batch_size = hidden_states.shape[0]
        for address_chunk, value_chunk, route_chunk in zip(
            routed_addresses.split(self.delta_chunk_size, dim=1),
            routed_values.split(self.delta_chunk_size, dim=1),
            routes.split(self.delta_chunk_size, dim=1),
        ):
            write_addresses = torch.cat(
                (routed_previous[:, None], address_chunk[:, :-1]), dim=1
            )
            write_routes = torch.cat(
                (previous_route[:, None], route_chunk[:, :-1]), dim=1
            )
            route_scale = write_routes
            length = address_chunk.shape[1]
            grouped_keys = (
                route_scale[..., None] * write_addresses[:, :, None]
            ).permute(0, 2, 1, 3).reshape(
                batch_size * self.num_slots, length, self.routed_rank
            )
            grouped_values = (
                route_scale[..., None] * value_chunk[:, :, None]
            ).permute(0, 2, 1, 3).reshape(
                batch_size * self.num_slots, length, self.routed_rank
            )
            grouped_queries = address_chunk[:, None].expand(
                -1, self.num_slots, -1, -1
            ).reshape(batch_size * self.num_slots, length, self.routed_rank)
            grouped_state = routed_matrices.reshape(
                batch_size * self.num_slots,
                self.routed_rank,
                self.routed_rank,
            )
            grouped_state, grouped_reads = self.vectorized_delta_scan(
                grouped_keys,
                grouped_values,
                grouped_state,
                queries=grouped_queries,
            )
            routed_matrices = grouped_state.reshape(
                batch_size,
                self.num_slots,
                self.routed_rank,
                self.routed_rank,
            )
            slot_reads = grouped_reads.reshape(
                batch_size, self.num_slots, length, self.routed_rank
            ).permute(0, 2, 1, 3)
            routed_reads.append(
                torch.sum(route_chunk[..., None] * slot_reads, dim=2)
            )
            routed_previous = address_chunk[:, -1]
            previous_route = route_chunk[:, -1]

        routed_read = torch.cat(routed_reads, dim=1).to(
            self.routed_output_proj.weight.dtype
        )
        routed_output = self.residual_scale * self.routed_output_proj(routed_read)
        output = common_output + torch.tanh(self.routed_gate) * routed_output
        return output, (
            common_matrix,
            common_previous,
            routed_matrices,
            routed_previous,
            previous_route,
        )


class AugmentedRoutedWVR(StableDeltaAssociativeAttention):
    """Single-scan WVR with shared and orthogonal routed address subspaces."""

    def __init__(
        self,
        hidden_size: int,
        state_rank: int = 128,
        num_slots: int = 4,
        slot_rank: int = 8,
        vocab_size: int | None = None,
        contextual_mix_init: float = 0.1,
        residual_scale: float = 0.1,
        delta_chunk_size: int = 512,
    ):
        if num_slots < 2:
            raise ValueError("num_slots must be at least 2")
        if slot_rank <= 0:
            raise ValueError("slot_rank must be positive")
        super().__init__(
            hidden_size,
            state_rank,
            vocab_size=vocab_size,
            contextual_mix_init=contextual_mix_init,
            residual_scale=residual_scale,
            delta_chunk_size=delta_chunk_size,
        )
        self.state_rank = int(state_rank)
        self.num_slots = int(num_slots)
        self.slot_rank = int(slot_rank)
        self.total_address_rank = state_rank + num_slots * slot_rank
        self.routed_address_proj = nn.Linear(hidden_size, slot_rank, bias=False)
        self.route_proj = nn.Linear(hidden_size, num_slots, bias=False)
        self.context_route_gate = nn.Parameter(torch.tensor(0.0))
        self.routed_address_gate = nn.Parameter(torch.tensor(0.0))

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del dtype
        matrix = torch.zeros(
            batch_size,
            self.total_address_rank,
            self.state_rank,
            device=device,
            dtype=torch.float32,
        )
        previous_write = torch.zeros(
            batch_size,
            self.total_address_rank,
            device=device,
            dtype=torch.float32,
        )
        return matrix, previous_write

    def route_slots(
        self, hidden_states: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        lexical_slots = torch.remainder(token_ids, self.num_slots)
        lexical_logits = 2.0 * F.one_hot(
            lexical_slots, num_classes=self.num_slots
        ).float()
        logits = lexical_logits + torch.tanh(
            self.context_route_gate.float()
        ) * self.route_proj(hidden_states).float()
        soft = torch.softmax(logits, dim=-1)
        hard = F.one_hot(
            logits.argmax(dim=-1), num_classes=self.num_slots
        ).float()
        return hard + soft - soft.detach()

    def addresses(
        self, hidden_states: torch.Tensor, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lexical = self._lexical_keys(token_ids).float()
        contextual = F.normalize(self.address_proj(hidden_states).float(), dim=-1)
        common = F.normalize(
            lexical + torch.tanh(self.contextual_mix.float()) * contextual,
            dim=-1,
        )
        local = F.normalize(
            self.routed_address_proj(hidden_states).float()
            + lexical[..., : self.slot_rank],
            dim=-1,
        )
        routes = self.route_slots(hidden_states, token_ids)
        routed = (routes[..., None] * local[:, :, None]).flatten(2)
        read_addresses = torch.cat((common, routed), dim=-1)
        write_addresses = torch.cat(
            (
                common,
                torch.tanh(self.routed_address_gate.float()) * routed,
            ),
            dim=-1,
        )
        return read_addresses, write_addresses

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        association_state, previous_write = state
        read_addresses, write_addresses = self.addresses(hidden_states, token_ids)
        values = self.value_proj(hidden_states).float()
        outputs = []
        for read_chunk, write_chunk, value_chunk in zip(
            read_addresses.split(self.delta_chunk_size, dim=1),
            write_addresses.split(self.delta_chunk_size, dim=1),
            values.split(self.delta_chunk_size, dim=1),
        ):
            shifted_writes = torch.cat(
                (previous_write[:, None], write_chunk[:, :-1]), dim=1
            )
            association_state, reads = self.vectorized_delta_scan(
                shifted_writes,
                value_chunk,
                association_state,
                queries=read_chunk,
            )
            outputs.append(reads)
            previous_write = write_chunk[:, -1]
        read_values = torch.cat(outputs, dim=1).to(self.output_proj.weight.dtype)
        residual = self.residual_scale * self.output_proj(read_values)
        return residual, (association_state, previous_write)
