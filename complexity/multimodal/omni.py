"""
OmniModel — unified any-to-any multimodal model.

Handles text + image + audio + video in a single transformer backbone.
Inspired by Gemini / GPT-4o / Chameleon.

Architecture
------------
1. Modality encoders
   - Text   : nn.Embedding
   - Image  : VisionTransformer  (patch tokens)
   - Audio  : MelSpectrogramEncoder  (frame tokens)
   - Video  : VideoTransformer  (tubelet tokens)
   All projected to shared hidden_size.

2. PositionRoutedMLP  ← generic base
   Single reusable class: routes tokens by *sequential position* within the
   sequence (pos % num_experts). Deterministic, fullgraph=True safe.
   Fused BMM: gate+up → SwiGLU → down.

3. OmniBlock  ← key design
   One INDEPENDENT PositionRoutedMLP per modality:
       self.text_mlp   — experts specialized for text tokens
       self.image_mlp  — experts specialized for image patches
       self.audio_mlp  — experts specialized for audio frames
       self.video_mlp  — experts specialized for video tubelets

   Each modality's MLP can have its own num_experts / intermediate_size.
   Dispatch via masked sum (fullgraph=True safe):
       out = text_mlp(x) * text_mask + image_mlp(x) * image_mask + …

4. Output: text logits + last_hidden_state over all tokens.

Usage
-----
    from complexity.multimodal.omni import OmniModel, OmniConfig

    model = OmniModel(OmniConfig(hidden_size=1024, vocab_size=32000))
    out = model(
        text_ids=torch.randint(0, 32000, (2, 128)),
        pixel_values=torch.randn(2, 3, 224, 224),
        audio_features=torch.randn(2, 80, 3000),
        video_frames=torch.randn(2, 16, 3, 224, 224),
    )
    logits = out["logits"]             # [2, 128, vocab_size]
    tokens = out["last_hidden_state"]  # [2, total_tokens, hidden_size]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision import VisionTransformer, VisionConfig
from .audio import MelSpectrogramEncoder, AudioConfig
from .video import VideoTransformer, VideoConfig


# =============================================================================
# Modality registry — single source of truth
# =============================================================================

class Modality(IntEnum):
    """
    Modality identifiers used for expert dispatch in OmniBlock.

    IntEnum → usable directly as tensor indices and in arithmetic.
    Adding a new modality here is the ONLY change needed in this file.
    Everything else (ModuleDict, validation, dispatch) derives from it.
    """
    TEXT  = 0
    IMAGE = 1
    AUDIO = 2
    VIDEO = 3

    @classmethod
    def count(cls) -> int:
        return len(cls)


# =============================================================================
# Per-modality MLP config
# =============================================================================

@dataclass
class ModalityMLPConfig:
    """
    Expert configuration for one modality's specialised MLP.

    Rule enforced at OmniConfig validation:
        intermediate_size % num_experts == 0
    """
    num_experts: int = 4
    intermediate_size: int = 4096


# =============================================================================
# Omni config
# =============================================================================

@dataclass
class OmniConfig:
    """Unified configuration for OmniModel."""

    # ---- Backbone ----
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    layer_norm_eps: float = 1e-6
    dropout: float = 0.0

    # ---- General MLP (shared by ALL tokens) ----
    # Rule: general_intermediate_size % general_num_experts == 0
    general_num_experts: int = 8        # 4096 / 8 = 512 per expert ✓
    general_intermediate_size: int = 4096
    # block_size = K → general_expert = (position // K) % general_num_experts
    # Set to the number of specialized experts so each general expert covers
    # exactly one "block" of specialized positions.
    # 0 = auto (derived from modality_mlp defaults at build time)
    general_block_size: int = 0

    # ---- Per-modality MLPs ----
    # Keyed by Modality enum — no modality names hardcoded here.
    # Default: 4 experts, 4096 intermediate for every modality.
    modality_mlp: Dict[Modality, ModalityMLPConfig] = field(
        default_factory=lambda: {m: ModalityMLPConfig() for m in Modality}
    )

    # ---- Text encoder ----
    vocab_size: int = 32000

    # ---- Image encoder ----
    image_size: int = 224
    patch_size: int = 16
    vision_hidden_size: int = 768
    vision_num_layers: int = 12
    vision_num_heads: int = 12

    # ---- Audio encoder ----
    n_mels: int = 80
    audio_hidden_size: int = 768
    audio_num_layers: int = 6
    audio_num_heads: int = 12
    audio_max_length: int = 3000

    # ---- Video encoder ----
    num_frames: int = 16
    temporal_patch_size: int = 2
    video_hidden_size: int = 768
    video_num_layers: int = 12
    video_num_heads: int = 12

    def __post_init__(self):
        """Validate all expert / intermediate-size pairs at construction time."""
        errors = []

        # General MLP
        if self.general_num_experts < 1:
            errors.append(f"  general_num_experts={self.general_num_experts} must be >= 1")
        elif self.general_intermediate_size % self.general_num_experts != 0:
            errors.append(
                f"  general_intermediate_size={self.general_intermediate_size} must be "
                f"divisible by general_num_experts={self.general_num_experts} "
                f"(remainder {self.general_intermediate_size % self.general_num_experts})"
            )

        # Per-modality MLPs — loop derives from Modality enum, never hardcoded
        for m in Modality:
            cfg = self.modality_mlp.get(m)
            if cfg is None:
                errors.append(f"  modality_mlp missing entry for {m.name}")
                continue
            if cfg.num_experts < 1:
                errors.append(f"  {m.name} num_experts={cfg.num_experts} must be >= 1")
            elif cfg.intermediate_size % cfg.num_experts != 0:
                errors.append(
                    f"  {m.name} intermediate_size={cfg.intermediate_size} must be "
                    f"divisible by num_experts={cfg.num_experts} "
                    f"(remainder {cfg.intermediate_size % cfg.num_experts})"
                )

        # Attention
        if self.hidden_size % self.num_attention_heads != 0:
            errors.append(
                f"  hidden_size={self.hidden_size} must be divisible by "
                f"num_attention_heads={self.num_attention_heads}"
            )

        if errors:
            raise ValueError("OmniConfig validation failed:\n" + "\n".join(errors))


# =============================================================================
# PositionRoutedMLP — generic reusable base
# =============================================================================

class PositionRoutedMLP(nn.Module):
    """
    Generic Position-Routed MLP with optional hierarchical routing.

    Routing formula controlled by block_size:

        block_size = 1  (default, fine-grained):
            expert_id = position % num_experts

        block_size = K  (coarse/hierarchical):
            expert_id = (position // K) % num_experts

    Hierarchical use in OmniBlock:
        - general_mlp : block_size = specialized_num_experts
                        → each expert covers a BLOCK of K positions
        - modal_mlps  : block_size = 1
                        → each expert covers individual positions within block

    Result: a 2-level position tree:
        position p → general_expert  = (p // K) % G
                   → modal_expert    = p % K

    Fused BMM: gate+up → SwiGLU → down (fullgraph=True safe).

    Parameters
    ----------
    hidden_size       : transformer hidden dimension
    intermediate_size : total intermediate dim (split across experts)
    num_experts       : number of experts
    block_size        : routing granularity (1 = fine, K = coarse blocks)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        block_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.block_size = block_size
        self.expert_intermediate_size = intermediate_size // num_experts

        # Fused gate+up: [E, H, 2*I_e]
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, self.expert_intermediate_size * 2) * 0.02
        )
        # Down: [E, I_e, H]
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, self.expert_intermediate_size, hidden_size) * 0.02
        )

    def forward(
        self,
        x: torch.Tensor,              # [B, N, H]
        position_ids: torch.Tensor,   # [B, N] or [N] — per-modality positions
    ) -> torch.Tensor:
        """
        Args:
            x           : [B, N, H]
            position_ids: [B, N] or [N]  (indices within this modality's segment)

        Returns:
            [B, N, H]
        """
        B, N, H = x.shape

        # Broadcast position_ids to [B, N] if needed
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(B, -1)

        # Hierarchical routing: block_size=1 → fine, block_size=K → coarse
        expert_ids = (position_ids // self.block_size) % self.num_experts  # [B, N]

        flat = x.view(B * N, H)
        eids = expert_ids.reshape(B * N)

        gu_w   = self.gate_up_proj[eids]   # [B*N, H, 2*I_e]
        down_w = self.down_proj[eids]      # [B*N, I_e, H]

        gu = torch.bmm(flat.unsqueeze(1), gu_w).squeeze(1)         # [B*N, 2*I_e]
        gate, up = gu.split(self.expert_intermediate_size, dim=-1)
        inter = F.silu(gate) * up                                   # [B*N, I_e]
        out = torch.bmm(inter.unsqueeze(1), down_w).squeeze(1)     # [B*N, H]

        return out.view(B, N, H)


# =============================================================================
# Attention
# =============================================================================

class OmniAttention(nn.Module):
    """Multi-head self-attention over all modality tokens (shared)."""

    def __init__(self, config: OmniConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = self.dropout(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj(out)


# =============================================================================
# OmniBlock — 4 independent PositionRoutedMLPs, one per modality
# =============================================================================

class OmniBlock(nn.Module):
    """
    Pre-norm transformer block with 2-layer MLP cascade.

    Per forward pass:

        1. Shared attention  — all tokens attend to all tokens.

        2. General MLP  (general_num_experts, fully shared)
           Every token passes through this regardless of modality.
           Captures cross-modal common knowledge.
               x = x + general_mlp(norm2(x), position_ids)

        3. Specialised MLPs  (one per modality, each with its own experts)
           Each token is routed to its modality's dedicated MLP.
           Masked sum → fullgraph=True safe, no dynamic shapes.
               x = x + (text_mlp * text_mask + image_mlp * image_mask + …)

    Expert budget (defaults):
        General   : 12 experts   — shared "backbone" knowledge
        Specialised: 4 experts × 4 modalities = 16 modal experts

    Both counts are fully configurable via OmniConfig.
    """

    def __init__(self, config: OmniConfig):
        super().__init__()
        H = config.hidden_size

        self.attn  = OmniAttention(config)
        self.norm1 = nn.LayerNorm(H, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(H, eps=config.layer_norm_eps)  # feeds general MLP
        self.norm3 = nn.LayerNorm(H, eps=config.layer_norm_eps)  # feeds specialised MLPs

        # Resolve block_size: 0 = auto → min specialized expert count
        block_size = config.general_block_size or min(
            cfg.num_experts for cfg in config.modality_mlp.values()
        )

        # General MLP — shared by all tokens, coarse routing by block
        self.general_mlp = PositionRoutedMLP(
            H, config.general_intermediate_size, config.general_num_experts,
            block_size=block_size,
        )

        # Specialised MLPs — built from Modality enum, no names hardcoded
        self.modal_mlps = nn.ModuleDict({
            m.name: PositionRoutedMLP(
                H,
                config.modality_mlp[m].intermediate_size,
                config.modality_mlp[m].num_experts,
            )
            for m in Modality
        })

    def forward(
        self,
        x: torch.Tensor,                    # [B, N, H]
        modality_ids: torch.Tensor,          # [B, N]  values in Modality
        position_ids: torch.Tensor,          # [B, N]  per-modality positions
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Shared attention
        x = x + self.attn(self.norm1(x), attention_mask)

        # 2. General MLP — every token, routing by position
        x = x + self.general_mlp(self.norm2(x), position_ids)

        # 3. Specialised MLPs — loop derives from Modality enum, masked sum
        normed = self.norm3(x)
        spec_out = sum(
            self.modal_mlps[m.name](normed, position_ids)
            * (modality_ids == m).unsqueeze(-1).float()
            for m in Modality
        )
        return x + spec_out


# =============================================================================
# OmniModel
# =============================================================================

class OmniModel(nn.Module):
    """
    Unified any-to-any multimodal model.

    Each modality is encoded, projected to hidden_size, then packed into a
    single token sequence. A learned boundary token is prepended to each
    modality segment. All tokens are processed jointly by OmniBlocks where
    attention is shared but the MLP is modality-specific.
    """

    def __init__(self, config: OmniConfig):
        super().__init__()
        self.config = config
        H = config.hidden_size

        # Learned boundary token per modality — size from enum, not hardcoded
        self.modality_tokens = nn.Embedding(Modality.count(), H)

        # ---- Text ----
        self.text_embed = nn.Embedding(config.vocab_size, H)

        # ---- Image ----
        vision_cfg = VisionConfig(
            image_size=config.image_size,
            patch_size=config.patch_size,
            hidden_size=config.vision_hidden_size,
            num_hidden_layers=config.vision_num_layers,
            num_attention_heads=config.vision_num_heads,
            use_class_token=False,
            num_experts=config.modality_mlp[Modality.IMAGE].num_experts,
        )
        self.image_encoder = VisionTransformer(vision_cfg)
        self.image_proj = nn.Linear(config.vision_hidden_size, H)

        # ---- Audio ----
        audio_cfg = AudioConfig(
            n_mels=config.n_mels,
            hidden_size=config.audio_hidden_size,
            num_hidden_layers=config.audio_num_layers,
            num_attention_heads=config.audio_num_heads,
            max_length=config.audio_max_length,
            num_experts=config.modality_mlp[Modality.AUDIO].num_experts,
        )
        self.audio_encoder = MelSpectrogramEncoder(audio_cfg)
        self.audio_proj = nn.Linear(config.audio_hidden_size, H)

        # ---- Video ----
        video_cfg = VideoConfig(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_frames=config.num_frames,
            temporal_patch_size=config.temporal_patch_size,
            hidden_size=config.video_hidden_size,
            num_hidden_layers=config.video_num_layers,
            num_attention_heads=config.video_num_heads,
            num_experts=config.modality_mlp[Modality.VIDEO].num_experts,
        )
        self.video_encoder = VideoTransformer(video_cfg)
        self.video_proj = nn.Linear(config.video_hidden_size, H)

        # ---- Shared backbone ----
        self.blocks = nn.ModuleList([OmniBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(H, eps=config.layer_norm_eps)

        # ---- LM head ----
        self.lm_head = nn.Linear(H, config.vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Token packing
    # ------------------------------------------------------------------

    def _pack(
        self,
        tokens: torch.Tensor,   # [B, L, H]
        modality_id: int,
        device: torch.device,
    ):
        """
        Prepend boundary token, build modality_ids and position_ids.

        Returns:
            tokens   [B, 1+L, H]
            mod_ids  [1+L]   long  (constant = modality_id)
            pos_ids  [1+L]   long  (0 for boundary, 0..L-1 for tokens)
        """
        B, L, H = tokens.shape
        boundary = (
            self.modality_tokens(torch.tensor(modality_id, device=device))
            .view(1, 1, H).expand(B, -1, -1)
        )
        tokens = torch.cat([boundary, tokens], dim=1)

        mod_ids = torch.full((1 + L,), modality_id, dtype=torch.long, device=device)
        pos_ids = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.arange(L, dtype=torch.long, device=device),
        ])
        return tokens, mod_ids, pos_ids

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        text_ids: Optional[torch.Tensor] = None,       # [B, Lt]
        pixel_values: Optional[torch.Tensor] = None,   # [B, C, H, W]
        audio_features: Optional[torch.Tensor] = None, # [B, n_mels, T_a]
        video_frames: Optional[torch.Tensor] = None,   # [B, T, C, H, W]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        All inputs optional — any subset of modalities is valid.

        Returns
        -------
        last_hidden_state : [B, total_tokens, H]
        logits            : [B, text_tokens, vocab_size]  (empty if no text)
        text_length       : int
        """
        device = self._first_device(text_ids, pixel_values, audio_features, video_frames)
        B = self._batch_size(text_ids, pixel_values, audio_features, video_frames)

        segs_tokens, segs_mod, segs_pos = [], [], []
        text_len = 0

        if text_ids is not None:
            t, m, p = self._pack(self.text_embed(text_ids), Modality.TEXT, device)
            segs_tokens.append(t); segs_mod.append(m); segs_pos.append(p)
            text_len = t.shape[1]

        if pixel_values is not None:
            enc = self.image_proj(self.image_encoder(pixel_values)["last_hidden_state"])
            enc, m, p = self._pack(enc, Modality.IMAGE, device)
            segs_tokens.append(enc); segs_mod.append(m); segs_pos.append(p)

        if audio_features is not None:
            enc = self.audio_proj(self.audio_encoder(audio_features)["last_hidden_state"])
            enc, m, p = self._pack(enc, Modality.AUDIO, device)
            segs_tokens.append(enc); segs_mod.append(m); segs_pos.append(p)

        if video_frames is not None:
            enc = self.video_proj(self.video_encoder(video_frames)["last_hidden_state"])
            enc, m, p = self._pack(enc, Modality.VIDEO, device)
            segs_tokens.append(enc); segs_mod.append(m); segs_pos.append(p)

        x = torch.cat(segs_tokens, dim=1)                                # [B, N, H]
        mod_ids = torch.cat(segs_mod).unsqueeze(0).expand(B, -1)         # [B, N]
        pos_ids = torch.cat(segs_pos).unsqueeze(0).expand(B, -1)         # [B, N]

        for block in self.blocks:
            x = block(x, mod_ids, pos_ids, attention_mask)
        x = self.norm(x)

        logits = (
            self.lm_head(x[:, :text_len])
            if text_len > 0
            else x.new_zeros(B, 0, self.config.vocab_size)
        )

        return {
            "last_hidden_state": x,
            "logits": logits,
            "text_length": text_len,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _batch_size(*tensors) -> int:
        for t in tensors:
            if t is not None:
                return t.shape[0]
        raise ValueError("At least one input modality must be provided.")

    @staticmethod
    def _first_device(*tensors) -> torch.device:
        for t in tensors:
            if t is not None:
                return t.device
        raise ValueError("At least one input modality must be provided.")
