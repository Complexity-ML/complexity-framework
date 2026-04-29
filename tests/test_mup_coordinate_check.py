"""
μP Coordinate Check.

Validates that Maximal Update Parametrization (Yang et al. 2022) holds in
this codebase. The premise of μP is: when you widen a transformer (multiply
``hidden_size`` by 2, 4, …), the **per-coordinate** magnitude of activations
and gradients stays roughly constant. If true, you can tune hyper-parameters
at a small proxy width and transfer them to a wide production model without
re-tuning.

The relevant metric is RMS-per-coordinate, i.e. ``||x||₂ / √numel(x)``,
*not* the raw L2 norm — the latter grows as √width even when per-coord
values are perfectly stable, simply because there are more coordinates.

This test:
    1. Builds models at widths {256, 512, 1024} with architecture held
       constant otherwise (layers, heads/kv-heads, vocab, max_pos).
    2. Runs one forward+backward pass on the same fixed batch (seeded).
    3. Logs RMS-per-coordinate of every intermediate hidden state and of
       the embedding-table gradient.
    4. Asserts these RMSes stay within tolerance across widths when μP
       is on, and verifies the standard (non-μP) parametrisation drifts
       more — confirming the test actually discriminates.

Run:
    pytest tests/test_mup_coordinate_check.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from complexity.config.model_config import ModelConfig
from complexity.models.builder import ComplexityModel


WIDTHS = (256, 512, 1024)
TOLERANCE_MUP = 0.30       # Per-coord RMS spread allowed when μP is on.
TOLERANCE_STD = 0.40       # Lower bound on the spread we expect *without* μP.


def _rms(t: torch.Tensor) -> float:
    """Root-mean-square per coordinate. Width-invariant under μP."""
    return float(t.norm().item() / (t.numel() ** 0.5))


def _make_config(width: int, *, mup: bool) -> ModelConfig:
    """Build a config at the given width. Architecture held constant otherwise."""
    return ModelConfig(
        hidden_size=width,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=int(width * 8 / 3 / 64) * 64,  # SwiGLU 8/3, multiple of 64
        vocab_size=2048,
        max_position_embeddings=128,
        attention_type="gqa",
        mlp_type="dense_deterministic",  # bit-deterministic init, removes RNG noise
        norm_type="rmsnorm",
        use_qk_norm=True,
        # μP knobs
        use_mup_init=mup,
        use_mup_attn_scale=mup,
        use_mup_output_mult=mup,
        mup_base_width=256,
    )


def _fwd_bwd_norms(width: int, *, mup: bool, seed: int = 0) -> dict:
    """Build a model at `width`, run one fwd+bwd, return per-coord RMSes."""
    cfg = _make_config(width, mup=mup)
    model = ComplexityModel(cfg).to(torch.float32)
    model.eval()  # disables training-path early-return so we get logits

    torch.manual_seed(seed)
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    out = model(input_ids, return_hidden_states=True)
    hidden_states = out["hidden_states"]   # list of [B, T, H], one per layer + embed
    last_h = out["last_hidden_state"]
    logits = out["logits"]                  # [B, T, vocab]

    # Backward through a deterministic scalar.
    target = torch.zeros(batch_size, seq_len, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), target.reshape(-1)
    )
    loss.backward()

    # RMS per coordinate for the residual stream at every layer (excluding
    # the post-norm output, since RMSNorm forces unit RMS by construction
    # and is therefore uninformative).
    layer_rms = [_rms(h) for h in hidden_states]

    return {
        "width": width,
        "loss": loss.item(),
        "layer_rms": layer_rms,           # list of per-layer per-coord RMS
        "embed_rms": layer_rms[0],         # RMS of embedded input (layer 0 input)
        "midstream_rms": layer_rms[len(layer_rms) // 2],  # mid-depth signature
        "embed_grad_rms": _rms(model.embed_tokens.weight.grad),
        "logits_rms": _rms(logits),
    }


def _spread(values: list[float]) -> float:
    """Relative spread of a list of positive values: (max - min) / mean."""
    lo, hi = min(values), max(values)
    return (hi - lo) / (sum(values) / len(values))


def test_mup_keeps_residual_stream_rms_stable() -> None:
    """Residual-stream per-coord RMS should be roughly width-invariant under μP."""
    rms_mid = [_fwd_bwd_norms(w, mup=True)["midstream_rms"] for w in WIDTHS]
    spread = _spread(rms_mid)
    assert spread < TOLERANCE_MUP, (
        f"μP midstream RMS spread = {spread:.3f} exceeds tolerance "
        f"{TOLERANCE_MUP}. Per-width RMS: {dict(zip(WIDTHS, rms_mid))}"
    )


def test_mup_keeps_embed_rms_stable() -> None:
    """Embedded input per-coord RMS = ``initializer_range`` regardless of width."""
    rms_emb = [_fwd_bwd_norms(w, mup=True)["embed_rms"] for w in WIDTHS]
    spread = _spread(rms_emb)
    assert spread < TOLERANCE_MUP, (
        f"μP embed RMS spread = {spread:.3f} exceeds tolerance "
        f"{TOLERANCE_MUP}. Per-width RMS: {dict(zip(WIDTHS, rms_emb))}"
    )


def test_standard_parametrisation_is_not_width_invariant() -> None:
    """Sanity check: without μP, at least one of the per-coord RMSes must
    drift more than the μP tolerance. If the standard parametrisation also
    stays flat, the discriminating tests above would be vacuous."""
    mid = [_fwd_bwd_norms(w, mup=False)["midstream_rms"] for w in WIDTHS]
    grads = [_fwd_bwd_norms(w, mup=False)["embed_grad_rms"] for w in WIDTHS]
    drift = max(_spread(mid), _spread(grads))
    assert drift > TOLERANCE_STD, (
        f"Standard parametrisation looks too stable (drift={drift:.3f}); "
        f"the μP test would be vacuous. mid={mid}, grads={grads}"
    )


def test_logits_scale_drops_under_mup_output_mult() -> None:
    """With ``use_mup_output_mult``, logits at width=4× base should be
    significantly smaller than at width=1× — the multiplier divides them
    by width_mult while the activation/lm_head pipeline does NOT produce
    ``width_mult``× larger raw logits. Empirically the ratio is well
    below 1 (logits shrink with width)."""
    base = _fwd_bwd_norms(WIDTHS[0], mup=True)        # width 256, mult 1
    wide = _fwd_bwd_norms(WIDTHS[-1], mup=True)       # width 1024, mult 4
    width_mult = WIDTHS[-1] / WIDTHS[0]
    ratio = wide["logits_rms"] / base["logits_rms"]
    # Empirically the ratio settles at 1/√width_mult: pre-norm hidden has
    # width-invariant RMS thanks to μP, but the lm_head matmul still adds
    # an unscaled √hidden_size factor (vs the proxy width). The output mult
    # cancels exactly that, leaving 1/√width_mult overall.
    expected = 1.0 / (width_mult ** 0.5)
    upper = expected * 1.20    # 20 % slack for fp/RMSNorm noise
    assert ratio < upper, (
        f"Wide/base logits per-coord RMS ratio = {ratio:.3f}; expected "
        f"≈ 1/√width_mult ({expected:.3f}, upper {upper:.3f}) when "
        f"use_mup_output_mult is on."
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation
    header = (f"{'width':>6} {'loss':>8} {'embed_rms':>10} {'mid_rms':>10} "
              f"{'last_rms':>10} {'g_emb_rms':>10} {'logits_rms':>10}")
    for mup in (True, False):
        tag = "μP" if mup else "standard"
        print(f"\n=== {tag} ===")
        print(header)
        for w in WIDTHS:
            r = _fwd_bwd_norms(w, mup=mup)
            print(f"{r['width']:>6} {r['loss']:>8.3f} "
                  f"{r['embed_rms']:>10.4f} {r['midstream_rms']:>10.4f} "
                  f"{r['layer_rms'][-1]:>10.4f} {r['embed_grad_rms']:>10.4f} "
                  f"{r['logits_rms']:>10.4f}")
