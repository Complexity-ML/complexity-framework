"""
Review script — multimodal stack.

Instantiates every encoder + OmniModel, runs a forward pass,
and logs shapes + param counts. No training, no data loading.

Usage:
    python scripts/review_multimodal.py
    python scripts/review_multimodal.py --device cuda
    python scripts/review_multimodal.py --hidden 512 --vocab 16000
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from complexity.multimodal.vision import VisionEncoder
from complexity.multimodal.audio import AudioEncoder
from complexity.multimodal.video import VideoEncoder
from complexity.multimodal.omni import OmniModel, OmniConfig, Modality, ModalityMLPConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def n_params(m: torch.nn.Module) -> str:
    return fmt(sum(p.numel() for p in m.parameters()))


def sep(title: str = "") -> None:
    bar = "=" * 60
    if title:
        log.info("%s", bar)
        log.info("  %s", title)
        log.info("%s", bar)
    else:
        log.info("%s", bar)


# ── review functions ──────────────────────────────────────────────────────────

def review_modality_enum() -> None:
    sep("Modality enum")
    for m in Modality:
        log.info("  %-8s  value=%d  (auto, 0-based)", m.name, int(m))
    log.info("")
    log.info("  Total modalities : %d", Modality.count())
    log.info("  Values           : %s", [int(m) for m in Modality])


def review_vision(device: torch.device, H: int, num_heads: int) -> None:
    sep("VisionEncoder")
    enc = VisionEncoder(image_size=224, patch_size=16, hidden_size=H, num_heads=num_heads, num_experts=4).to(device)
    images = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out = enc(images)
    S = (224 // 16) ** 2
    log.info("  params      : %s", n_params(enc))
    log.info("  input       : %s", list(images.shape))
    log.info("  output      : %s", list(out.shape))
    log.info("  num_patches : %d  (+1 cls = %d)", S, S + 1)


def review_audio(device: torch.device, H: int, num_heads: int) -> None:
    sep("AudioEncoder")
    enc = AudioEncoder(n_mels=80, hidden_size=H, num_heads=num_heads, num_experts=4).to(device)
    mel = torch.randn(2, 80, 300, device=device)
    with torch.no_grad():
        out = enc(mel)
    log.info("  params  : %s", n_params(enc))
    log.info("  input   : %s  [B, n_mels, T]", list(mel.shape))
    log.info("  output  : %s", list(out.shape))


def review_video(device: torch.device, H: int, num_heads: int) -> None:
    sep("VideoEncoder (ViViT factorised)")
    enc = VideoEncoder(
        image_size=64, patch_size=16, num_frames=4, hidden_size=H, num_heads=num_heads, num_experts=4,
    ).to(device)
    video = torch.randn(2, 3, 4, 64, 64, device=device)
    with torch.no_grad():
        out = enc(video)
    S = (64 // 16) ** 2
    T = 4
    log.info("  params        : %s", n_params(enc))
    log.info("  input         : %s  [B, C, T, H, W]", list(video.shape))
    log.info("  output        : %s", list(out.shape))
    log.info("  spatial=%d  temporal=%d  total=%d", S, T, S * T)


def review_omni(device: torch.device, H: int, num_heads: int, vocab: int) -> None:
    sep("OmniModel — any-to-any")

    cfg = OmniConfig(
        hidden_size=H,
        vocab_size=vocab,
        num_hidden_layers=2,
        num_attention_heads=num_heads,
        general_num_experts=8,
        general_intermediate_size=H * 4,
        modality_mlp={m: ModalityMLPConfig(num_experts=4, intermediate_size=H * 4) for m in Modality},
        # small encoders for speed
        image_size=64,
        patch_size=16,
        vision_hidden_size=H,
        vision_num_heads=num_heads,
        vision_num_layers=2,
        audio_hidden_size=H,
        audio_num_heads=num_heads,
        audio_num_layers=2,
        num_frames=4,
        video_hidden_size=H,
        video_num_heads=num_heads,
        video_num_layers=2,
    )
    model = OmniModel(cfg).to(device)

    B = 2
    with torch.no_grad():
        out = model(
            text_ids=torch.randint(0, vocab, (B, 16), device=device),
            pixel_values=torch.randn(B, 3, 64, 64, device=device),
            audio_features=torch.randn(B, 80, 100, device=device),
            video_frames=torch.randn(B, 3, 4, 64, 64, device=device),
        )

    log.info("  params            : %s", n_params(model))
    log.info("  logits            : %s", list(out["logits"].shape))
    log.info("  last_hidden_state : %s", list(out["last_hidden_state"].shape))
    log.info("")
    log.info("  Modality dispatch (auto() enum):")
    for m in Modality:
        mcfg = cfg.modality_mlp[m]
        log.info("    %-8s  value=%d  experts=%d  intermediate=%d",
                 m.name, int(m), mcfg.num_experts, mcfg.intermediate_size)
    log.info("")
    log.info("  General MLP:")
    log.info("    experts=%d  intermediate=%d  block_size=%s",
             cfg.general_num_experts, cfg.general_intermediate_size,
             cfg.general_block_size or "auto")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Review multimodal stack")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden", type=int, default=256,
                        help="hidden_size (default 256 for speed)")
    parser.add_argument("--vocab", type=int, default=32000)
    args = parser.parse_args()

    device = torch.device(args.device)
    H = args.hidden
    # num_heads must divide hidden_size — pick largest power-of-2 ≤ H//32, capped at 16
    num_heads = 1
    for k in [16, 8, 4, 2, 1]:
        if H % k == 0 and H // k >= 32:
            num_heads = k
            break

    log.info("device=%s  hidden=%d  num_heads=%d  vocab=%d", device, H, num_heads, args.vocab)

    review_modality_enum()
    review_vision(device, H, num_heads)
    review_audio(device, H, num_heads)
    review_video(device, H, num_heads)
    review_omni(device, H, num_heads, args.vocab)

    sep()
    log.info("  All checks passed.")
    sep()


if __name__ == "__main__":
    main()
