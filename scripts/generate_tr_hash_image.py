#!/usr/bin/env python3
"""Generate a 256×256 image from a trained TR-Hash latent-flow checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from tokenizers import Tokenizer

from complexity.generative.image import FrozenAutoencoderKL, TRHashImageConfig, TRHashTextToImage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"))
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("sample.png"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    config = TRHashImageConfig.from_dict(
        json.loads((args.checkpoint / "config.json").read_text())
    )
    model = TRHashTextToImage(config).to(device)
    model.load_state_dict(load_file(str(args.checkpoint / "model.safetensors")))
    model.eval()
    codec = FrozenAutoencoderKL(args.vae).to(device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    ids = tokenizer.encode(args.prompt).ids[: config.max_text_length]
    if not ids:
        ids = [0]
    caption_ids = torch.tensor([ids], dtype=torch.long, device=device)
    caption_mask = torch.ones_like(caption_ids, dtype=torch.bool)
    if device.type == "mps":
        torch.manual_seed(args.seed)
        generator = None
    else:
        generator = torch.Generator(device=device.type).manual_seed(args.seed)
    latents = model.sample(
        caption_ids,
        caption_mask,
        steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )
    pixels = codec.decode(latents)[0].add(1).mul(127.5).round().clamp(0, 255).byte()
    array = pixels.permute(1, 2, 0).cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
