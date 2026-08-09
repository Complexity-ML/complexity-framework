#!/usr/bin/env python3
"""Edit an image with an instruction-guided TR-Hash checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from tokenizers import Tokenizer

from complexity.generative.image import (
    FrozenAutoencoderKL,
    TRHashImageEditConfig,
    TRHashImageEditor,
)
from complexity.generative.image.edit_data import image_payload_to_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"))
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--output", type=Path, default=Path("edited.png"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--image-guidance", type=float, default=1.5)
    parser.add_argument("--text-guidance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    config = TRHashImageEditConfig.from_dict(
        json.loads((args.checkpoint / "config.json").read_text())
    )
    model = TRHashImageEditor(config).to(device)
    model.load_state_dict(load_file(str(args.checkpoint / "model.safetensors")))
    model.eval()
    codec = FrozenAutoencoderKL(args.vae).to(device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    source_pixels = (
        image_payload_to_tensor(
            args.source.read_bytes(),
            config.image_size,
        )
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        source_latents = codec.encode(source_pixels)
    ids = tokenizer.encode(args.instruction).ids[: config.max_text_length]
    if not ids:
        ids = [0]
    instruction_ids = torch.tensor([ids], dtype=torch.long, device=device)
    instruction_mask = torch.ones_like(instruction_ids, dtype=torch.bool)
    if device.type == "mps":
        torch.manual_seed(args.seed)
        generator = None
    else:
        generator = torch.Generator(device=device.type).manual_seed(args.seed)
    edited_latents = model.edit(
        source_latents,
        instruction_ids,
        instruction_mask,
        steps=args.steps,
        image_guidance_scale=args.image_guidance,
        text_guidance_scale=args.text_guidance,
        generator=generator,
    )
    pixels = codec.decode(edited_latents)[0].add(1).mul(127.5).round().clamp(0, 255).byte()
    array = pixels.permute(1, 2, 0).cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
