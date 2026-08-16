#!/usr/bin/env python3
"""Generate PNG images from a trained TR-Hash text-to-image checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from tokenizers import Tokenizer

from complexity.generative.image.codec import FrozenAutoencoderKL
from complexity.generative.image.config import TRHashImageConfig
from complexity.generative.image.model import TRHashTextToImage
from complexity.tr_hash import TRHashBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"))
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--backend",
        choices=tuple(backend.value for backend in TRHashBackend),
        default=TRHashBackend.PYTORCH.value,
        help="Use PyTorch by default for numerically robust one-image sampling.",
    )
    return parser.parse_args()


def tokenize_prompt(
    tokenizer: Tokenizer,
    prompt: str,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer.encode(prompt)
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0
    values = encoded.ids[:max_length] or [int(pad_id)]
    ids = torch.tensor([values], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    return ids, mask


def save_png(pixels: torch.Tensor, path: Path) -> None:
    image = (
        pixels.detach()
        .float()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .add(1.0)
        .mul(127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )
    Image.fromarray(image, mode="RGB").save(path)


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    config = TRHashImageConfig.from_dict(
        json.loads((args.checkpoint / "config.json").read_text())
    )
    model = TRHashTextToImage(config).to(device=device, dtype=dtype)
    state = load_file(str(args.checkpoint / "model.safetensors"), device="cpu")
    model.load_state_dict(state)
    backend = TRHashBackend(args.backend)
    for block in model.blocks:
        block.mlp.config = replace(block.mlp.config, backend=backend)
    model.eval()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    codec = FrozenAutoencoderKL(args.vae).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    manifest: list[dict[str, object]] = []
    for index, prompt in enumerate(args.prompt):
        caption_ids, caption_mask = tokenize_prompt(
            tokenizer, prompt, config.max_text_length, device
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=dtype == torch.bfloat16 and device.type == "cuda",
        ):
            latents = model.sample(
                caption_ids,
                caption_mask,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            pixels = codec.decode(latents)
        target = args.output_dir / f"sample-{index:02d}.png"
        save_png(pixels, target)
        manifest.append(
            {
                "file": target.name,
                "prompt": prompt,
                "seed": args.seed,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
            }
        )
        print(f"saved {target}")

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
