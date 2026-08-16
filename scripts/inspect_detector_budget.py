#!/usr/bin/env python3
"""Report the stored parameter and executed forward-compute budget of a preset."""

from __future__ import annotations

import argparse
import json

import torch

from complexity.generative.detection import (
    COCO_V8_NANO_NAME,
    TRHashObjectDetector,
    coco_v8_nano_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=(COCO_V8_NANO_NAME,), default=COCO_V8_NANO_NAME)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-flops", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = coco_v8_nano_config()
    model = TRHashObjectDetector(config).eval()
    pixel_values = torch.zeros(args.batch_size, 3, config.image_size, config.image_size)

    forward_flops = None
    with torch.inference_mode():
        if args.skip_flops:
            raw = model(pixel_values)
        else:
            from torch.utils.flop_counter import FlopCounterMode

            with FlopCounterMode(display=False) as counter:
                raw = model(pixel_values)
            forward_flops = counter.get_total_flops()

    total_parameters = model.num_parameters()
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    report = {
        "preset": args.preset,
        "architecture_version": config.architecture_version,
        "image_size": config.image_size,
        "batch_size": args.batch_size,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameters_millions": total_parameters / 1e6,
        "forward_flops": forward_flops,
        "forward_gflops": None if forward_flops is None else forward_flops / 1e9,
        "grid_sizes": config.grid_sizes,
        "raw_output_shape": tuple(raw.shape),
        "num_experts": config.vision_num_experts,
        "top_k": config.vision_top_k,
        "branch": "o2m-only" if model.one_to_one_head is None else "o2m+o2o",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
