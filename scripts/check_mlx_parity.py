#!/usr/bin/env python
"""Logit parity check: torch ComplexityModel vs the mlx-lm port, same checkpoint."""

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/Users/boris/Dev/mlx-lm")

from complexity.config import ModelConfig
from complexity.models import ComplexityModel
from mlx_lm.models.complexity import Model, ModelArgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mlx-dir", required=True)
    args = ap.parse_args()

    cfg_json = json.loads((Path(args.mlx_dir) / "config.json").read_text())
    pw = cfg_json["top_k_primary_weight"]

    # --- torch ---
    d = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw = dict(d["config"])
    valid = {f.name for f in fields(ModelConfig)}
    tcfg = {k: v for k, v in raw.items() if k in valid}
    tm = ComplexityModel(ModelConfig(**tcfg))
    missing, unexpected = tm.load_state_dict(d["model"], strict=False)
    tm.eval()
    # match the (non-persisted) scheduled primary route weight used by the MLX config
    for m in tm.modules():
        if hasattr(m, "_primary_weight"):
            m._primary_weight = float(pw)

    ids = list(range(1, 17))
    with torch.no_grad():
        tl = tm(torch.tensor([ids]), return_logits=True)["logits"][0].float().numpy()

    # --- mlx ---
    margs = ModelArgs(**cfg_json)
    mm = Model(margs)
    w = mx.load(str(Path(args.mlx_dir) / "model.safetensors"))
    mm.load_weights(list(w.items()))
    mm.eval()
    ml = np.array(mm(mx.array([ids]), cache=None))[0]

    diff = np.abs(tl - ml)
    print(f"torch missing={len(missing)} unexpected={len(unexpected)}")
    print(f"logits shape torch={tl.shape} mlx={ml.shape}")
    print(f"max abs diff : {diff.max():.6f}")
    print(f"mean abs diff: {diff.mean():.6f}")
    for pos in (0, len(ids) - 1):
        t5 = tl[pos].argsort()[-5:][::-1]
        m5 = ml[pos].argsort()[-5:][::-1]
        print(f"pos {pos:2d} top5 torch={list(t5)} mlx={list(m5)} match={list(t5) == list(m5)}")


if __name__ == "__main__":
    main()
