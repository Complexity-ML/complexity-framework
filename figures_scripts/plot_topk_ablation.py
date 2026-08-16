"""
Generate the two figures for the Top-K Ablation section of the paper.

Inputs (expected CSV paths, can be overridden via CLI):
  - MoE top-1 : ~/Dev/checkpoints/abl-moe-adamw/abl-moe-adamw.csv
  - MoE top-2 : ~/Dev/checkpoints/abl-moe-topk2/abl-moe-topk2.csv
  - Dense     : ~/Dev/checkpoints/abl-dense-adamw/abl-dense-adamw.csv

Outputs:
  - fig_384m_topk_curves.png : 3-way loss curves (EMA smoothed)
  - fig_384m_topk_gap.png    : (MoE − Dense) gap curves for K=1,2

Usage:
    python3 figures_scripts/plot_topk_ablation.py [--out-dir figures]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_loss(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps, losses = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            steps.append(int(r["step"]))
            losses.append(float(r["loss"]))
    return np.array(steps), np.array(losses)


def ema(x: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top1", default=Path.home() / "Dev/checkpoints/abl-moe-adamw/abl-moe-adamw.csv")
    ap.add_argument("--top2", default=Path.home() / "Dev/checkpoints/abl-moe-topk2/abl-moe-topk2.csv")
    ap.add_argument("--dense", default=Path.home() / "Dev/checkpoints/abl-dense-adamw/abl-dense-adamw.csv")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    s1, l1 = load_loss(Path(args.top1))
    s2, l2 = load_loss(Path(args.top2))
    sd, ld = load_loss(Path(args.dense))

    l1s, l2s, lds = ema(l1), ema(l2), ema(ld)

    # ── Figure 1 : 3-way loss curves ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(sd, lds, label="Dense (SwiGLU, 384.5M)", linewidth=2, color="#444")
    ax.plot(s1, l1s, label="MoE top-1 deterministic (383.5M)", linewidth=2, color="#1f77b4")
    ax.plot(s2, l2s, label="MoE top-2 deterministic (primary 0.95)", linewidth=2, color="#d62728")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss (EMA smoothed)")
    ax.set_title("384M iso-params: top-1 vs top-2 vs dense (8B tokens FineWeb-Edu)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out / "fig_384m_topk_curves.png", dpi=150)
    plt.close()
    print(f"✓ {out / 'fig_384m_topk_curves.png'}")

    # ── Figure 2 : gap vs dense curves ──────────────────────────────────
    common_end = min(s1[-1], s2[-1], sd[-1])

    def align_to(steps, vals, target_steps):
        d = dict(zip(steps, vals))
        return np.array([d.get(int(s), np.nan) for s in target_steps])

    target = np.arange(1, common_end + 1)
    d_aligned = align_to(sd, ld, target)
    gap1 = align_to(s1, l1, target) - d_aligned
    gap2 = align_to(s2, l2, target) - d_aligned

    # EMA-smooth each gap
    mask = ~np.isnan(gap1) & ~np.isnan(gap2)
    target = target[mask]
    gap1 = ema(gap1[mask])
    gap2 = ema(gap2[mask])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="Dense baseline")
    ax.plot(target, gap1, label="MoE top-1 − Dense", linewidth=2, color="#1f77b4")
    ax.plot(target, gap2, label="MoE top-2 − Dense", linewidth=2, color="#d62728")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss gap (MoE − Dense)")
    ax.set_title("384M iso-params: loss gap vs dense (top-1 and top-2 plateau near +0.06)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out / "fig_384m_topk_gap.png", dpi=150)
    plt.close()
    print(f"✓ {out / 'fig_384m_topk_gap.png'}")


if __name__ == "__main__":
    main()
