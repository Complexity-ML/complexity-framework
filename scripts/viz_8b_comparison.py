"""
Generate figures for 384M iso-params comparison (8B tokens).

Produces:
1. fig_384m_loss_curves.png — MoE vs Dense loss over 15,259 steps
2. fig_384m_loss_gap.png — Loss gap (MoE - Dense) with colored fill
3. fig_384m_avg_per_1000.png — Average loss per 1000-step bucket
4. fig_384m_ppl_curves.png — Perplexity comparison

Usage:
    python scripts/viz_8b_comparison.py \
        --moe training_moe.csv \
        --dense training_dense.csv \
        --output figures

Complexity-ML / INL — 2026
"""

import argparse
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_csv(path):
    """Load training CSV and return dict of lists."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {
        "step": [int(r["step"]) for r in rows],
        "loss": [float(r["loss"]) for r in rows],
        "ppl": [float(r["ppl"]) for r in rows],
    }


def smooth(values, window=50):
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def fig_loss_curves(moe, dense, output_dir):
    """Loss curves: MoE vs Dense over full training."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    max_step = min(max(moe["step"]), max(dense["step"]))
    moe_mask = [i for i, s in enumerate(moe["step"]) if s <= max_step]
    dense_mask = [i for i, s in enumerate(dense["step"]) if s <= max_step]

    moe_steps = [moe["step"][i] for i in moe_mask]
    moe_loss = smooth([moe["loss"][i] for i in moe_mask])
    dense_steps = [dense["step"][i] for i in dense_mask]
    dense_loss = smooth([dense["loss"][i] for i in dense_mask])

    # Trim steps to match smoothed length
    offset = len(moe_steps) - len(moe_loss)
    moe_steps_s = moe_steps[offset:]
    offset_d = len(dense_steps) - len(dense_loss)
    dense_steps_s = dense_steps[offset_d:]

    ax.plot(moe_steps_s, moe_loss, label="Token-Routed MoE (383.5M)", color="#2196F3", linewidth=1.5)
    ax.plot(dense_steps_s, dense_loss, label="Dense SwiGLU (384.5M)", color="#FF5722", linewidth=1.5)

    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("384M Iso-Params: Token-Routed MoE vs Dense (8B tokens)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_384m_loss_curves.png", dpi=200)
    plt.close(fig)
    print(f"Saved {output_dir / 'fig_384m_loss_curves.png'}")


def fig_loss_gap(moe, dense, output_dir):
    """Loss gap (MoE - Dense) with colored fill."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    max_step = min(max(moe["step"]), max(dense["step"]))

    # Build aligned gap
    moe_dict = dict(zip(moe["step"], moe["loss"]))
    dense_dict = dict(zip(dense["step"], dense["loss"]))

    common_steps = sorted(set(moe["step"]) & set(dense["step"]))
    common_steps = [s for s in common_steps if s <= max_step]

    gap = [moe_dict[s] - dense_dict[s] for s in common_steps]
    gap_smooth = smooth(gap, window=50)
    steps_smooth = common_steps[len(common_steps) - len(gap_smooth):]

    gap_arr = np.array(gap_smooth)
    steps_arr = np.array(steps_smooth)

    ax.plot(steps_arr, gap_arr, color="#1565C0", linewidth=1.5)
    ax.fill_between(steps_arr, gap_arr, 0,
                    where=gap_arr > 0, color="#FFCDD2", alpha=0.7, label="Dense wins")
    ax.fill_between(steps_arr, gap_arr, 0,
                    where=gap_arr <= 0, color="#C8E6C9", alpha=0.7, label="TR wins")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Loss Gap (TR - Dense)", fontsize=12)
    ax.set_title("Loss Gap (negative = TR wins)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_384m_loss_gap.png", dpi=200)
    plt.close(fig)
    print(f"Saved {output_dir / 'fig_384m_loss_gap.png'}")


def fig_avg_per_1000(moe, dense, output_dir):
    """Bar chart: average loss per 1000-step bucket."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    max_step = min(max(moe["step"]), max(dense["step"]))

    moe_dict = dict(zip(moe["step"], moe["loss"]))
    dense_dict = dict(zip(dense["step"], dense["loss"]))

    buckets = []
    moe_avgs = []
    dense_avgs = []

    start = 1
    while start <= max_step:
        end = start + 999
        moe_vals = [moe_dict[s] for s in range(start, min(end + 1, max_step + 1)) if s in moe_dict]
        dense_vals = [dense_dict[s] for s in range(start, min(end + 1, max_step + 1)) if s in dense_dict]
        if moe_vals and dense_vals:
            buckets.append(f"{start}-{end}")
            moe_avgs.append(sum(moe_vals) / len(moe_vals))
            dense_avgs.append(sum(dense_vals) / len(dense_vals))
        start += 1000

    x = np.arange(len(buckets))
    width = 0.35

    bars1 = ax.bar(x - width / 2, moe_avgs, width, label="Token-Routed MoE", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, dense_avgs, width, label="Dense SwiGLU", color="#FF5722", alpha=0.85)

    # Add gap labels on top
    for i in range(len(buckets)):
        gap = moe_avgs[i] - dense_avgs[i]
        y_pos = max(moe_avgs[i], dense_avgs[i]) + 0.02
        ax.text(x[i], y_pos, f"+{gap:.2f}", ha="center", fontsize=7, color="#555")

    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Average Loss", fontsize=12)
    ax.set_title("Average Loss per 1000 Steps (384M, 8B tokens)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_dir / "fig_384m_avg_per_1000.png", dpi=200)
    plt.close(fig)
    print(f"Saved {output_dir / 'fig_384m_avg_per_1000.png'}")


def fig_ppl_curves(moe, dense, output_dir):
    """Perplexity curves: MoE vs Dense."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    max_step = min(max(moe["step"]), max(dense["step"]))
    moe_mask = [i for i, s in enumerate(moe["step"]) if s <= max_step]
    dense_mask = [i for i, s in enumerate(dense["step"]) if s <= max_step]

    # Cap PPL for visualization
    cap = 200
    moe_ppl = smooth([min(moe["ppl"][i], cap) for i in moe_mask])
    dense_ppl = smooth([min(dense["ppl"][i], cap) for i in dense_mask])

    moe_steps = [moe["step"][i] for i in moe_mask]
    dense_steps = [dense["step"][i] for i in dense_mask]

    offset = len(moe_steps) - len(moe_ppl)
    moe_steps_s = moe_steps[offset:]
    offset_d = len(dense_steps) - len(dense_ppl)
    dense_steps_s = dense_steps[offset_d:]

    ax.plot(moe_steps_s, moe_ppl, label="Token-Routed MoE (383.5M)", color="#2196F3", linewidth=1.5)
    ax.plot(dense_steps_s, dense_ppl, label="Dense SwiGLU (384.5M)", color="#FF5722", linewidth=1.5)

    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title("384M Iso-Params: Perplexity Comparison (8B tokens)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_384m_ppl_curves.png", dpi=200)
    plt.close(fig)
    print(f"Saved {output_dir / 'fig_384m_ppl_curves.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate 384M comparison figures")
    parser.add_argument("--moe", type=str, required=True, help="MoE training CSV")
    parser.add_argument("--dense", type=str, required=True, help="Dense training CSV")
    parser.add_argument("--output", type=str, default="figures", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSVs...")
    moe = load_csv(args.moe)
    dense = load_csv(args.dense)
    print(f"MoE: {max(moe['step'])} steps, Dense: {max(dense['step'])} steps")

    fig_loss_curves(moe, dense, output_dir)
    fig_loss_gap(moe, dense, output_dir)
    fig_avg_per_1000(moe, dense, output_dir)
    fig_ppl_curves(moe, dense, output_dir)

    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
