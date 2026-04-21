"""
Plot loss / PPL trajectories side-by-side for the Hadamard-vs-Kaiming
70M ablation.

Expects the two runs produced by `train_70m_hadamard_ablation.py`:
    runs/swiglu/metrics.csv           (PyTorch default Kaiming)
    runs/dense_deterministic/metrics.csv  (Hadamard init)

Usage:
    python scripts/plot_hadamard_ablation.py

Outputs `runs/hadamard_ablation.png` with loss + PPL subplots.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_run(csv_path: Path) -> tuple[list[int], list[float], list[float]]:
    """Parse step / loss / ppl columns from a metrics.csv."""
    steps, losses, ppls = [], [], []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
                ppls.append(float(row["ppl"]))
            except (KeyError, ValueError):
                continue
    return steps, losses, ppls


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Hadamard vs default init")
    parser.add_argument("--kaiming", type=Path, default=Path("runs/swiglu/metrics.csv"))
    parser.add_argument("--hadamard", type=Path,
                        default=Path("runs/dense_deterministic/metrics.csv"))
    parser.add_argument("--out", type=Path, default=Path("runs/hadamard_ablation.png"))
    args = parser.parse_args()

    runs = []
    for label, path, color in [
        ("Kaiming (PyTorch default)", args.kaiming, "tab:blue"),
        ("Hadamard (deterministic)",  args.hadamard, "tab:orange"),
    ]:
        if not path.exists():
            print(f"[skip] missing {path} — run the ablation first")
            continue
        runs.append((label, *load_run(path), color))

    if len(runs) < 2:
        raise SystemExit("Need both metrics.csv files to compare.")

    fig, (ax_loss, ax_ppl) = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, steps, losses, ppls, color in runs:
        ax_loss.plot(steps, losses, label=label, color=color, lw=1.5)
        ax_ppl.plot(steps, ppls, label=label, color=color, lw=1.5)

    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss (cross-entropy)")
    ax_loss.set_title("Training loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="upper right", fontsize=9)

    ax_ppl.set_xlabel("step")
    ax_ppl.set_ylabel("perplexity")
    ax_ppl.set_yscale("log")
    ax_ppl.set_title("Perplexity (log scale)")
    ax_ppl.grid(True, alpha=0.3, which="both")
    ax_ppl.legend(loc="upper right", fontsize=9)

    fig.suptitle("70M Dense — Hadamard vs Kaiming init", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved: {args.out}")

    # Quick numeric summary.
    print("\nFinal step summary:")
    for label, steps, losses, ppls, _ in runs:
        if steps:
            print(f"  {label:32s}  step={steps[-1]:5d}  loss={losses[-1]:.4f}  ppl={ppls[-1]:.2f}")


if __name__ == "__main__":
    main()
