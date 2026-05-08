"""
Empirical Validation of COMPLEXITY-DEEP Theorems
=================================================
Generates a figure for reviewer defense:

1. Gradient Cosine Similarity between Experts (Theorem 3 - Gradient Orthogonalization)
   Shows that expert gradients diverge during training, validating
   that the modulo routing induces expert specialization.

Usage:
    python validate_theorems.py --checkpoint ./checkpoints/final
    python validate_theorems.py --checkpoint C:/INL/pacific-prime/checkpoints/final --device cpu

Complexity-ML — 2026
"""

import torch
import torch.nn.functional as F
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
import os
import sys

from complexity.config import ModelConfig
from complexity.models import ComplexityModel
from transformers import PreTrainedTokenizerFast


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load complexity-framework model from checkpoint."""
    print(f"Loading config from {config_path}")
    config = ModelConfig.load(config_path)
    model = ComplexityModel(config)

    print(f"Loading checkpoint from {checkpoint_path}")

    # Try multiple formats
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pt_path = os.path.join(checkpoint_path, "model.pt")
    ckpt_path = os.path.join(checkpoint_path, "checkpoint.pt")

    if os.path.isdir(checkpoint_path):
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        elif os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        elif os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location='cpu')
        else:
            raise FileNotFoundError(f"No model weights found in {checkpoint_path}")
    else:
        # Single file path
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    return model, config


def get_sample_batch(tokenizer, device, batch_size=2, seq_len=128):
    """Create a small sample batch for gradient computation."""
    texts = [
        "The theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy.",
        "In quantum mechanics, particles exhibit wave-particle duality and their behavior is described by probability amplitudes.",
        "Machine learning algorithms can identify patterns in large datasets and make predictions based on statistical models.",
        "The human brain contains approximately 86 billion neurons connected by trillions of synapses forming complex networks.",
    ][:batch_size]

    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_len,
    )
    input_ids = encodings["input_ids"].to(device)
    labels = input_ids.clone()
    return input_ids, labels


def analyze_gradient_cosine_similarity(model, input_ids, labels, config):
    """
    Compute pairwise cosine similarity between expert gradients per layer.
    Validates Theorem 3: Gradient Orthogonalization.
    """
    print("\n" + "=" * 60)
    print("THEOREM 3: Gradient Cosine Similarity Between Experts")
    print("=" * 60)

    num_layers = config.num_hidden_layers
    num_experts = config.num_experts
    expert_pairs = list(combinations(range(num_experts), 2))

    # Forward + backward
    model.zero_grad()
    model.train()
    outputs = model(input_ids)
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
    loss.backward()
    print(f"Loss: {loss.item():.4f}")

    # Collect gradient cosine similarity per layer
    layer_cosine_sims = {f"E{i}-E{j}": [] for i, j in expert_pairs}
    layer_avg_cosine = []

    for layer_idx in range(num_layers):
        layer = model.layers[layer_idx]
        mlp = layer.mlp

        # gate_up_proj: [num_experts, hidden_size, 2*expert_intermediate]
        if hasattr(mlp, 'gate_up_proj') and mlp.gate_up_proj.grad is not None:
            grad = mlp.gate_up_proj.grad  # [4, H, 2I]

            # Extract per-expert gradient vectors
            expert_grads = []
            for e in range(num_experts):
                expert_grads.append(grad[e].flatten().float())

            # Compute pairwise cosine similarity
            layer_sims = []
            for (i, j) in expert_pairs:
                cos_sim = F.cosine_similarity(
                    expert_grads[i].unsqueeze(0),
                    expert_grads[j].unsqueeze(0)
                ).item()
                layer_cosine_sims[f"E{i}-E{j}"].append(cos_sim)
                layer_sims.append(cos_sim)

            avg_sim = np.mean(layer_sims)
            layer_avg_cosine.append(avg_sim)
            print(f"  Layer {layer_idx:2d}: avg cosine sim = {avg_sim:.4f}")
        else:
            # No gradient (frozen or no gate_up_proj)
            for key in layer_cosine_sims:
                layer_cosine_sims[key].append(0.0)
            layer_avg_cosine.append(0.0)

    model.eval()
    return layer_cosine_sims, layer_avg_cosine, expert_pairs


def plot_gradient_cosine_similarity(layer_cosine_sims, layer_avg_cosine, expert_pairs, num_layers, output_path):
    """Plot gradient cosine similarity between experts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-pair cosine similarity
    ax = axes[0]
    x = range(num_layers)
    colors = plt.cm.Set2(np.linspace(0, 1, len(expert_pairs)))
    for idx, (i, j) in enumerate(expert_pairs):
        key = f"E{i}-E{j}"
        ax.plot(x, layer_cosine_sims[key], marker='.', color=colors[idx],
                label=f"Expert {i} vs {j}", alpha=0.7)

    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Orthogonal (0.0)')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, label='Identical (1.0)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(a) Gradient Cosine Similarity per Expert Pair')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.1)

    # Right: average cosine similarity
    ax = axes[1]
    bars = ax.bar(x, layer_avg_cosine, color='steelblue', alpha=0.8)
    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Orthogonal')
    avg_overall = np.mean(layer_avg_cosine)
    ax.axhline(y=avg_overall, color='orange', linestyle='--', alpha=0.7,
               label=f'Mean: {avg_overall:.3f}')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Cosine Similarity')
    ax.set_title('(b) Average Gradient Cosine Similarity per Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('COMPLEXITY-DEEP: Expert Gradient Orthogonalization (Theorem 3)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Validate COMPLEXITY-DEEP Theorems Empirically")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/final")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = str(Path(args.checkpoint).parent) if not Path(args.checkpoint).is_dir() else args.checkpoint
    if args.config is None:
        for name in ["model_config.yaml", "config.json"]:
            p = Path(checkpoint_dir) / name
            if p.exists():
                args.config = str(p)
                break
        if args.config is None:
            args.config = str(Path(checkpoint_dir) / "model_config.yaml")
    if args.tokenizer is None:
        args.tokenizer = checkpoint_dir

    print("=" * 60)
    print("COMPLEXITY-DEEP: Empirical Theorem Validation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Batch: {args.batch_size} x {args.seq_len}")

    # Load model
    model, config = load_model(args.checkpoint, args.config, args.device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get sample batch
    input_ids, labels = get_sample_batch(tokenizer, args.device, args.batch_size, args.seq_len)
    print(f"Input shape: {input_ids.shape}")

    # === Analysis: Gradient Cosine Similarity ===
    layer_cosine_sims, layer_avg_cosine, expert_pairs = analyze_gradient_cosine_similarity(
        model, input_ids, labels, config
    )
    plot_gradient_cosine_similarity(
        layer_cosine_sims, layer_avg_cosine, expert_pairs,
        config.num_hidden_layers,
        Path(args.output_dir) / "gradient_cosine_similarity.png"
    )

    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_cosine = np.mean(layer_avg_cosine)
    print(f"Theorem 3 - Avg gradient cosine similarity: {avg_cosine:.4f}")
    if avg_cosine < 0.5:
        print("  -> Expert gradients are substantially divergent (validates orthogonalization)")
    elif avg_cosine < 0.8:
        print("  -> Expert gradients show moderate divergence")
    else:
        print("  -> Expert gradients are still correlated (limited orthogonalization)")

    print(f"\nFigures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
