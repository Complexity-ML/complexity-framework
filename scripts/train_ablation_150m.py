"""
Ablation Study — 4 × 150M models, 32k vocab, 2B tokens, PiD in all.

Run 1: Dense float      — SwiGLU standard, absolute reference
Run 2: Full archi float  — Token-Routed + Mu + INL Dynamics
Run 3: Full archi no-Mu  — Token-Routed + INL Dynamics, Mu disabled
Run 4: Full archi INL    — Integer-first (i64), same archi as Run 2

Progressive increasing Depth (PiD): start with L/2 active layers,
add one layer every (total_steps / (L/2)) steps until all layers active.

Usage:
    python scripts/train_ablation_150m.py --run 1           # Dense baseline
    python scripts/train_ablation_150m.py --run 2           # Full archi
    python scripts/train_ablation_150m.py --run 3           # No Mu ablation
    python scripts/train_ablation_150m.py --run 4           # INL integer-first
    python scripts/train_ablation_150m.py --run all         # All 4 sequentially
    python scripts/train_ablation_150m.py --run 2 --resume checkpoints/run2-full/step_50000

INL - 2025
"""

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader, IterableDataset
import torch
import torch.nn as nn
import argparse
import os
import math

from complexity.config import ModelConfig
from complexity.models import ComplexityModel
from complexity.training import Trainer, TrainingConfig


# ── Architecture configs (all ~150M, 32k vocab) ──────────────────────────

def make_config_run1() -> ModelConfig:
    """Run 1: Dense float — SwiGLU, no routing, no dynamics."""
    return ModelConfig(
        hidden_size=768,
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=2048,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="swiglu",
        num_experts=1,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_inl_dynamics=False,
    )


def make_config_run2() -> ModelConfig:
    """Run 2: Full archi float — Token-Routed + Mu-Guidance + INL Dynamics."""
    return ModelConfig(
        hidden_size=768,
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=2048,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="token_routed",
        num_experts=4,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_inl_dynamics=True,
        inl_beta_max=2.0,
        inl_velocity_max=10.0,
    )


def make_config_run3() -> ModelConfig:
    """Run 3: Full archi float sans Mu — Token-Routed + INL Dynamics, Mu disabled."""
    config = make_config_run2()
    config.extra_config["disable_mu_guidance"] = True
    return config


def make_config_run4() -> ModelConfig:
    """Run 4: Full archi INL — Integer-first (i64), same structure as Run 2."""
    return ModelConfig(
        hidden_size=768,
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=2048,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="i64",
        mlp_type="i64_swiglu",
        num_experts=1,
        norm_type="i64_rmsnorm",
        use_qk_norm=True,
        use_inl_dynamics=True,
        inl_beta_max=2.0,
        inl_velocity_max=10.0,
    )


RUN_CONFIGS = {
    1: ("run1-dense",   "Dense float (SwiGLU baseline)",        make_config_run1),
    2: ("run2-full",    "Full archi (Token-Routed + Mu + PiD)",  make_config_run2),
    3: ("run3-no-mu",   "Full archi sans Mu-Guidance",           make_config_run3),
    4: ("run4-inl",     "Full archi INL (integer-first)",        make_config_run4),
}


# ── PiD: Progressive increasing Depth ────────────────────────────────────

class ProgressiveDepth:
    """
    PiD — start with L/2 active layers, progressively activate the rest.

    Strategy: layers are activated from bottom to top. Inactive layers are
    frozen with identity pass-through (residual only, no computation).
    A new layer is unfrozen every `total_steps / (L - L_init)` steps.
    """

    def __init__(self, model: ComplexityModel, total_steps: int):
        self.model = model
        self.total_steps = total_steps
        self.num_layers = len(model.layers)
        self.initial_layers = self.num_layers // 2
        self.active_layers = self.initial_layers

        # Steps between each new layer activation
        layers_to_add = self.num_layers - self.initial_layers
        self.steps_per_layer = total_steps // (layers_to_add + 1)

        # Freeze upper layers initially
        self._update_frozen()
        print(f"PiD: {self.initial_layers}/{self.num_layers} layers active, "
              f"new layer every {self.steps_per_layer:,} steps")

    def _update_frozen(self):
        """Freeze/unfreeze layers based on current active count."""
        for i, layer in enumerate(self.model.layers):
            if i < self.active_layers:
                for p in layer.parameters():
                    p.requires_grad = True
            else:
                for p in layer.parameters():
                    p.requires_grad = False

    def step(self, global_step: int):
        """Call each training step — activates new layers when due."""
        target_active = min(
            self.num_layers,
            self.initial_layers + global_step // self.steps_per_layer
        )
        if target_active > self.active_layers:
            self.active_layers = target_active
            self._update_frozen()
            print(f"PiD: activated layer {self.active_layers}/{self.num_layers} "
                  f"at step {global_step:,}")


# ── Mu-Guidance disabler (for Run 3 ablation) ────────────────────────────

def disable_mu_propagation(model: ComplexityModel):
    """
    Disable mu guidance by monkey-patching the forward to skip mu propagation.
    INL Dynamics still runs (velocity tracking), but mu is NOT passed between layers.
    """
    original_forward = model.forward

    def forward_no_mu(input_ids, **kwargs):
        # Run normal forward but intercept mu flow
        batch_size, seq_len = input_ids.shape
        hidden_states = model.embed_tokens(input_ids)

        new_past_key_values = [] if kwargs.get("use_cache", False) else None
        new_velocity_states = [] if model.config.use_inl_dynamics else None
        past_key_values = kwargs.get("past_key_values", None)
        velocity_states = kwargs.get("velocity_states", None)

        for i, layer in enumerate(model.layers):
            past_kv = past_key_values[i] if past_key_values else None
            vel_state = velocity_states[i] if velocity_states else None

            hidden_states, new_kv, new_vel, _ = layer(
                hidden_states,
                attention_mask=kwargs.get("attention_mask"),
                past_key_value=past_kv,
                use_cache=kwargs.get("use_cache", False),
                token_ids=input_ids,
                velocity_state=vel_state,
                mu_prev=None,  # <-- Mu ALWAYS None = disabled
            )

            if new_past_key_values is not None:
                new_past_key_values.append(new_kv)
            if new_velocity_states is not None:
                new_velocity_states.append(new_vel)

        hidden_states = model.norm(hidden_states)
        if model.lm_head is not None:
            logits = model.lm_head(hidden_states)
        else:
            logits = torch.matmul(hidden_states, model.embed_tokens.weight.T)

        return {
            "logits": logits,
            "past_key_values": new_past_key_values,
            "hidden_states": None,
            "last_hidden_state": hidden_states,
            "velocity_states": new_velocity_states,
        }

    model.forward = forward_no_mu
    print("Mu-Guidance DISABLED (ablation mode)")
    return model


# ── Dataset ───────────────────────────────────────────────────────────────

class FineWebStreamingDataset(IterableDataset):
    """Streaming tokenized chunks from FineWeb-Edu — 2B tokens."""

    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
        )

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)

            while len(buffer) >= self.max_length + 1:
                chunk = buffer[:self.max_length + 1]
                buffer = buffer[self.max_length:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


# ── Training ──────────────────────────────────────────────────────────────

def compute_steps_for_tokens(target_tokens: int, batch_size: int,
                              grad_accum: int, seq_len: int) -> int:
    """Compute training steps to reach target token count."""
    tokens_per_step = batch_size * grad_accum * seq_len
    return math.ceil(target_tokens / tokens_per_step)


def train_run(run_id: int, args):
    """Train a single run."""
    name, desc, config_fn = RUN_CONFIGS[run_id]
    print(f"\n{'='*70}")
    print(f"  Run {run_id}: {desc}")
    print(f"  Output: {args.checkpoint_dir}/{name}")
    print(f"{'='*70}\n")

    # Tokenizer
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(
            f"Tokenizer not found: {args.tokenizer}\n"
            f"Train one first or point to an existing HF tokenizer directory."
        )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    # Model
    config = config_fn()
    config.vocab_size = min(len(tokenizer), 32000)  # Cap at 32k
    model = ComplexityModel(config)
    print(f"Model: {model.num_parameters():,} params "
          f"({model.num_parameters()/1e6:.1f}M)")
    print(f"  hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}")
    print(f"  mlp={config.mlp_type}, experts={config.num_experts}, "
          f"dynamics={config.use_inl_dynamics}")

    # Disable mu for Run 3
    if run_id == 3:
        model = disable_mu_propagation(model)

    # Compute steps for 2B tokens
    max_steps = compute_steps_for_tokens(
        target_tokens=args.target_tokens,
        batch_size=args.batch_size,
        grad_accum=args.gradient_accumulation,
        seq_len=2048,
    )
    print(f"  Training for {max_steps:,} steps "
          f"(~{args.target_tokens/1e9:.1f}B tokens)")

    # PiD — Progressive increasing Depth
    pid = ProgressiveDepth(model, total_steps=max_steps)

    # Dataset
    dataset = FineWebStreamingDataset(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Trainer
    checkpoint_dir = os.path.join(args.checkpoint_dir, name)
    train_config = TrainingConfig(
        max_steps=max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        precision="bf16",
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        checkpoint_dir=checkpoint_dir,
        resume_from=args.resume if run_id == args.resume_run else None,
    )

    trainer = Trainer(model=model, config=train_config, train_dataloader=dataloader)

    # Override loss to handle dict output from ComplexityModel
    def compute_loss(model, batch):
        input_ids = batch["input_ids"].to(trainer.device)
        labels = batch["labels"].to(trainer.device)
        outputs = model(input_ids)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, :shift_logits.size(1)].contiguous()
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    trainer.compute_loss = compute_loss

    # Register PiD callback
    trainer._pid = pid

    summary = trainer.train()
    print(f"Run {run_id} complete: {summary}")

    # Save final model
    model.save_pretrained(os.path.join(checkpoint_dir, "final"))
    config.save(os.path.join(checkpoint_dir, "final", "model_config.yaml"))
    print(f"Model saved to {checkpoint_dir}/final/")

    # For Run 4: also save quantized version
    if run_id == 4:
        print("Quantizing Run 4 to INT8...")
        model.quantize_all()
        model.save_pretrained(os.path.join(checkpoint_dir, "final-int8"))
        print(f"INT8 model saved to {checkpoint_dir}/final-int8/")

    return summary


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ablation Study: 4 × 150M models, 32k vocab, 2B tokens"
    )
    parser.add_argument("--run", type=str, default="all",
                        help="Run ID: 1, 2, 3, 4, or 'all'")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer",
                        help="Path to HF tokenizer directory")
    parser.add_argument("--target-tokens", type=int, default=2_000_000_000,
                        help="Target token count (default: 2B)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = batch × accum)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--save-steps", type=int, default=10000)
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/ablation-150m")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # Figure out which run to resume (if any)
    args.resume_run = None
    if args.resume:
        for rid in [1, 2, 3, 4]:
            if RUN_CONFIGS[rid][0] in args.resume:
                args.resume_run = rid
                break

    # Print study overview
    tokens_per_step = args.batch_size * args.gradient_accumulation * 2048
    total_steps = math.ceil(args.target_tokens / tokens_per_step)
    print(f"Ablation Study: 150M × 4 variants")
    print(f"  Tokens/step: {tokens_per_step:,} "
          f"(batch={args.batch_size} × accum={args.gradient_accumulation} × seq=2048)")
    print(f"  Total steps: {total_steps:,} ({args.target_tokens/1e9:.1f}B tokens)")
    print(f"  LR: {args.lr}, warmup: {args.warmup_steps}")
    print()

    # Run
    if args.run == "all":
        results = {}
        for run_id in [1, 2, 3, 4]:
            results[run_id] = train_run(run_id, args)
        print("\n" + "="*70)
        print("All runs complete!")
        for rid, summary in results.items():
            print(f"  Run {rid} ({RUN_CONFIGS[rid][1]}): {summary}")
    else:
        run_id = int(args.run)
        if run_id not in RUN_CONFIGS:
            raise ValueError(f"Invalid run ID: {run_id}. Choose 1-4 or 'all'.")
        train_run(run_id, args)


if __name__ == "__main__":
    main()
