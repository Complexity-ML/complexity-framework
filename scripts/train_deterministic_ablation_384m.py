"""
384M dense ablation, Chinchilla-scale — three init / parametrisation arms.

Direct twin of the existing ``abl-dense-adamw`` run
(checkpoints/abl-dense-adamw/, 3815 steps, 8B tokens on 8 GPUs). Same
architecture, tokenizer, data, batch size, schedule, grad clipping —
only the init / parametrisation differs:

  --init-type default                 → framework default
                                        (random normal_(0.02) + GPT-style
                                        residual scaling 1/√(2N)).
                                        Reproduces ``abl-dense-adamw``
                                        when seeded identically.

  --init-type deterministic           → ``mlp_type=dense_deterministic`` —
                                        locally-seeded RNG-free Gaussian
                                        on every Linear + embedding (see
                                        complexity/core/mlp/
                                        deterministic_init.py).

  --init-type deterministic --use-mup → above + the full μP triplet:
                                        init scaled by 1/√width_mult on
                                        hidden→hidden, attention logits
                                        / d_head (vs / √d_head), lm_head
                                        output divided by width_mult.
                                        Optimiser switches to
                                        ``adamw_mup`` so the LR side of
                                        μP matches the init side.

Hardware: built for 8×B200 (or 8×H100 / 8×A100) FSDP. Runs through the
framework's ``Trainer`` API, so checkpoints, resume, gradient
accumulation, and bf16 mixed-precision behave identically to other
ablations in this codebase.

Usage:
    # Reproduce the existing AdamW dense baseline
    torchrun --nproc_per_node 8 scripts/train_deterministic_ablation_384m.py \\
        --init-type default \\
        --checkpoint-dir ./checkpoints/abl-dense-default

    # Deterministic init, same optimiser
    torchrun --nproc_per_node 8 scripts/train_deterministic_ablation_384m.py \\
        --init-type deterministic \\
        --checkpoint-dir ./checkpoints/abl-dense-deterministic

    # Deterministic + μP (matches abl-dense-adamw at 1× width; transfers
    # to wider widths without re-tuning)
    torchrun --nproc_per_node 8 scripts/train_deterministic_ablation_384m.py \\
        --init-type deterministic --use-mup \\
        --checkpoint-dir ./checkpoints/abl-dense-mup

Compare metrics.csv files head-to-head against
``./checkpoints/abl-dense-adamw/abl-dense-adamw.csv`` — at convergence,
|Δloss| < 0.01 between ``default`` and ``deterministic`` validates that
the deterministic re-init does not regress, and the μP arm is expected
to track within batch noise at this width while remaining transferable.

Complexity-ML — 2026
"""

from __future__ import annotations

# Configure CUDA backends (TF32, cuDNN benchmark, cuDNN SDPA disable) at
# import time so it runs before any module touches torch.cuda.
from complexity.gpu import setup_gpu
setup_gpu()

import argparse
import logging
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerFast

from complexity.config import ModelConfig
from complexity.models import ComplexityModel
from complexity.training import Trainer, TrainingConfig
from complexity.parallel import (
    init_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    cleanup,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("ablation_384m")

# Silence HF / httpx INFO floods that otherwise drown training logs.
for _lib in ("httpx", "httpcore", "huggingface_hub", "datasets",
             "transformers", "urllib3"):
    logging.getLogger(_lib).setLevel(logging.WARNING)


# --------------------------------------------------------------------------
# Model config — 384M dense SwiGLU, matches abl-dense-adamw exactly
# --------------------------------------------------------------------------

def make_config(
    *,
    init_type: str,
    use_mup: bool,
    mup_base_width: int,
) -> ModelConfig:
    """~384M dense SwiGLU baseline — same shape as ``abl-dense-adamw``.

    Identical backbone to the 383M MoE ablation set in this repository
    (hidden=1024, 20 layers, GQA 16/4, RMSNorm + QK-norm). Intermediate
    size 3200 matches the sum of routed + shared expert widths in the
    MoE recipe so the parameter budget is comparable across init schemes.

    The ``mlp_type`` switches between the framework-default SwiGLU
    (random init, consumes the global PRNG) and the deterministic
    variant (``dense_deterministic`` — RNG-free Gaussian per matrix).
    The μP knobs are wired here too; they are no-ops at base width.
    """
    if init_type == "deterministic":
        mlp_type = "dense_deterministic"
    elif init_type == "default":
        mlp_type = "swiglu"
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    # Matches scripts/train_400m_dense.py (the script that produced
    # checkpoints/abl-dense-adamw): hidden=1024, layers=20, GQA 16/4,
    # intermediate_size=4358, max_pos=4096, vocab=32000.
    return ModelConfig(
        hidden_size=1024,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=4358,
        vocab_size=32000,
        max_position_embeddings=4096,
        attention_type="gqa",
        mlp_type=mlp_type,
        num_experts=1,
        shared_expert=False,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=False,
        # μP triplet — no-ops when use_mup=False or hidden_size == mup_base_width.
        use_mup_init=use_mup,
        use_mup_attn_scale=use_mup,
        use_mup_output_mult=use_mup,
        mup_base_width=mup_base_width,
    )


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class FineWebStreamingDataset(IterableDataset):
    """FineWeb-Edu streaming, rank-sharded for multi-GPU."""

    def __init__(self, tokenizer, max_length: int = 2048,
                 rank: int = 0, world_size: int = 1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        # Rank sharding so each worker sees a disjoint slice.
        for idx, sample in enumerate(ds):
            if (idx % self.world_size) != self.rank:
                continue
            text = sample.get("text", "")
            if not text:
                continue
            tok = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            yield {"input_ids": tok["input_ids"][0]}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def compute_steps_for_tokens(
    *,
    target_tokens: int,
    batch_size: int,
    grad_accum: int,
    seq_len: int,
    world_size: int,
) -> int:
    tokens_per_step = batch_size * grad_accum * seq_len * world_size
    return max(1, math.ceil(target_tokens / tokens_per_step))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="384M dense ablation: default ↔ deterministic ↔ μP"
    )
    parser.add_argument(
        "--init-type",
        type=str,
        default="deterministic",
        choices=["default", "deterministic"],
        help="default = framework SwiGLU + random init; "
             "deterministic = dense_deterministic (RNG-free Gaussian).",
    )
    parser.add_argument(
        "--use-mup",
        action="store_true",
        help="Enable the μP triplet (init / attn-scale / output mult). "
             "Switches the optimiser to adamw_mup so the LR side "
             "matches the init side.",
    )
    parser.add_argument("--mup-base-width", type=int, default=256,
                        help="Reference width for μP scaling (default: 256).")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--target-tokens", type=int, default=8_000_000_000,
                        help="Token budget — default 8B matches abl-dense-adamw.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Per-GPU batch size.")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="0 = auto 5%% of total steps.")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["cosine", "wsd", "constant"])
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Defaults to ./checkpoints/abl-dense-<init>"
                             "[+'-mup' when --use-mup].")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # --- Distributed setup ---------------------------------------------------
    init_distributed()
    rank = get_rank()
    world_size = get_world_size()

    if args.checkpoint_dir is None:
        suffix = "-mup" if args.use_mup else ""
        args.checkpoint_dir = f"./checkpoints/abl-dense-{args.init_type}{suffix}"
    if is_main_process():
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    # --- Tokenizer + dataset -------------------------------------------------
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    dataset = FineWebStreamingDataset(
        tokenizer=tokenizer,
        max_length=args.seq_len,
        rank=rank,
        world_size=world_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Model ---------------------------------------------------------------
    config = make_config(
        init_type=args.init_type,
        use_mup=args.use_mup,
        mup_base_width=args.mup_base_width,
    )
    config.vocab_size = len(tokenizer)
    model = ComplexityModel(config)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Model: 384M dense SwiGLU — {n_params/1e6:.1f}M params, "
            f"mlp_type={config.mlp_type}"
        )
        logger.info(
            f"Config: hidden={config.hidden_size}, "
            f"layers={config.num_hidden_layers}, "
            f"heads={config.num_attention_heads}/{config.num_key_value_heads}, "
            f"inter={config.intermediate_size}"
        )
        if args.init_type == "deterministic":
            logger.info(
                "Init: dense_deterministic — locally-seeded Gaussian on "
                "every Linear + embedding (RNG-free). Residual scaling "
                "1/√(2N) preserved."
            )
        else:
            logger.info(
                f"Init: framework default (normal_(0.02) + residual "
                f"scaling). Seeded with torch.manual_seed({args.seed})."
            )
        if args.use_mup:
            wm = config.hidden_size / args.mup_base_width
            logger.info(
                f"μP: enabled (base_width={args.mup_base_width}, "
                f"width_mult={wm:.2f}). Optimiser → adamw_mup."
            )
        else:
            logger.info("μP: disabled (standard parametrisation).")

    # --- Steps budget to match abl-dense-adamw -------------------------------
    max_steps = compute_steps_for_tokens(
        target_tokens=args.target_tokens,
        batch_size=args.batch_size,
        grad_accum=args.gradient_accumulation,
        seq_len=args.seq_len,
        world_size=world_size,
    )
    warmup = (args.warmup_steps if args.warmup_steps > 0
              else max(1, int(max_steps * 0.05)))

    if is_main_process():
        tokens_per_step = (
            args.batch_size * args.gradient_accumulation
            * args.seq_len * world_size
        )
        logger.info(
            f"Budget: target {args.target_tokens/1e9:.1f}B tokens, "
            f"{tokens_per_step/1e6:.2f}M tokens/step → {max_steps} steps"
        )
        logger.info(f"Warmup: {warmup} steps ({warmup / max_steps * 100:.1f}%)")

    # --- Trainer (FSDP, bf16, cosine, grad-clip, checkpointing) --------------
    train_config = TrainingConfig(
        max_steps=max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        optimizer_type="adamw_mup" if args.use_mup else "adamw",
        learning_rate=args.lr,
        warmup_steps=warmup,
        lr_scheduler=args.lr_scheduler,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        precision="bf16",
        save_steps=args.save_steps,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        mup_base_width=args.mup_base_width,
        log_steps=1,  # one line per step — live feedback
    )

    # ComplexityModel returns a dict in training mode (logits=None,
    # last_hidden_state present). The Trainer's default loss expects a
    # bare tensor — provide a compute_loss that handles the dict shape
    # AND masks padding so the loss isn't artificially low.
    pad_id = (tokenizer.pad_token_id
              if tokenizer.pad_token_id is not None
              else tokenizer.eos_token_id)

    def compute_loss(model, batch):
        input_ids = batch["input_ids"].to(trainer.device)
        out = model(input_ids)
        h = out["last_hidden_state"] if isinstance(out, dict) else out
        if model.lm_head is not None:
            logits = model.lm_head(h)
        else:
            logits = torch.nn.functional.linear(h, model.embed_tokens.weight)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        # Mask pad tokens so they don't contribute to the loss.
        if pad_id is not None:
            shift_labels = shift_labels.masked_fill(
                shift_labels == pad_id, -100
            )
        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100,
        )

    trainer = Trainer(
        model=model, config=train_config, train_dataloader=dataloader,
        compute_loss=compute_loss,
    )
    # tqdm progress bar (collective reductions — ALL ranks must register).
    from complexity.training.callbacks import TqdmCallback
    tqdm_cb = TqdmCallback(total_steps=max_steps, desc=Path(args.checkpoint_dir).name)
    trainer.callbacks.append(tqdm_cb)

    # CSV logger — writes one row per step to <ckpt>/metrics.csv so the
    # run is comparable to the existing abl-dense-adamw.csv reference.
    if is_main_process():
        import csv as _csv, math as _math, time as _time
        _csv_path = Path(args.checkpoint_dir) / "metrics.csv"
        _csv_path.parent.mkdir(parents=True, exist_ok=True)
        _csv_f = _csv_path.open("w", newline="")
        _csv_w = _csv.writer(_csv_f)
        _csv_w.writerow(["step", "loss", "ppl", "lr",
                         "tokens_seen", "elapsed_s"])
        _csv_f.flush()
        _t0 = _time.perf_counter()
        _tokens_per_step = (args.batch_size * args.gradient_accumulation
                            * args.seq_len * world_size)
        def _csv_cb(_trainer, step, loss):
            lr = _trainer.scheduler.get_last_lr()[0] if _trainer.scheduler else args.lr
            ppl = _math.exp(min(loss, 20))
            _csv_w.writerow([step, f"{loss:.6f}", f"{ppl:.2f}",
                             f"{lr:.6e}", step * _tokens_per_step,
                             f"{_time.perf_counter() - _t0:.1f}"])
            _csv_f.flush()
        trainer.callbacks.append(_csv_cb)
    try:
        summary = trainer.train()
    finally:
        tqdm_cb.close() if hasattr(tqdm_cb, "close") else None

    if is_main_process():
        logger.info(f"Training complete: {summary}")
        logger.info(
            f"Compare: diff {args.checkpoint_dir}/metrics.csv vs "
            f"./checkpoints/abl-dense-adamw/abl-dense-adamw.csv"
        )

    cleanup()


if __name__ == "__main__":
    main()
