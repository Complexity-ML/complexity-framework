"""
Local residual Token-Routed pretraining runner for o200k tokenizer profiles.

Runs on CUDA, MPS, or CPU with the same CLI/log schema as the 300M scaling
script. This profile is sized around 100M parameters after the large o200k
embedding table is included.

Examples:
    python3 scripts/train_100m_o200k_tr_local.py --steps 100 --dataset random
    python3 scripts/train_100m_o200k_tr_local.py --profile 50m --steps 100
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import string
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from complexity.config import ModelConfig
from complexity.core.losses import causal_lm_loss_from_hidden
from complexity.models import ComplexityModel
from complexity.tokenizer import Tokenizer
from complexity.training.moe_telemetry import global_expert_shares, global_tr_diagnostics
from complexity.training.run_config import (
    args_to_run_config,
    format_run_summary,
    parse_args_with_yaml_config,
    write_or_validate_run_config,
)
from complexity.utils import autocast, autocast_dtype, empty_cache, setup_mps, synchronize
from complexity.utils.device import backend_metadata, configure_torch_acceleration
from complexity.utils.local_checkpoint import load_local_checkpoint, resolve_checkpoint_path, save_local_checkpoint


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logging.getLogger("complexity.core.mlp.token_routed").setLevel(logging.WARNING)
for noisy_logger in ("httpx", "httpcore", "huggingface_hub", "datasets"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


PROFILES = {
    "50m": {
        "hidden_size": 224,
        "num_hidden_layers": 8,
        "num_attention_heads": 7,
        "num_key_value_heads": 1,
        "intermediate_size": 128,
        "shared_intermediate_size": 1024,
        "run_name": "50m-o200k-tr-local",
        "save_dir": "checkpoints/50m-o200k-tr-local",
        "description": "50M o200k TR",
    },
    "100m": {
        "hidden_size": 384,
        "num_hidden_layers": 10,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "shared_intermediate_size": 1536,
        "run_name": "100m-o200k-tr-local",
        "save_dir": "checkpoints/100m-o200k-tr-local",
        "description": "100M o200k TR",
    },
    "300m": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 3,
        "intermediate_size": 256,
        "shared_intermediate_size": 3072,
        "run_name": "300m-o200k-tr-local",
        "save_dir": "checkpoints/300m-o200k-tr-local",
        "description": "300M o200k TR",
    },
    "8b": {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 3072,
        "shared_intermediate_size": 12288,
        "run_name": "8b-o200k-tr-local",
        "save_dir": "checkpoints/8b-o200k-tr-local",
        "description": "8B o200k TR",
    },
}


def make_config(args) -> ModelConfig:
    return ModelConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        vocab_size=args.vocab_size,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="token_routed",
        num_experts=4,
        shared_expert=True,
        shared_intermediate_size=args.shared_intermediate_size,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=args.use_mu_guidance,
        use_shared_routed_gates=args.learn_shared_routed_gates,
        shared_gate_init=args.shared_gate_init,
        routed_gate_init=args.routed_gate_init,
        top_k=args.top_k,
        top_k_primary_weight=args.top_k_primary_weight,
        use_custom_kernels=getattr(args, "use_custom_kernels", "auto"),
        static_expert_capacity=bool(getattr(args, "static_expert_capacity", False)),
        routing_strategy=getattr(args, "routing_strategy", "zipf"),
        clamp_mu_contextual=args.mu_clamp,
        use_mu_norm=args.mu_norm,
        mu_alpha_init=args.mu_alpha_init,
        mu_init_value=args.mu_init_value,
        mu_context_min=args.mu_context_min,
        mu_context_max=args.mu_context_max,
    )


class RandomTokenDataset(IterableDataset):
    def __init__(self, vocab_size: int, seq_len: int, seed: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed)
        while True:
            ids = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=gen)
            yield {"input_ids": ids[:-1], "labels": ids[1:]}


class LocalTextDataset(IterableDataset):
    def __init__(self, tokens: list[int], seq_len: int, seed: int):
        if len(tokens) < seq_len + 2:
            raise ValueError(f"Need at least {seq_len + 2} tokens, got {len(tokens)}")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed)
        high = self.tokens.numel() - self.seq_len - 1
        while True:
            start = torch.randint(0, high + 1, (1,), generator=gen).item()
            chunk = self.tokens[start : start + self.seq_len + 1]
            yield {"input_ids": chunk[:-1], "labels": chunk[1:]}


class FineWebDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len: int, rank: int, world_size: int):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

    def __iter__(self):
        buffer: list[int] = []
        for idx, example in enumerate(self.dataset):
            if idx % self.world_size != self.rank:
                continue
            text = example.get("text", "")
            if not text:
                continue
            buffer.extend(self.tokenizer.encode(text, add_special_tokens=False))
            if self.tokenizer.eos_token_id is not None:
                buffer.append(self.tokenizer.eos_token_id)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }


def load_text_tokens(path: str, tokenizer_path: str) -> list[int]:
    tokenizer = Tokenizer.load(tokenizer_path)
    text = Path(path).read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    logger.info(f"Text dataset: {path} ({len(tokens):,} tokens)")
    return tokens


def infer_vocab_size(args) -> int:
    if args.vocab_size is not None:
        return args.vocab_size
    if args.dataset == "random":
        return 32000
    return Tokenizer.load(args.tokenizer).vocab_size


def text_token_frequencies(path: str, tokenizer_path: str, vocab_size: int) -> torch.Tensor:
    tokens = load_text_tokens(path, tokenizer_path)
    ids = torch.tensor(tokens, dtype=torch.long)
    ids = ids[(ids >= 0) & (ids < vocab_size)]
    freqs = torch.zeros(vocab_size, dtype=torch.float32)
    if ids.numel() > 0:
        freqs.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
    logger.info(
        f"Zipf routing frequencies: {int(freqs.sum().item()):,} tokens, "
        f"{int((freqs > 0).sum().item()):,} vocab entries"
    )
    return freqs


def tokenizer_token_classes(tokenizer_path: str, vocab_size: int) -> torch.Tensor:
    """Classify each token into coarse lexical buckets for static routing."""

    tokenizer = Tokenizer.load(tokenizer_path)
    classes = torch.zeros(vocab_size, dtype=torch.long)
    encoding = getattr(getattr(tokenizer, "_tokenizer", None), "encoding", None)
    for token_id in range(vocab_size):
        text = _decode_token_for_class(tokenizer, encoding, token_id)
        classes[token_id] = _classify_token_text(text)
    counts = torch.bincount(classes, minlength=8)
    logger.info(
        "Token classes: "
        + ", ".join(f"{idx}={int(count)}" for idx, count in enumerate(counts.tolist()) if count)
    )
    return classes


def _decode_token_for_class(tokenizer: Tokenizer, encoding, token_id: int) -> str:
    try:
        if encoding is not None and hasattr(encoding, "decode_single_token_bytes"):
            return encoding.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return ""


def _classify_token_text(text: str) -> int:
    if not text:
        return 0
    if text.isspace():
        return 1
    stripped = text.strip()
    if not stripped:
        return 1
    if stripped.isdigit():
        return 2
    if stripped.isalpha() and stripped.isascii():
        return 3
    if stripped.isalnum() and stripped.isascii():
        return 4
    if any(ord(ch) > 127 for ch in stripped):
        return 6
    if all(ch in string.punctuation for ch in stripped):
        return 5
    return 7


def split_tokens(tokens: list[int], eval_ratio: float) -> tuple[list[int], list[int]]:
    n_eval = max(2048, int(len(tokens) * eval_ratio))
    n_eval = min(n_eval, max(1, len(tokens) // 5))
    return tokens[:-n_eval], tokens[-n_eval:]


@torch.no_grad()
def evaluate(model, raw_model, loader, device, amp_dtype, eval_batches, label_smoothing, z_loss, loss_chunk_tokens, distributed):
    was_training = model.training
    model.eval()
    losses = []
    for idx, batch in enumerate(loader):
        if idx >= eval_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(input_ids, return_logits=False)
            _, metrics = causal_lm_loss_from_hidden(
                outputs["last_hidden_state"],
                raw_model.embed_tokens.weight,
                labels,
                label_smoothing=label_smoothing,
                z_loss_coef=z_loss,
                chunk_tokens=loss_chunk_tokens,
                checkpoint_chunks=False,
            )
        losses.append(metrics.ce)
    if was_training:
        model.train()
    eval_loss = sum(losses) / max(1, len(losses))
    if distributed:
        loss_tensor = torch.tensor(eval_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        eval_loss = loss_tensor.item()
    return eval_loss


def build_loaders(args, config, rank: int, world_size: int):
    if args.dataset == "fineweb":
        tokenizer = Tokenizer.load(args.tokenizer)
        if rank == 0:
            logger.info("Dataset: FineWeb-Edu sample-10BT streaming")
        train_ds = FineWebDataset(tokenizer, args.seq_len, rank, world_size)
        eval_ds = FineWebDataset(tokenizer, args.seq_len, rank, world_size) if args.eval_steps > 0 else None
    elif args.dataset == "text":
        if not args.text_file:
            raise ValueError("--text-file is required when --dataset text")
        tokens = load_text_tokens(args.text_file, args.tokenizer)
        train_tokens, eval_tokens = split_tokens(tokens, args.eval_ratio)
        train_ds = LocalTextDataset(train_tokens, args.seq_len, args.seed + rank)
        eval_ds = LocalTextDataset(eval_tokens, args.seq_len, args.seed + 10_000 + rank)
    else:
        train_ds = RandomTokenDataset(config.vocab_size, args.seq_len, args.seed + rank)
        eval_ds = RandomTokenDataset(config.vocab_size, args.seq_len, args.seed + 10_000 + rank)

    loader_kwargs = {"batch_size": args.batch_size, "pin_memory": False}
    if args.num_workers > 0:
        loader_kwargs.update(num_workers=args.num_workers, persistent_workers=True)
    eval_loader = DataLoader(eval_ds, **loader_kwargs) if eval_ds is not None else None
    return DataLoader(train_ds, **loader_kwargs), eval_loader


def init_distributed(seed: int):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires a CUDA-compatible GPU backend (NVIDIA CUDA or AMD ROCm).")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        torch.manual_seed(seed + rank)
        return torch.device("cuda", local_rank), distributed, rank, local_rank, world_size

    device = setup_mps(unlimited_watermark=True, cpu_fallback=True, seed=seed)
    return device, distributed, rank, local_rank, world_size


def reduce_average(value: float, device: torch.device, distributed: bool) -> float:
    if not distributed:
        return value
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor.item()


def batch_expert_counts(raw_model, input_ids: torch.Tensor, num_experts: int, distributed: bool) -> torch.Tensor:
    """Return per-expert token counts for the current batch."""
    for module in raw_model.modules():
        if hasattr(module, "token_to_expert"):
            token_to_expert = module.token_to_expert
            token_ids = input_ids.clamp(0, token_to_expert.numel() - 1)
            expert_ids = token_to_expert[token_ids].reshape(-1)
            counts = torch.bincount(expert_ids, minlength=num_experts).to(
                device=input_ids.device,
                dtype=torch.float32,
            )
            if distributed:
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            return counts
    counts = torch.ones(num_experts, device=input_ids.device, dtype=torch.float32)
    if distributed:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return counts


def build_optimizer(args, raw_model):
    """Build AdamW or MuonTR for the o200k TR runner."""
    if args.optimizer == "adamw":
        decay, no_decay = [], []
        for name, p in raw_model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim < 2 or "bias" in name or "norm" in name else decay).append(p)
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr,
            betas=(0.9, 0.95),
        )
        return optimizer, {"adamw_params": sum(p.numel() for p in decay + no_decay)}

    if args.optimizer == "muon_tr":
        from complexity.training.muon_tr import MuonTRWithAdamW, split_params_for_muon_tr

        muon_groups, adam_groups = split_params_for_muon_tr(
            raw_model,
            num_experts=4,
            muon_scope=args.muon_scope,
        )
        optimizer = MuonTRWithAdamW(
            muon_params=muon_groups,
            adam_params=adam_groups,
            lr=args.muon_lr,
            adam_lr=args.lr,
            weight_decay=args.weight_decay,
            expert_lr_scale=args.expert_lr_scale,
            shared_lr_scale=args.shared_lr_scale,
            expert_weight_decay=args.expert_weight_decay,
            shared_weight_decay=args.shared_weight_decay,
            ns_steps=args.muon_ns_steps,
            adaptive_ns=args.muon_adaptive_ns,
            max_lr_ratio=args.muon_max_lr_ratio,
            lr_warmup_steps=args.muon_lr_warmup_steps,
            lr_decay_start_step=getattr(args, "muon_lr_decay_start_step", 0),
            lr_decay_end_step=getattr(args, "muon_lr_decay_end_step", 0),
            lr_decay_min_mult=getattr(args, "muon_lr_decay_min_mult", 1.0),
            skip_ns_warmup_steps=args.muon_skip_ns_warmup_steps,
            nesterov=not getattr(args, "muon_no_nesterov", False),
            orthogonal_blend=getattr(args, "muon_orthogonal_blend", 0.5),
            orthogonal_blend_start=getattr(args, "muon_orthogonal_blend_start", None),
            orthogonal_blend_decay_steps=getattr(args, "muon_orthogonal_blend_decay_steps", 0),
            max_param_rms_ratio=getattr(args, "muon_max_param_rms_ratio", None),
            token_count_scaling=args.muon_token_count_scaling,
            max_update_rms=args.muon_max_update_rms,
            num_experts=4,
        )
        return optimizer, {
            "muon_expert_params": sum(
                p.numel() for group in muon_groups for p in group["params"]
                if group.get("param_type") == "expert"
            ),
            "muon_shared_params": sum(
                p.numel() for group in muon_groups for p in group["params"]
                if group.get("param_type") == "shared"
            ),
            "muon_dense_params": sum(
                p.numel() for group in muon_groups for p in group["params"]
                if group.get("param_type") == "dense"
            ),
            "adamw_params": sum(p.numel() for group in adam_groups for p in group["params"]),
        }

    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def save_checkpoint(args, raw_model, optimizer, scheduler, config, step: int, is_main: bool, distributed: bool):
    if distributed:
        dist.barrier()
    if not is_main or args.save_steps <= 0:
        if distributed:
            dist.barrier()
        return

    ckpt_dir = save_local_checkpoint(
        args.save_dir,
        step=step,
        total_limit=args.save_total_limit,
        state={
            "step": step,
            "model": {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.to_dict(),
            "args": vars(args),
            "backend": backend_metadata(kernel_policy=getattr(args, "use_custom_kernels", "auto")),
        },
    )
    logger.info(f"Checkpoint saved: {ckpt_dir}")
    if distributed:
        dist.barrier()


def load_checkpoint(path: str, raw_model, optimizer, scheduler, device, is_main: bool) -> int:
    ckpt_dir, state = load_local_checkpoint(path, map_location=device)
    raw_model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    step = int(state["step"])
    if is_main:
        logger.info(f"Resumed from {ckpt_dir} at step {step}")
    return step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local o200k residual Token-Routed pretraining runner")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="100m")
    parser.add_argument("--dataset", choices=["random", "text", "fineweb"], default="random")
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="./tokenizer-o200k")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--optimizer", choices=["adamw", "muon_tr"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-lr", type=float, default=0.003)
    parser.add_argument("--muon-scope", choices=["expert", "expert_shared", "all"], default="expert")
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adaptive-ns", action="store_true")
    parser.add_argument("--muon-lr-warmup-steps", type=int, default=50)
    parser.add_argument("--muon-lr-decay-start-step", type=int, default=0)
    parser.add_argument("--muon-lr-decay-end-step", type=int, default=0)
    parser.add_argument("--muon-lr-decay-min-mult", type=float, default=1.0)
    parser.add_argument("--muon-skip-ns-warmup-steps", type=int, default=0)
    parser.add_argument("--muon-no-nesterov", action="store_true")
    parser.add_argument("--muon-orthogonal-blend", type=float, default=0.5)
    parser.add_argument("--muon-orthogonal-blend-start", type=float, default=None)
    parser.add_argument("--muon-orthogonal-blend-decay-steps", type=int, default=0)
    parser.add_argument("--muon-max-param-rms-ratio", type=float, default=None)
    parser.add_argument("--muon-token-count-scaling", action="store_true")
    parser.add_argument("--muon-max-lr-ratio", type=float, default=2.0)
    parser.add_argument("--muon-max-update-rms", type=float, default=1.0)
    parser.add_argument("--expert-lr-scale", type=float, default=1.0)
    parser.add_argument("--expert-weight-decay", type=float, default=0.005)
    parser.add_argument("--shared-lr-scale", type=float, default=1.0)
    parser.add_argument("--shared-weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-hidden-layers", type=int, default=None)
    parser.add_argument("--num-attention-heads", type=int, default=None)
    parser.add_argument("--num-key-value-heads", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--shared-intermediate-size", type=int, default=None)
    parser.add_argument("--shared-gate-init", type=float, default=1.0)
    parser.add_argument("--routed-gate-init", type=float, default=0.1)
    parser.add_argument("--learn-shared-routed-gates", dest="learn_shared_routed_gates", action="store_true", default=True)
    parser.add_argument("--no-learn-shared-routed-gates", dest="learn_shared_routed_gates", action="store_false")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--top-k-primary-weight", type=float, default=0.5)
    parser.add_argument(
        "--use-custom-kernels",
        choices=["auto", "true", "false"],
        default="auto",
        help="Custom Triton/CUDA kernels. auto enables NVIDIA CUDA, disables ROCm by default.",
    )
    parser.add_argument(
        "--static-expert-capacity",
        action="store_true",
        help="Use export-friendly TR dispatch for torch.distributed.pipelining.",
    )
    parser.add_argument("--routing-strategy", choices=["zipf", "zipf_token_class"], default="zipf")
    parser.add_argument("--use-mu-guidance", action="store_true")
    parser.add_argument("--mu-clamp", action="store_true")
    parser.add_argument("--mu-norm", action="store_true")
    parser.add_argument("--mu-alpha-init", type=float, default=1.0)
    parser.add_argument("--mu-init-value", type=float, default=0.0)
    parser.add_argument("--mu-context-min", type=float, default=-2.0)
    parser.add_argument("--mu-context-max", type=float, default=2.0)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--z-loss", type=float, default=0.0)
    parser.add_argument("--loss-chunk-tokens", type=int, default=1024)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--force-resume", action="store_true")
    parser.add_argument(
        "--no-zipf-from-text",
        action="store_true",
        help="Disable token-frequency balanced routing when --dataset text.",
    )
    return parser


def main():
    parser = build_parser()
    args = parse_args_with_yaml_config(parser)
    profile = PROFILES[args.profile]
    for key in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "shared_intermediate_size",
        "run_name",
        "save_dir",
    ):
        if getattr(args, key) is None:
            setattr(args, key, profile[key])

    device, distributed, rank, local_rank, world_size = init_distributed(args.seed)
    is_main = rank == 0
    kernel_policy = (
        True if args.use_custom_kernels == "true"
        else False if args.use_custom_kernels == "false"
        else "auto"
    )
    args.use_custom_kernels = kernel_policy
    configure_torch_acceleration(kernel_policy=kernel_policy, log=is_main)
    args.vocab_size = infer_vocab_size(args)
    config = make_config(args)
    if args.dataset == "text" and not args.no_zipf_from_text:
        config.token_frequencies = text_token_frequencies(
            args.text_file,
            args.tokenizer,
            config.vocab_size,
        )
    if args.routing_strategy == "zipf_token_class":
        config.token_classes = tokenizer_token_classes(args.tokenizer, config.vocab_size)
    raw_model = ComplexityModel(config).to(device)
    if args.grad_ckpt:
        raw_model.gradient_checkpointing_enable()

    params = raw_model.num_parameters()
    run_dir = Path("runs") / args.run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        run_config = args_to_run_config(
            args,
            model_config=config.to_dict(),
            params=params,
            world_size=world_size,
            backend=backend_metadata(kernel_policy=kernel_policy),
        )
        write_or_validate_run_config(
            run_dir,
            run_config,
            resume=bool(args.resume),
            force_resume=args.force_resume,
        )
        logger.info(f"Model: {params / 1e6:.1f}M params")
        for line in format_run_summary(run_config):
            logger.info(line)
        logger.info(
            "Config: Token-Routed residual, "
            f"hidden={args.hidden_size}, layers={args.num_hidden_layers}, "
            f"GQA={args.num_attention_heads}/{args.num_key_value_heads}, "
            f"inter={args.intermediate_size}, shared_inter={args.shared_intermediate_size}, "
            f"experts=4, top_k={args.top_k}, primary_w={args.top_k_primary_weight}, "
            f"learn_gates={args.learn_shared_routed_gates}, "
            f"gates=({args.shared_gate_init},{args.routed_gate_init}), "
            f"use_mu={args.use_mu_guidance}, mu_clamp={args.mu_clamp}, mu_norm={args.mu_norm}, "
            f"mu_alpha={args.mu_alpha_init}, mu_init={args.mu_init_value}"
        )
        logger.info(
            "Optimizer: "
            f"{args.optimizer}, adam_lr={args.lr:.2e}, weight_decay={args.weight_decay}, "
            f"muon_lr={args.muon_lr:.2e}, muon_scope={args.muon_scope}, "
            f"expert_lr_scale={args.expert_lr_scale}"
        )
        if distributed:
            logger.info(f"DDP: world_size={world_size}, per_gpu_batch={args.batch_size}")

    model = raw_model
    if distributed:
        model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    amp_dtype = autocast_dtype(device) if args.bf16 else None
    train_loader, eval_loader = build_loaders(args, config, rank, world_size)

    optimizer, optimizer_stats = build_optimizer(args, raw_model)
    if is_main:
        if args.optimizer == "muon_tr":
            logger.info(
                "MuonTR params: "
                f"expert={optimizer_stats['muon_expert_params'] / 1e6:.1f}M, "
                f"shared={optimizer_stats['muon_shared_params'] / 1e6:.1f}M, "
                f"dense={optimizer_stats['muon_dense_params'] / 1e6:.1f}M, "
                f"adamw={optimizer_stats['adamw_params'] / 1e6:.1f}M"
            )
        else:
            logger.info(f"AdamW params: {optimizer_stats['adamw_params'] / 1e6:.1f}M")
    warmup = max(1, int(args.steps * 0.05))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_step = 0
    if distributed:
        dist.barrier()
    if args.resume:
        start_step = load_checkpoint(args.resume, raw_model, optimizer, scheduler, device, is_main)
    if distributed:
        dist.barrier()

    csv_path = run_dir / "metrics.csv"
    csv_file = None
    writer = None
    if is_main:
        csv_mode = "a" if args.resume and csv_path.exists() else "w"
        csv_file = csv_path.open(csv_mode, newline="")
        writer = csv.writer(csv_file)
        if csv_mode == "w":
            writer.writerow([
                "step", "train_loss", "train_ppl", "eval_loss", "eval_ppl", "lr", "tok_s",
                "expert_0_share", "expert_1_share", "expert_2_share", "expert_3_share",
                "expert_dead_count", "shared_gate", "routed_gate", "shared_rms", "routed_rms",
                "shared_grad_norm", "routed_grad_norm", "expert_0_grad_norm",
                "expert_1_grad_norm", "expert_2_grad_norm", "expert_3_grad_norm",
            ])
        csv_file.flush()

    model.train()
    pbar = (
        tqdm(total=args.steps, initial=start_step, desc=profile["description"], unit="step", dynamic_ncols=True)
        if is_main else None
    )
    t_log = time.perf_counter()
    tokens_since_log = 0
    last_step = start_step

    for step, batch in enumerate(train_loader, start=start_step + 1):
        if step > args.steps:
            break
        last_step = step
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(input_ids, return_logits=False)
            loss, metrics = causal_lm_loss_from_hidden(
                outputs["last_hidden_state"],
                raw_model.embed_tokens.weight,
                labels,
                label_smoothing=args.label_smoothing,
                z_loss_coef=args.z_loss,
                chunk_tokens=args.loss_chunk_tokens,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if hasattr(optimizer, "update_token_counts"):
            optimizer.update_token_counts(
                batch_expert_counts(raw_model, input_ids, config.num_experts, distributed)
            )
        optimizer.step()
        scheduler.step()

        tokens_since_log += args.batch_size * args.seq_len * world_size
        if pbar is not None:
            pbar.update(1)

        should_eval = args.eval_steps > 0 and step % args.eval_steps == 0
        should_log = step == 1 or step % args.log_steps == 0 or should_eval
        if should_log:
            synchronize(device)
            now = time.perf_counter()
            tok_s = tokens_since_log / max(1e-9, now - t_log)
            eval_loss = float("nan")
            if should_eval and eval_loader is not None:
                eval_loss = evaluate(
                    model, raw_model, eval_loader, device, amp_dtype, args.eval_batches,
                    args.label_smoothing, args.z_loss, args.loss_chunk_tokens, distributed,
                )
            train_loss = reduce_average(metrics.ce, device, distributed)
            train_ppl = math.exp(min(train_loss, 20))
            eval_ppl = math.exp(min(eval_loss, 20)) if math.isfinite(eval_loss) else float("nan")
            lr_now = scheduler.get_last_lr()[0]
            shares, dead = global_expert_shares(raw_model, config.num_experts)
            if not shares:
                shares = [float("nan")] * config.num_experts
            tr_diag = global_tr_diagnostics(raw_model, config.num_experts)
            if is_main:
                writer.writerow([
                    step, f"{train_loss:.6f}", f"{train_ppl:.2f}",
                    f"{eval_loss:.6f}", f"{eval_ppl:.2f}",
                    f"{lr_now:.6e}", f"{tok_s:.0f}",
                    *[f"{s:.4f}" for s in shares], dead,
                    f"{tr_diag.get('shared_gate', float('nan')):.6f}",
                    f"{tr_diag.get('routed_gate', float('nan')):.6f}",
                    f"{tr_diag.get('shared_rms', float('nan')):.6f}",
                    f"{tr_diag.get('routed_rms', float('nan')):.6f}",
                    f"{tr_diag.get('shared_grad_norm', float('nan')):.6f}",
                    f"{tr_diag.get('routed_grad_norm', float('nan')):.6f}",
                    *[
                        f"{tr_diag.get(f'expert_{idx}_grad_norm', float('nan')):.6f}"
                        for idx in range(config.num_experts)
                    ],
                ])
                csv_file.flush()
                pbar.set_postfix(loss=f"{train_loss:.4f}", eval=f"{eval_loss:.4f}", tok_s=f"{tok_s:.0f}")
            t_log = now
            tokens_since_log = 0

        if args.empty_cache_every > 0 and step % args.empty_cache_every == 0:
            empty_cache(device)

        if args.save_steps > 0 and step % args.save_steps == 0:
            save_checkpoint(args, raw_model, optimizer, scheduler, config, step, is_main, distributed)

    if args.save_steps > 0 and last_step > start_step and last_step % args.save_steps != 0:
        save_checkpoint(args, raw_model, optimizer, scheduler, config, last_step, is_main, distributed)

    if pbar is not None:
        pbar.close()
    if csv_file is not None:
        csv_file.close()
        logger.info(f"Metrics saved: {csv_path}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
