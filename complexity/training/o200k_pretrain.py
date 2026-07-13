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

import csv
import logging
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from complexity.core.losses import (
    causal_lm_loss_from_hidden,
    fused_linear_causal_lm_loss,
    has_liger_fused_linear_ce,
)
from complexity.models import ComplexityModel
from complexity.training.o200k import (
    PROFILES,
    apply_shared_routed_gates,
    apply_topk_primary_weight,
    batch_expert_counts,
    build_parser,
    build_loaders,
    build_optimizer,
    evaluate,
    expert_diversity_loss,
    init_distributed,
    load_checkpoint,
    make_config,
    reduce_average_tensor,
    runtime_controls,
    scheduled_value,
    save_checkpoint,
    scheduled_topk_primary_weight,
    text_token_frequencies,
    token_shard_frequencies,
)
from complexity.training.moe_telemetry import global_expert_shares, global_tr_diagnostics
from complexity.training.run_config import (
    args_to_run_config,
    format_run_summary,
    parse_args_with_yaml_config,
    write_or_validate_run_config,
)
from complexity.utils import autocast, autocast_dtype, empty_cache, synchronize
from complexity.utils.device import backend_metadata, configure_torch_acceleration
from complexity.utils.local_checkpoint import resolve_checkpoint_path
from complexity.tokenizer import Tokenizer


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logging.getLogger("complexity.core.mlp.token_routed").setLevel(logging.WARNING)
for noisy_logger in ("httpx", "httpcore", "huggingface_hub", "datasets"):
    library_logger = logging.getLogger(noisy_logger)
    library_logger.setLevel(logging.ERROR)
    library_logger.propagate = False


def infer_vocab_size(args) -> int:
    """Backward-compatible wrapper for tests/callers patching this module."""
    if args.vocab_size is not None:
        return args.vocab_size
    vocab_size = Tokenizer.load(args.tokenizer).vocab_size
    logger.info(f"Tokenizer vocab size: {vocab_size:,} ({args.tokenizer})")
    return vocab_size


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
    liger_loss_available = has_liger_fused_linear_ce()
    if args.loss_backend == "liger" and not liger_loss_available:
        raise RuntimeError(
            "Requested --loss-backend liger but liger-kernel is not importable. "
            "Install liger-kernel or use --loss-backend chunked."
        )
    args.loss_backend_active = (
        "liger" if args.loss_backend in {"auto", "liger"} and liger_loss_available else "chunked"
    )
    config = make_config(args)
    if args.dataset == "tokens":
        config.token_frequencies = token_shard_frequencies(args.tokens_path, config.vocab_size)
        if is_main:
            logger.info(
                f"Zipf routing frequencies: {int(config.token_frequencies.sum().item()):,} mmap tokens, "
                f"{int((config.token_frequencies > 0).sum().item()):,} vocab entries"
            )
    elif args.dataset == "text" and not args.no_zipf_from_text:
        config.token_frequencies = text_token_frequencies(
            args.text_file,
            args.tokenizer,
            config.vocab_size,
        )
    if args.routing_strategy == "lsh_hidden":
        config.lsh_routing = True
        config.lsh_bits = int(getattr(args, "lsh_bits", 0))
        config.lsh_from_layer = int(getattr(args, "lsh_from_layer", 0))
        config.lsh_threshold_mode = getattr(args, "lsh_threshold_mode", "zero")
    raw_model = ComplexityModel(config).to(device)
    active_controls = runtime_controls(raw_model)
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
        if "lexical_object_gate" in active_controls.capabilities:
            micro_gate = (
                f"{active_controls.micro_gate:.4f}"
                if active_controls.micro_gate is not None
                else "disabled"
            )
            logger.info(
                "Config: attention-free lexical residual, "
                f"mixer={args.attention_type}, hidden={args.hidden_size}, "
                f"layers={args.num_hidden_layers}, inter={args.intermediate_size}, "
                f"conv_kernel={args.causal_conv_kernel_size}, "
                f"dilation_cycle={args.causal_conv_dilation_cycle}, "
                f"object_rank={args.lexical_object_rank}, "
                f"tied_objects={args.tie_lexical_object_embeddings}, "
                f"micro_experts={args.micro_num_experts}x{args.micro_expert_width}, "
                f"object_gate={active_controls.object_gate:.4f}, "
                f"micro_gate={micro_gate}, "
                f"grad_ckpt={args.grad_ckpt}"
            )
        else:
            logger.info(
                "Config: Token-Routed residual, "
                f"hidden={args.hidden_size}, layers={args.num_hidden_layers}, "
                f"GQA={args.num_attention_heads}/{args.num_key_value_heads}, "
                f"inter={args.intermediate_size}, shared_inter={args.shared_intermediate_size}, "
                f"shared_chunk={args.shared_expert_chunk_tokens}, "
                f"grad_ckpt={args.grad_ckpt}, experts=4, top_k={args.top_k}, "
                f"primary_w={args.top_k_primary_weight}, "
                f"primary_w_final={args.top_k_primary_weight_final}, "
                f"lsh_threshold={getattr(args, 'lsh_threshold_mode', 'zero')}, "
                f"learn_gates={args.learn_shared_routed_gates}, "
                f"gates=({args.shared_gate_init}->{args.shared_gate_final},"
                f"{args.routed_gate_init}->{args.routed_gate_final}), "
                f"expert_diversity={args.expert_diversity_lambda} "
                f"target={args.expert_diversity_target}"
            )
        logger.info(
            "Loss: "
            f"backend={args.loss_backend_active}, chunk_tokens={args.loss_chunk_tokens}, "
            f"checkpoint_chunks={args.loss_checkpoint_chunks}, vocab=exact"
        )
        optimizer_summary = (
            f"{args.optimizer}, adam_lr={args.lr:.2e}, weight_decay={args.weight_decay}"
        )
        if args.optimizer == "muon_tr":
            optimizer_summary += (
                f", muon_lr={args.muon_lr:.2e}, muon_scope={args.muon_scope}, "
                f"expert_lr_scale={args.expert_lr_scale}"
            )
        logger.info(f"Optimizer: {optimizer_summary}")
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

    # Optional torch.compile wrap — applied AFTER DDP so the compiled graph
    # subsumes the gradient all-reduce. `raw_model` keeps the un-compiled
    # module for save_pretrained / state_dict / param iteration. `model` is
    # what the training loop calls.
    if getattr(args, "compile", False):
        if is_main:
            logger.info(
                f"[compile] wrapping model with torch.compile(mode={args.compile_mode!r}). "
                "First step will hit a 1-2 min graph-compile penalty; "
                "subsequent steps run the fused graph."
            )
        # dynamic=False lets Inductor specialise on the (batch, seq) shape for
        # max throughput on steady-state training. If shapes actually vary
        # (e.g. variable-length packing), pass --compile-mode default and we
        # could expose --compile-dynamic later.
        model = torch.compile(model, mode=args.compile_mode, dynamic=False)

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
            logger.info(
                f"AdamW params: {optimizer_stats['adamw_params'] / 1e6:.1f}M "
                f"impl={optimizer_stats.get('adamw_impl', 'default')}"
            )
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
                "expert_diversity_loss", "expert_diversity_lambda", "total_loss",
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
        should_eval = args.eval_steps > 0 and step % args.eval_steps == 0
        should_log = step == 1 or step % args.log_steps == 0 or should_eval
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        topk_primary_weight = None
        if "topk_primary_weight" in active_controls.capabilities:
            topk_primary_weight = scheduled_topk_primary_weight(
                step - 1,
                args.steps,
                args.top_k_primary_weight,
                args.top_k_primary_weight_final,
                args.top_k_primary_weight_schedule_ratio,
            )
            apply_topk_primary_weight(raw_model, topk_primary_weight)
        shared_gate = (
            scheduled_value(
                step - 1,
                args.steps,
                args.shared_gate_init,
                args.shared_gate_final,
                args.gate_schedule_ratio,
            )
            if "shared_routed_gates" in active_controls.capabilities
            and args.shared_gate_final is not None
            else None
        )
        routed_gate = (
            scheduled_value(
                step - 1,
                args.steps,
                args.routed_gate_init,
                args.routed_gate_final,
                args.gate_schedule_ratio,
            )
            if "shared_routed_gates" in active_controls.capabilities
            and args.routed_gate_final is not None
            else None
        )
        apply_shared_routed_gates(raw_model, shared_gate, routed_gate)
        optimizer.zero_grad(set_to_none=True)
        diversity_lambda = scheduled_value(
            step - 1,
            args.steps,
            0.0,
            args.expert_diversity_lambda,
            args.expert_diversity_schedule_ratio,
        )
        with autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(input_ids, return_logits=False)
            if args.loss_backend_active == "liger":
                loss, _ = fused_linear_causal_lm_loss(
                    outputs["last_hidden_state"],
                    raw_model.embed_tokens.weight,
                    labels,
                    label_smoothing=args.label_smoothing,
                    z_loss_coef=args.z_loss,
                    sync_metrics=False,
                )
            else:
                loss, _ = causal_lm_loss_from_hidden(
                    outputs["last_hidden_state"],
                    raw_model.embed_tokens.weight,
                    labels,
                    label_smoothing=args.label_smoothing,
                    z_loss_coef=args.z_loss,
                    chunk_tokens=args.loss_chunk_tokens,
                    checkpoint_chunks=args.loss_checkpoint_chunks,
                    sync_metrics=False,
                )
            diversity = expert_diversity_loss(raw_model, args.expert_diversity_target)
            total_loss = (
                loss
                if diversity is None or diversity_lambda <= 0.0
                else loss + diversity_lambda * diversity
            )
        total_loss.backward()
        if args.max_grad_norm and args.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if hasattr(optimizer, "update_token_counts"):
            optimizer.update_token_counts(
                batch_expert_counts(raw_model, input_ids, config.num_experts, distributed)
            )
        optimizer.step()
        scheduler.step()

        tokens_since_log += args.batch_size * args.seq_len * world_size
        if pbar is not None:
            pbar.update(1)

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
            train_loss = reduce_average_tensor(loss, distributed)
            train_total_loss = reduce_average_tensor(total_loss, distributed)
            train_diversity_loss = (
                reduce_average_tensor(diversity, distributed)
                if diversity is not None
                else float("nan")
            )
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
                    f"{train_diversity_loss:.8f}",
                    f"{diversity_lambda:.8e}",
                    f"{train_total_loss:.6f}",
                ])
                csv_file.flush()
                postfix = {
                    "loss": f"{train_loss:.4f}",
                    "eval": f"{eval_loss:.4f}",
                    "tok_s": f"{tok_s:.0f}",
                }
                current_controls = runtime_controls(raw_model)
                postfix.update(
                    {
                        name: f"{value:.3f}"
                        for name, value in current_controls.telemetry.items()
                    }
                )
                if "expert_diversity" in current_controls.capabilities:
                    postfix["div"] = f"{diversity_lambda:.1e}"
                pbar.set_postfix(postfix)
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
