"""Optimizer construction for o200k pretraining."""

from __future__ import annotations

import torch


def build_optimizer(args, raw_model):
    """Build AdamW or MuonTR for the o200k TR runner."""

    if args.optimizer == "adamw":
        decay, no_decay = [], []
        for name, p in raw_model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim < 2 or "bias" in name or "norm" in name else decay).append(p)
        param_groups = [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        kwargs = {
            "lr": args.lr,
            "betas": (0.9, 0.95),
            "foreach": True,
        }
        try:
            optimizer = torch.optim.AdamW(param_groups, **kwargs)
            adamw_impl = "foreach"
        except TypeError:
            kwargs.pop("foreach", None)
            optimizer = torch.optim.AdamW(param_groups, **kwargs)
            adamw_impl = "default"
        return optimizer, {
            "adamw_params": sum(p.numel() for p in decay + no_decay),
            "adamw_impl": adamw_impl,
        }

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
