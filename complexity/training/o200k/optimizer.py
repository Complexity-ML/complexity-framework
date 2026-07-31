"""Optimizer construction for o200k pretraining."""

from __future__ import annotations

import torch


def build_optimizer(args, raw_model):
    """Build AdamW or MuonTR for the o200k TR runner."""

    if args.optimizer == "adamw":
        expert_lr_scale = float(getattr(args, "expert_lr_scale", 1.0))
        shared_lr_scale = float(getattr(args, "shared_lr_scale", 1.0))
        if expert_lr_scale <= 0.0 or shared_lr_scale <= 0.0:
            raise ValueError(
                "expert_lr_scale and shared_lr_scale must be positive"
            )

        buckets = {
            ("base", True): [],
            ("base", False): [],
            ("shared", True): [],
            ("shared", False): [],
            ("expert", True): [],
            ("expert", False): [],
        }
        for name, p in raw_model.named_parameters():
            if not p.requires_grad:
                continue
            is_routed_expert = ".mlp." in name and name.endswith(
                (
                    "gate_proj_w",
                    "up_proj_w",
                    "down_proj_w",
                    "hash_pair_gate_logits",
                    "hash_channel_scale",
                )
            )
            is_shared_expert = ".mlp.shared_" in name
            kind = (
                "expert"
                if is_routed_expert
                else "shared"
                if is_shared_expert
                else "base"
            )
            use_decay = not (
                p.ndim < 2
                or "bias" in name
                or "norm" in name
                or name.endswith("hash_channel_scale")
            )
            buckets[(kind, use_decay)].append(p)

        scales = {
            "base": 1.0,
            "shared": shared_lr_scale,
            "expert": expert_lr_scale,
        }
        param_groups = []
        for kind in ("base", "shared", "expert"):
            for use_decay in (True, False):
                params = buckets[(kind, use_decay)]
                if not params:
                    continue
                param_groups.append(
                    {
                        "params": params,
                        "lr": args.lr * scales[kind],
                        "weight_decay": (
                            args.weight_decay if use_decay else 0.0
                        ),
                        "group_name": kind,
                        "lr_scale": scales[kind],
                    }
                )
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
        counts = {
            kind: sum(
                p.numel()
                for use_decay in (True, False)
                for p in buckets[(kind, use_decay)]
            )
            for kind in ("base", "shared", "expert")
        }
        return optimizer, {
            "adamw_params": sum(counts.values()),
            "adamw_impl": adamw_impl,
            "adamw_base_params": counts["base"],
            "adamw_shared_params": counts["shared"],
            "adamw_expert_params": counts["expert"],
            "expert_lr_scale": expert_lr_scale,
            "shared_lr_scale": shared_lr_scale,
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


def optimizer_group_lrs(optimizer) -> dict[str, float]:
    """Return current optimizer LRs keyed by semantic parameter group."""

    values: dict[str, list[float]] = {}
    for group in optimizer.param_groups:
        values.setdefault(str(group.get("group_name", "base")), []).append(
            float(group["lr"])
        )
    return {
        name: sum(group_values) / len(group_values)
        for name, group_values in values.items()
        if group_values
    }
