from __future__ import annotations

import json

import torch
from safetensors import safe_open

from complexity.inference.chat_template import default_chat_template
from scripts.export_reasoning_sft_release import export_release


def test_export_reasoning_release_is_f32_and_preserves_routing_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step_000250"
    checkpoint.mkdir(parents=True)
    torch.save(
        {
            "step": 250,
            "model": {
                "embed_tokens.weight": torch.randn(16, 8, dtype=torch.bfloat16),
                "layers.0.mlp.engine.route_table": torch.tensor(
                    [[0, 1, 2, 3] * 4, [1, 2, 3, 0] * 4], dtype=torch.long
                ),
            },
            "config": {
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 8,
                "shared_intermediate_size": 16,
                "vocab_size": 16,
                "max_position_embeddings": 32,
                "attention_type": "gqa",
                "mlp_type": "tr_hash_engine",
                "num_experts": 4,
                "top_k": 2,
                "top_k_primary_weight": 0.5,
                "routing_strategy": "token_id_multi_hash",
                "route_hash_count": 2,
            },
            "chat_template": default_chat_template(),
        },
        checkpoint / "checkpoint.pt",
    )
    summary = {
        "release_ready": True,
        "selection_policy": "full PIQA test policy",
        "selected": {
            "step": 250,
            "checkpoint": str(checkpoint),
            "piqa_acc": 0.68,
            "piqa_acc_norm": 0.69,
            "matched_eval_loss": 1.0,
            "matched_eval_ppl": 2.72,
        },
        "candidates": [
            {
                "step": 250,
                "piqa_acc": 0.68,
                "piqa_acc_norm": 0.69,
                "matched_eval_loss": 1.0,
                "matched_eval_ppl": 2.72,
            }
        ],
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("step,matched_eval_loss\n250,1.0\n")
    evaluations = tmp_path / "evaluations"
    evaluations.mkdir()
    (evaluations / "summary.json").write_text(json.dumps(summary))
    source_arc = {
        "benchmarks": {
            "arc_easy": {"acc": 0.50, "acc_norm": 0.51},
            "arc_challenge": {"acc": 0.25, "acc_norm": 0.26},
        },
        "combined": {"acc": 0.42, "acc_norm": 0.43},
    }
    selected_arc = {
        "benchmarks": {
            "arc_easy": {"acc": 0.52, "acc_norm": 0.53},
            "arc_challenge": {"acc": 0.27, "acc_norm": 0.28},
        },
        "combined": {"acc": 0.44, "acc_norm": 0.45},
    }
    (evaluations / "source_arc_zero_shot_full.json").write_text(json.dumps(source_arc))
    (evaluations / "selected_arc_zero_shot_full.json").write_text(json.dumps(selected_arc))
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": "</s>",
                "added_tokens_decoder": {"0": {"content": "</s>"}},
            }
        )
    )
    dataset_audit = tmp_path / "release-audit.json"
    dataset_audit.write_text('{"status":"passed"}\n')
    output = tmp_path / "release"

    manifest = export_release(
        summary_path=summary_path,
        metrics_path=metrics,
        evaluation_root=evaluations,
        tokenizer_dir=tokenizer,
        dataset_audit=dataset_audit,
        output=output,
    )

    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "tr_hash_moe"
    assert config["top_k"] == 2
    assert config["num_experts_per_tok"] == 2
    assert config["torch_dtype"] == "float32"
    assert manifest["weights_dtype"] == "float32"
    assert manifest["selected_step"] == 250
    assert manifest["weights_sha256"]
    with safe_open(output / "model.safetensors", framework="pt") as handle:
        assert handle.get_tensor("embed_tokens.weight").dtype == torch.float32
        assert "layers.0.mlp.engine.route_table" in handle.keys()
    readme = (output / "README.md").read_text()
    assert "500,000,669 unique formatted tokens" in readme
    assert "does not force or fabricate a hidden `<think>` block" in readme
    assert "ARC zero-shot retention" in readme
    assert "| Combined ARC | 42.00% | 44.00% | 43.00% | 45.00% |" in readme
    assert manifest["arc_zero_shot"]["source"]["combined"]["acc"] == 0.42
