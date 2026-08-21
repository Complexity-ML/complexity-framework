from __future__ import annotations

import json

import torch
from safetensors import safe_open

from complexity.inference.chat_template import default_chat_template
from scripts.export_sft_v2_release import export_release


def test_export_release_writes_transformers_and_native_routing_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step_000200"
    checkpoint.mkdir(parents=True)
    state = {
        "step": 200,
        "model": {
            "embed_tokens.weight": torch.randn(16, 8),
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
    }
    torch.save(state, checkpoint / "checkpoint.pt")
    summary = {
        "release_ready": True,
        "selection_policy": "test policy",
        "selected": {
            "epoch": 2,
            "step": 200,
            "checkpoint": str(checkpoint),
            "promotion_passed": True,
            "piqa_acc": 0.69,
            "piqa_acc_norm": 0.70,
            "matched_eval_loss": 1.2,
            "matched_eval_ppl": 3.32,
        },
        "candidates": [
            {
                "epoch": epoch,
                "step": step,
                "promotion_passed": True,
                "piqa_acc": 0.68 + epoch / 100,
                "piqa_acc_norm": 0.69 + epoch / 100,
                "matched_eval_loss": 1.4 - epoch / 10,
                "matched_eval_ppl": 3.5 - epoch / 10,
            }
            for epoch, step in enumerate((100, 200, 300), start=1)
        ],
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("step,matched_eval_loss\n200,1.2\n")
    evaluations = tmp_path / "evaluations"
    evaluations.mkdir()
    (evaluations / "summary.json").write_text(json.dumps(summary))
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
    output = tmp_path / "release"

    manifest = export_release(
        summary_path=summary_path,
        metrics_path=metrics,
        evaluation_root=evaluations,
        tokenizer_dir=tokenizer,
        output=output,
    )

    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "tr_hash_moe"
    assert config["architectures"] == ["TRHashForCausalLM"]
    assert config["top_k"] == 2
    assert config["num_experts_per_tok"] == 2
    assert config["eos_token_id"] == 0
    assert config["torch_dtype"] == "float32"
    assert manifest["selected_epoch"] == 2
    assert manifest["behavior_gate_passed"] is True
    assert manifest["weights_sha256"]
    assert manifest["weights_dtype"] == "float32"
    assert manifest["floating_parameters"] == 16 * 8
    with safe_open(output / "model.safetensors", framework="pt") as checkpoint_file:
        assert checkpoint_file.metadata()["format"] == "pt"
        assert "layers.0.mlp.engine.route_table" in checkpoint_file.keys()
    assert (output / "configuration_tr_hash_moe.py").is_file()
    assert (output / "chat_template.jinja").is_file()
    assert "Apache-2.0" in (output / "README.md").read_text()
