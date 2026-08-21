from pathlib import Path


def test_clean_v2_launcher_is_full_parameter_not_lora() -> None:
    launcher = Path("scripts/vast_sft_200m_clean_v2_full_3e.sh").read_text()

    assert "--full-parameter" in launcher
    assert "--lora-" not in launcher
    assert "--epochs 3" in launcher
    assert "--seq-len 2048" in launcher
    assert "--sft-bin" in launcher
    assert "--require-release-ready" in launcher
    assert "liger_fused_linear_ce=required+available" in launcher
    assert "COMPLEXITY_REQUIRE_LIGER=1" in launcher
    assert "--no-sft-fp32-loss" in launcher
    assert "--sft-liger-loss" in launcher
    assert "RESUME_FROM is not allowed" in launcher
    assert '.training_complete' in launcher
    assert 'NPROC_PER_NODE="${NPROC_PER_NODE:-${DETECTED_GPU_COUNT}}"' in launcher
    assert '"${NPROC_PER_NODE}" != "4" && "${NPROC_PER_NODE}" != "8"' in launcher
    assert 'BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"' in launcher
    supervisor = Path(
        "deploy/supervisor/tr_hash_200m_clean_sft_v2_full_3e.conf"
    ).read_text()
    assert 'BATCH_SIZE_PER_GPU="16"' in supervisor


def test_clean_v2_launcher_preflight_guards_dataset_contract() -> None:
    launcher = Path("scripts/vast_sft_200m_clean_v2_full_3e.sh").read_text()

    assert 'train.get("examples") == 300_000' in launcher
    assert 'manifest.get("tokenizer_vocab_size") == 32_000' in launcher
    assert 'manifest.get("sequence_length_cap") == 2_048' in launcher
    assert 'manifest.get("chat_template_eos_token") == "</s>"' in launcher
    assert 'get("token_truncation") is False' in launcher


def test_clean_v2_launcher_rejects_stale_training_artifacts() -> None:
    launcher = Path("scripts/vast_sft_200m_clean_v2_full_3e.sh").read_text()

    for artifact in ("step_*", "best", "final", "interrupted_*", "token_pack_*"):
        assert artifact in launcher
    assert "refuses stale training artifact" in launcher
    assert 'if [[ -e "${exact_artifact}" ]]' in launcher


def test_unweighted_liger_run_does_not_forward_neutral_collator_weights() -> None:
    trainer = Path("scripts/sft_500m_32k_tr.py").read_text()

    assert 'batch.get("loss_weight") if fp32_loss else None' in trainer
    assert 'batch.get("loss_weight") if args.sft_fp32_loss else None' in trainer


def test_clean_v2_bootstrap_is_revision_locked_and_minimal() -> None:
    bootstrap = Path("scripts/vast_prepare_200m_clean_sft_v2.sh").read_text()

    assert "ad4e9217b637720fb939babe8c8ce285a804ade2" in bootstrap
    assert "084a658ec47e4ee872f6d67fdbad3602f599424b" in bootstrap
    assert 'count not in {4, 8}' in bootstrap
    assert '"5090" in name' in bootstrap
    assert '"model.safetensors"' in bootstrap
    assert '"tokenized/tr-hash-32k-v2-2048/train/input_ids.bin"' in bootstrap
    assert '"tokenized/tr-hash-32k-v2-2048/train/labels.bin"' in bootstrap
    assert '"tokenized/tr-hash-32k-v2-2048/train/examples.jsonl"' in bootstrap
    assert '"tokenized/tr-hash-32k-v2-2048/eval/examples.jsonl"' in bootstrap
    assert "Dataset SHA256 verification failed" in bootstrap
    assert "physicaliqa-train-dev.zip" in bootstrap
    assert "examples != 1_838 or labels != 1_838" in bootstrap
    assert '"checkpoints"' not in bootstrap.split("allow_patterns=[", 1)[1].split("]", 1)[0]
    assert "optimizer" not in bootstrap.split("allow_patterns=[", 1)[1].split("]", 1)[0]
    assert "supervisorctl update" in bootstrap


def test_regression_runner_covers_chat_and_full_piqa() -> None:
    runner = Path("scripts/eval_sft_v2_regression.sh").read_text()

    assert "scripts.eval_torch_chat_panel" in runner
    assert "scripts.eval_torch_piqa" in runner
    assert "scripts.check_sft_v2_regression" in runner
    assert "--max-length 2048" in runner
    assert "PROMOTION_STRICT" in runner


def test_all_epoch_evaluator_waits_for_clean_completion_and_runs_three_candidates() -> None:
    evaluator = Path("scripts/vast_eval_200m_clean_sft_v2_all.sh").read_text()

    assert ".training_complete" in evaluator
    assert "Expected exactly 3 epoch checkpoints" in evaluator
    assert "CUDA_VISIBLE_DEVICES" in evaluator
    assert "scripts.select_sft_v2_checkpoint" in evaluator
    assert "scripts.upload_sft_v2_evaluations" in evaluator
    assert "scripts.export_sft_v2_release" in evaluator
    assert "scripts.publish_sft_v2_release" in evaluator
