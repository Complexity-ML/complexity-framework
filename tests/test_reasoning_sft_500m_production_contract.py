from pathlib import Path


def test_reasoning_launcher_starts_from_refinement_and_resumes_only_own_run() -> None:
    launcher = Path("scripts/vast_sft_200m_reasoning_500m_full_1e.sh").read_text()
    assert 'BASE_CHECKPOINT="${BASE_CHECKPOINT:-/workspace/tr-hash-refinement}"' in launcher
    assert "--source-stage refinement" in launcher
    assert 'RESUME_ARGS=(--resume "${RESUME_FROM}")' in launcher
    assert '-path "${OUTPUT_ROOT}/step_*/*"' in launcher
    assert '"${RESUME_ARGS[@]}"' in launcher
    assert "best/ are intentionally excluded" in launcher
    assert "--epochs 1" in launcher
    assert '--save-steps "${SAVE_STEPS}"' in launcher
    assert '--eval-steps "${SAVE_STEPS}"' in launcher
    assert "--save-every-epoch" not in launcher
    assert "--eval-every-epoch" not in launcher
    assert "--save-total-limit 24" in launcher
    assert "--full-parameter" in launcher
    assert '--lr "${LR}"' in launcher
    assert 'LR="${LR:-5e-6}"' in launcher


def test_reasoning_launcher_requires_liger_and_no_fp32_loss() -> None:
    launcher = Path("scripts/vast_sft_200m_reasoning_500m_full_1e.sh").read_text()
    assert "COMPLEXITY_REQUIRE_LIGER=1" in launcher
    assert "liger_fused_linear_ce=required+available" in launcher
    assert "--sft-liger-loss" in launcher
    assert "--no-sft-fp32-loss" in launcher


def test_reasoning_launcher_checks_unique_token_contract() -> None:
    launcher = Path("scripts/vast_sft_200m_reasoning_500m_full_1e.sh").read_text()
    assert "500_000_000 <= actual < 500_020_000" in launcher
    assert '"no_truncation"' in launcher
    assert '"release_ready"' in launcher


def test_reasoning_supervisor_restarts_only_failed_processes() -> None:
    supervisor = Path("deploy/supervisor/tr_hash_200m_reasoning_sft_500m_full_1e.conf").read_text()
    assert "autorestart=unexpected" in supervisor
    assert "exitcodes=0" in supervisor
    assert "startretries=20" in supervisor
    assert "stopsignal=TERM" in supervisor
    assert 'BATCH_SIZE_PER_GPU="16"' in supervisor


def test_reasoning_checkpoint_sync_targets_dedicated_repo_and_keeps_resume_set() -> None:
    launcher = Path("scripts/vast_sync_200m_reasoning_sft_500m.sh").read_text()
    supervisor = Path("deploy/supervisor/tr_hash_200m_reasoning_sft_500m_hf_sync.conf").read_text()
    assert "AETHORIA-AI/TR-HASH-MoE-200M-160B-Reasoning-SFT" in launcher
    assert "training/reasoning-sft-500m/checkpoints" in launcher
    assert "--keep-local 24" in launcher
    assert "autorestart=unexpected" in supervisor


def test_low_lr_rerun_is_isolated_and_uploaded_separately() -> None:
    training = Path(
        "deploy/supervisor/tr_hash_200m_reasoning_sft_500m_lr5e7_full_1e.conf"
    ).read_text()
    sync = Path(
        "deploy/supervisor/tr_hash_200m_reasoning_sft_500m_lr5e7_hf_sync.conf"
    ).read_text()
    assert 'LR="5e-7"' in training
    assert (
        'OUTPUT_ROOT="artifacts/tr_hash_moe_200m_reasoning_sft_500m_lr5e7_full_1e"'
        in training
    )
    assert 'RUN_NAME="tr-hash-moe-200m-reasoning-sft-500m-lr5e7-full-1e"' in training
    assert 'HF_PATH_PREFIX="training/reasoning-sft-500m-lr5e7/checkpoints"' in sync


def test_reasoning_evaluator_waits_then_scores_every_checkpoint() -> None:
    launcher = Path("scripts/vast_eval_200m_reasoning_sft_500m.sh").read_text()
    supervisor = Path("deploy/supervisor/tr_hash_200m_reasoning_sft_500m_eval.conf").read_text()
    assert '.training_complete"' in launcher
    assert '-path "${CHECKPOINT_ROOT}/step_*/*"' in launcher
    assert "scripts.eval_torch_piqa" in launcher
    assert "scripts.eval_torch_arc_zero_shot" in launcher
    assert "source_arc_zero_shot_full.json" in launcher
    assert "selected_arc_zero_shot_full.json" in launcher
    assert "scripts.merge_arc_generative_shards" in launcher
    assert "scripts.promote_reasoning_sft_checkpoint" in launcher
    assert "add_candidate step250" in launcher
    assert "add_candidate step500" in launcher
    assert "add_candidate piqa" in launcher
    assert "add_candidate final" in launcher
    assert "scripts.select_reasoning_sft_checkpoint" in launcher
    assert "scripts.eval_arc_generative" in launcher
    assert "scripts.eval_torch_chat_panel" in launcher
    assert "scripts.upload_reasoning_sft_evaluations" in launcher
    assert "scripts.export_reasoning_sft_release" in launcher
    assert "scripts.publish_reasoning_sft_release" in launcher
    assert ".evaluation_complete" in launcher
    assert "autorestart=unexpected" in supervisor


def test_low_lr_evaluator_waits_for_isolated_run_and_publishes_best() -> None:
    supervisor = Path(
        "deploy/supervisor/tr_hash_200m_reasoning_sft_500m_lr5e7_eval.conf"
    ).read_text()
    assert 'CHECKPOINT_ROOT="artifacts/tr_hash_moe_200m_reasoning_sft_500m_lr5e7_full_1e"' in supervisor
    assert 'ARC_SAMPLES_PER_TASK="32"' in supervisor
    assert 'RELEASE_ROOT="artifacts/releases/tr_hash_moe_200m_reasoning_sft_500m_lr5e7"' in supervisor
    assert "autorestart=unexpected" in supervisor


def test_reasoning_bootstrap_downloads_audited_inputs_and_installs_supervisor() -> None:
    launcher = Path("scripts/vast_prepare_200m_reasoning_sft_500m.sh").read_text()
    assert "AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement" in launcher
    assert "ad4e9217b637720fb939babe8c8ce285a804ade2" in launcher
    assert "AETHORIA-AI/TR-HASH-MoE-200M-Reasoning-SFT-500M" in launcher
    assert "ba07ae135e4a8bdb6daf4cea30f4bc04d1a93033" in launcher
    assert "500_000_000 <= actual < 500_020_000" in launcher
    assert "Dataset file verification failed" in launcher
    assert "liger_fused_linear_ce" in launcher
    assert "tr_hash_200m_reasoning_sft_500m_full_1e" in launcher
    assert "tr_hash_200m_reasoning_sft_500m_hf_sync" in launcher
    assert "tr_hash_200m_reasoning_sft_500m_eval" in launcher
