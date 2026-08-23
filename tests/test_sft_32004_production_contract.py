from pathlib import Path


def test_sft_32004_launcher_is_full_parameter_and_fresh() -> None:
    launcher = Path("scripts/vast_sft_200m_32004_full_3e.sh").read_text()

    assert "--full-parameter" in launcher
    assert "--lora-" not in launcher
    assert "--epochs 3" in launcher
    assert "--seq-len 2048" in launcher
    assert "--require-release-ready" in launcher
    assert "COMPLEXITY_REQUIRE_LIGER=1" in launcher
    assert "RESUME_FROM is forbidden" in launcher
    assert '--checkpoint "${BASE_CHECKPOINT}"' in launcher
    assert "--source-stage refinement" in launcher


def test_sft_32004_launcher_guards_vocab_template_and_marker_supervision() -> None:
    launcher = Path("scripts/vast_sft_200m_32004_full_3e.sh").read_text()

    assert 'config.get("vocab_size", -1)) == 32_004' in launcher
    assert 'manifest.get("tokenizer_vocab_size") == 32_004' in launcher
    assert 'train.get("vocab_size") == 32_004' in launcher
    assert 'manifest.get("chat_template_id") == "complexity-chat-v3-32004"' in launcher
    assert 'manifest.get("special_token_ids") == special' in launcher
    assert 'train.get("special_token_label_counts")' in launcher
    assert "token: examples for token in special" in launcher
    assert 'get("token_truncation") is False' in launcher


def test_sft_32004_launcher_refuses_stale_outputs() -> None:
    launcher = Path("scripts/vast_sft_200m_32004_full_3e.sh").read_text()

    for artifact in ("step_*", "best", "final", "interrupted_*", "token_pack_*"):
        assert artifact in launcher
    assert "refuses stale training artifact" in launcher
    assert ".training_complete" in launcher
