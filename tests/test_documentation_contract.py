"""Regression tests that keep current documentation aligned with executable code."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import unquote

from complexity.generative.detection import TRHashDetectorConfig, TRHashObjectDetector
from complexity.tr_hash import TRHashEngineConfig, supports_fused_cuda

ROOT = Path(__file__).resolve().parents[1]
CURRENT_MARKDOWN = (
    ROOT / "README.md",
    ROOT / "RESULTS_100M_MPS.md",
    ROOT / "TR_GQA.md",
    ROOT / "TR_MHA.md",
    *sorted((ROOT / "docs").glob("*.md")),
)


def _project_scripts() -> set[str]:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    return set(re.findall(r"^([\w-]+)\s*=", section, flags=re.MULTILINE))


def _shell_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r"```(?:bash|shell)\n(.*?)```", text, flags=re.DOTALL)


def test_current_markdown_local_links_resolve() -> None:
    missing: list[str] = []
    pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")

    for document in CURRENT_MARKDOWN:
        for raw_target in pattern.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip().strip("<>")
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not target:
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {raw_target}")

    assert not missing, "missing local Markdown targets:\n" + "\n".join(missing)


def test_current_documentation_python_examples_parse() -> None:
    invalid: list[str] = []
    for document in CURRENT_MARKDOWN:
        text = document.read_text(encoding="utf-8")
        for block_index, source in enumerate(
            re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL),
            start=1,
        ):
            try:
                ast.parse(source)
            except SyntaxError as error:
                invalid.append(
                    f"{document.relative_to(ROOT)} block {block_index}: "
                    f"line {error.lineno}: {error.msg}"
                )

    assert not invalid, "invalid documented Python examples:\n" + "\n".join(invalid)


def test_api_guide_lists_every_packaged_console_script() -> None:
    api = (ROOT / "docs/api.md").read_text(encoding="utf-8")
    missing = sorted(name for name in _project_scripts() if name not in api)
    assert not missing, f"console scripts missing from docs/api.md: {missing}"


def test_documented_training_commands_include_required_optimizer() -> None:
    guide = ROOT / "docs/tr-hash-object-detection.md"
    blocks = _shell_blocks(guide)
    training_blocks = [
        block for block in blocks if "cf-vision-pretrain" in block or "cf-detector-train" in block
    ]
    assert training_blocks
    for block in training_blocks:
        assert "--optimizer musgd" in block


def test_text_to_image_guide_uses_current_generator_flags_and_batch_math() -> None:
    guide = (ROOT / "docs/tr-hash-text-to-image.md").read_text(encoding="utf-8")
    assert "--batch-size 2" in guide
    assert "--guidance-scale 4.0" in guide
    assert "--output-dir samples" in guide
    assert "--guidance 4.0" not in guide
    assert "--output sample.png" not in guide


def test_documented_fused_cuda_expert_range_matches_runtime() -> None:
    guide = (ROOT / "docs/tr-hash-engine.md").read_text(encoding="utf-8")
    assert "two to eight experts" in guide
    for experts in (2, 4, 8):
        config = TRHashEngineConfig(
            hidden_size=16,
            vocab_size=32,
            num_experts=experts,
            top_k=2,
            expert_width=8,
        )
        assert supports_fused_cuda(config)

    unsupported = TRHashEngineConfig(
        hidden_size=16,
        vocab_size=32,
        num_experts=16,
        top_k=2,
        expert_width=8,
    )
    assert not supports_fused_cuda(unsupported)


def test_detector_parameter_claim_matches_documented_configuration() -> None:
    config = TRHashDetectorConfig(
        image_size=128,
        patch_size=8,
        vision_hidden_size=128,
        vision_layers=4,
        vision_heads=4,
        vision_num_experts=4,
        vision_shared_width=0,
        vision_expert_width=48,
        vision_stage_depths=(1, 1, 2),
        vision_window_size=8,
        num_classes=20,
        p2_head=True,
        end_to_end=True,
    )
    parameters = sum(parameter.numel() for parameter in TRHashObjectDetector(config).parameters())
    guide = (ROOT / "docs/tr-hash-object-detection.md").read_text(encoding="utf-8")
    assert parameters == 927_246
    assert "927,246 parameters" in guide


def test_supported_python_floor_is_consistent() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    getting_started = (ROOT / "docs/getting-started.md").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10"' in pyproject
    assert 'readme = "README.md"' in pyproject
    assert "Python 3.10 or newer" in getting_started


def test_current_repository_map_and_doc_index_are_complete() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    index = (ROOT / "docs/index.md").read_text(encoding="utf-8")
    assert "deploy/" in readme
    assert "spaces/                      Hugging Face Space wrappers" not in readme
    assert "tr_hash_sensor_fusion.md" in index
    assert "TR_HASH_DETECTOR_SPECIALIZATION.md" in index


def test_hugging_face_org_card_targets_current_organization() -> None:
    card = (ROOT / "docs/huggingface-org-card.md").read_text(encoding="utf-8")
    assert "title: AETHORIA-AI" in card
    assert "https://huggingface.co/AETHORIA-AI" in card


def test_current_200m_docs_report_released_sft_v2_epoch_3() -> None:
    documents = {
        path: path.read_text(encoding="utf-8")
        for path in (
            ROOT / "README.md",
            ROOT / "docs/index.md",
            ROOT / "docs/tr-hash-200m-release.md",
            ROOT / "docs/tr-hash-200m-clean-sft-v2.md",
            ROOT / "docs/huggingface-org-card.md",
        )
    }
    combined = "\n".join(documents.values())

    for expected in ("0.959617", "2.61", "68.01%", "69.10%"):
        assert expected in combined
    assert "step 5,982" in combined

    stale_release_claims = (
        "238.9M supervised tokens",
        "69.31% normalized accuracy",
        "epoch 2 / step 926",
        "prepared, not trained",
        "Candidate recipe, not a released model result",
    )
    for path, text in documents.items():
        for stale in stale_release_claims:
            assert stale not in text, f"stale SFT v1 claim in {path}: {stale}"
