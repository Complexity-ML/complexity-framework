from __future__ import annotations

from scripts.check_lora_promotion import audit_lora_promotion


def _piqa(score: float) -> dict:
    return {"benchmarks": {"piqa": {"acc_norm": score}}}


def _chat(*, template: bool = True, arithmetic: str = "377") -> dict:
    responses = {
        "greeting": "Hello! How can I help?",
        "casual_reflection": (
            "That moment of fluency sounds rewarding. What made this sentence "
            "feel more natural than before?"
        ),
        "arithmetic": f"The workshop has {arithmetic} usable components.",
        "simple_arithmetic": "2 + 2 = 4.",
        "summarization": (
            "The team reduced validation frequency, increased the per-GPU batch "
            "size, and retained exact checkpoints for resumable training."
        ),
        "rewrite": "The run failed again; please investigate now because GPU time is costly.",
        "instruction_constraints": "- Test held-out prompts\n- Measure response repetition\n- Compare checkpoint quality",
        "factual": (
            "Precision is the share of predicted positive cases that are correct; "
            "recall is the share of actual positive cases found."
        ),
        "safety": "Do not share your password; contact official support instead.",
    }
    return {
        "chat_template_applied": template,
        "results": [
            {
                "id": key,
                "response": value,
                "repetition": {"repeated_trigram_ratio": 0.0},
            }
            for key, value in responses.items()
        ],
    }


def test_lora_promotion_accepts_capability_preserving_chat_adapter() -> None:
    audit = audit_lora_promotion(_piqa(0.6953), _piqa(0.6900), _chat())

    assert audit["passed"] is True
    assert audit["violations"] == []


def test_lora_promotion_rejects_measured_final_regression() -> None:
    audit = audit_lora_promotion(_piqa(0.6953), _piqa(0.6665), _chat())

    assert audit["passed"] is False
    assert any(item.startswith("piqa_acc_norm_drop=") for item in audit["violations"])


def test_lora_promotion_requires_template_and_behavior_anchors() -> None:
    audit = audit_lora_promotion(
        _piqa(0.6953),
        _piqa(0.6953),
        _chat(template=False, arithmetic="17"),
    )

    assert audit["passed"] is False
    assert "official_chat_template_not_applied" in audit["violations"]
    assert "arithmetic_anchor_missing_377" in audit["violations"]


def test_lora_promotion_rejects_visible_panel_failures() -> None:
    report = _chat()
    bad = {
        "simple_arithmetic": "2 + 2 is equal to 2.",
        "casual_reflection": "I'm sure you will find it useful.",
        "rewrite": "For the run, GPU costs money.",
        "instruction_constraints": "The first bullet is about testing.",
        "factual": "Precision is accurate and recall is correct.",
    }
    for result in report["results"]:
        if result["id"] in bad:
            result["response"] = bad[result["id"]]

    audit = audit_lora_promotion(_piqa(0.6953), _piqa(0.6953), report)

    assert audit["passed"] is False
    assert "simple_arithmetic_anchor_missing_4" in audit["violations"]
    assert "casual_reflection_requires_one_question" in audit["violations"]
    assert "rewrite_missing_request_or_facts" in audit["violations"]
    assert "instruction_constraints_not_three_short_bullets" in audit["violations"]
    assert "factual_definition_incomplete" in audit["violations"]


def test_lora_promotion_rejects_prompt_echo_and_fabricated_greeting_identity() -> None:
    report = _chat()
    bad = {
        "greeting": "Hello, I'm a user at the University of California, Berkeley.",
        "rewrite": (
            "Rewrite this message so it is clear and professional: "
            "'hey the run broke again can u look now because gpu costs money'."
        ),
    }
    for result in report["results"]:
        if result["id"] in bad:
            result["response"] = bad[result["id"]]

    audit = audit_lora_promotion(_piqa(0.6953), _piqa(0.6953), report)

    assert audit["passed"] is False
    assert "greeting_not_neutral_or_fabricates_identity" in audit["violations"]
    assert "rewrite_missing_request_or_facts" in audit["violations"]
