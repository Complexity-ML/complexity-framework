from scripts.chat_generate_local import build_messages_prompt, build_prompt
from scripts.train_tr_hash_agentic_tokenizer import CHAT_TEMPLATE


AGENTIC_RUNTIME_TEMPLATE = {
    "id": "tokenizer-jinja",
    "jinja_source": CHAT_TEMPLATE,
    "eos_token": "<|end|>",
}


def test_agentic_prompt_uses_tokenizer_native_generation_contract() -> None:
    prompt = build_prompt(
        "What is two plus three?",
        False,
        AGENTIC_RUNTIME_TEMPLATE,
        system_prompt="Answer carefully.",
    )
    assert prompt == (
        "<|system|>Answer carefully.<|end_of_turn|>"
        "<|user|>What is two plus three?<|end_of_turn|>"
        "<|assistant|><|think_start|>"
    )


def test_agentic_messages_render_tool_result_before_next_assistant() -> None:
    prompt = build_messages_prompt(
        [
            {"role": "user", "content": "Calculate it."},
            {
                "role": "assistant",
                "reasoning": "I should call the calculator.",
                "tool_calls": [{"name": "calculator", "arguments": {"expression": "2+3"}}],
                "content": "",
            },
            {"role": "tool", "content": "5"},
        ],
        AGENTIC_RUNTIME_TEMPLATE,
    )
    assert "<|tool_call_start|>" in prompt
    assert "<|tool_result_start|>5<|tool_result_end|><|end_of_turn|>" in prompt
    assert prompt.endswith("<|assistant|><|think_start|>")
