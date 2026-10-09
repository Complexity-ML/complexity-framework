from complexity.inference.chat_template import render_jinja_messages
from complexity.tokenizer.training import CHAT_TEMPLATE

AGENTIC_RUNTIME_TEMPLATE = {
    "id": "tokenizer-jinja",
    "jinja_source": CHAT_TEMPLATE,
    "eos_token": "<|end|>",
}


def test_agentic_prompt_uses_tokenizer_native_generation_contract() -> None:
    prompt = render_jinja_messages(
        [
            {"role": "system", "content": "Answer carefully."},
            {"role": "user", "content": "What is two plus three?"},
        ],
        CHAT_TEMPLATE,
        eos_token=AGENTIC_RUNTIME_TEMPLATE["eos_token"],
        add_generation_prompt=True,
    )
    assert prompt == (
        "<|system|>Answer carefully.<|end_of_turn|>"
        "<|user|>What is two plus three?<|end_of_turn|>"
        "<|assistant|><|think_start|>"
    )


def test_agentic_messages_render_tool_result_before_next_assistant() -> None:
    prompt = render_jinja_messages(
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
        CHAT_TEMPLATE,
        eos_token=AGENTIC_RUNTIME_TEMPLATE["eos_token"],
        add_generation_prompt=True,
    )
    assert "<|tool_call_start|>" in prompt
    assert "<|tool_result_start|>5<|tool_result_end|><|end_of_turn|>" in prompt
    assert prompt.endswith("<|assistant|><|think_start|>")
