"""Native SFT example formatting and bounded deterministic generators.

These helpers never fetch datasets, execute generated code, publish releases,
compile a corpus or start training. Historical category names are provenance
labels, not evidence of model quality. Budgets and source pins stay in explicit
historical example recipes, outside these generators.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .sft_preparation import _encoded, compact_json, stable_hash

LEGACY_MARKERS = re.compile(r"</?(?:think|final)>|<TOOLCALL>|</TOOLCALL>")
LEADING_LEGACY_THINK = re.compile(r"(?is)^\s*<think>.*?</think>\s*")
NATIVE_MARKERS = re.compile(
    r"<\|(?:system|user|assistant|end_of_turn|think_start|think_end|final_start|final_end|tool_call_start|tool_call_end|tool_result_start|tool_result_end)\|>"
)
THINK_RE = re.compile(r"<\|think_start\|>(.*?)<\|think_end\|>", re.DOTALL)
END_OF_TURN = "<|end_of_turn|>"


def _nonnegative_index(index: int) -> None:
    if type(index) is not int or index < 0:
        raise ValueError("generator index must be a non-negative integer")


def final_completion(answer: str) -> str:
    return f"<|final_start|>{answer.strip()}<|final_end|>{END_OF_TURN}"


def thinking_completion(reasoning: str, answer: str) -> str:
    return f"<|think_start|>{reasoning.strip()}<|think_end|>" + final_completion(answer)


def tool_call_completion(name: str, arguments: dict[str, Any], reasoning: str = "") -> str:
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        raise ValueError("tool call requires a name and argument object")
    prefix = f"<|think_start|>{reasoning.strip()}<|think_end|>" if reasoning.strip() else ""
    return (
        prefix
        + "<|tool_call_start|>"
        + compact_json({"name": name, "arguments": arguments})
        + "<|tool_call_end|>"
        + END_OF_TURN
    )


def validate_target_completion(completion: str) -> dict[str, Any] | None:
    """Check exactly one native assistant action, optional reasoning and turn end."""
    if (
        not isinstance(completion, str)
        or completion.count(END_OF_TURN) != 1
        or not completion.endswith(END_OF_TURN)
    ):
        raise ValueError("target requires exactly one terminal end_of_turn")
    body = completion[: -len(END_OF_TURN)]
    if "<|think_start|>" in body or "<|think_end|>" in body:
        match = THINK_RE.match(body)
        if match is None or body.count("<|think_start|>") != 1 or body.count("<|think_end|>") != 1:
            raise ValueError("unbalanced or repeated thinking markers")
        if not match.group(1).strip() or NATIVE_MARKERS.search(match.group(1)):
            raise ValueError("thinking trace is empty or contains control markers")
        body = body[match.end() :]
    if body.startswith("<|tool_call_start|>") and body.endswith("<|tool_call_end|>"):
        payload = body[len("<|tool_call_start|>") : -len("<|tool_call_end|>")]
        if NATIVE_MARKERS.search(payload):
            raise ValueError("tool JSON contains control markers")
        try:
            call = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("tool call requires valid JSON") from exc
        if (
            not isinstance(call, dict)
            or set(call) != {"name", "arguments"}
            or not isinstance(call["name"], str)
            or not call["name"]
            or not isinstance(call["arguments"], dict)
        ):
            raise ValueError("tool call requires one canonical name and argument object")
        return call
    if body.startswith("<|final_start|>") and body.endswith("<|final_end|>"):
        answer = body[len("<|final_start|>") : -len("<|final_end|>")]
        if not answer.strip() or NATIVE_MARKERS.search(answer):
            raise ValueError("final answer is empty or contains control markers")
        return None
    raise ValueError("target requires exactly one complete call or final envelope")


def project_thinking(completion: str, *, thinking: bool) -> str:
    validate_target_completion(completion)
    if type(thinking) is not bool:
        raise ValueError("thinking policy must be a boolean")
    trace = THINK_RE.match(completion)
    if thinking and trace is None:
        raise ValueError("selected row has no thinking trace")
    return completion if thinking or trace is None else completion[trace.end() :]


def token_count(tokenizer: Any, prompt: str, completion: str) -> int:
    """Count native bodies without automatic BOS/EOS, then one owned EOS."""
    prompt_ids, completion_ids = _encoded(tokenizer, prompt), _encoded(tokenizer, completion)
    eos = getattr(tokenizer, "eos_token_id", None)
    engine = getattr(tokenizer, "_tokenizer", tokenizer)
    if eos is None and hasattr(engine, "token_to_id"):
        eos = engine.token_to_id("</s>")
    if eos is not None and eos in prompt_ids + completion_ids:
        raise ValueError("native record bodies must not already contain EOS")
    return len(prompt_ids) + len(completion_ids) + 1


def normalize_record(
    record: dict[str, Any], *, category: str, thinking: bool, tokenizer: Any, max_tokens: int = 2049
) -> dict[str, Any]:
    prompt = str(record["prompt"])
    completion = project_thinking(str(record["completion"]), thinking=thinking)
    count = token_count(tokenizer, prompt, completion)
    if type(max_tokens) is not int or max_tokens < 2 or count > max_tokens:
        raise ValueError(f"projected row exceeds context: {count}")
    source_id = str(record.get("source_id") or "")
    if not prompt or not source_id:
        raise ValueError("normalization requires prompt and source identity")
    record_hash = hashlib.sha256((prompt + "\0" + completion).encode()).hexdigest()[:16]
    return {
        "prompt": prompt,
        "completion": completion,
        "source_dataset": str(record.get("source_dataset") or "unspecified"),
        "source_id": f"{source_id}:{record_hash}",
        "source_conversation_id": source_id,
        "source_subset": str(record.get("source_subset") or "default"),
        "split_origin": str(record.get("split_origin") or "stable_hash"),
        "category": category,
        "thinking_supervised": thinking,
        "token_count": count,
    }


def user_prompt(content: str, system: str = "") -> str:
    prefix = f"<|system|>{system}{END_OF_TURN}" if system else ""
    return f"{prefix}<|user|>{content}{END_OF_TURN}<|assistant|>"


def stable_digest(value: str, salt: str) -> bytes:
    return hashlib.sha256(f"{salt}\0{value}".encode()).digest()


def normalized_hash(text: str) -> str:
    words = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()
    return hashlib.sha256(words.encode()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    score: bytes
    record: dict[str, Any]


def calculator_record(index: int, category: str, tokenizer: Any, *, max_tokens: int = 2049):
    """Format one archived calculator example; callers own collection budgets."""
    _nonnegative_index(index)
    if category not in {"calculator_tool_call", "calculator_tool_final"}:
        raise ValueError("unsupported calculator record category")
    expression, answer = calculator_values(index + (0 if category.endswith("call") else 100000))
    prompt_text = (
        f"Use the calculator to evaluate {expression}."
        if index % 2
        else f"Calculate {expression} accurately."
    )
    call = json.dumps(
        {"name": "calculator", "arguments": {"expression": expression}}, separators=(",", ":")
    )
    if category == "calculator_tool_call":
        prompt = user_prompt(prompt_text)
        completion = f"<|tool_call_start|>{call}<|tool_call_end|>{END_OF_TURN}"
    else:
        prompt = (
            user_prompt(prompt_text)
            + f"<|tool_call_start|>{call}<|tool_call_end|>{END_OF_TURN}"
            + f"<|tool_result_start|>{answer}<|tool_result_end|>{END_OF_TURN}<|assistant|>"
        )
        completion = final_completion(f"The result is {answer}.")
    validate_target_completion(completion)
    count = token_count(tokenizer, prompt, completion)
    if type(max_tokens) is not int or max_tokens < 2:
        raise ValueError("max_tokens must be an integer greater than one")
    if count > max_tokens:
        return None
    source_id = f"calculator-{category}-{index + 1:07d}"
    return {
        "prompt": prompt,
        "completion": completion,
        "source_dataset": "Complexity-ML/verified-calculator-generator-v1",
        "source_id": source_id,
        "source_conversation_id": source_id,
        "source_subset": category,
        "split_origin": "stable_sha256_v1",
        "category": category,
        "thinking_supervised": False,
        "token_count": count,
    }


def collect_calculator(
    tokenizer: Any, *, count: int, max_tokens: int = 2049, max_attempts: int | None = None
):
    """Collect a caller-sized, deterministic sample; never infer recipe quotas."""
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    attempts = max(count * 20, 100) if max_attempts is None else max_attempts
    if type(attempts) is not int or attempts < count:
        raise ValueError("max_attempts must be an integer at least count")
    result = {"calculator_tool_call": [], "calculator_tool_final": []}
    for category in result:
        seen = set()
        for index in range(attempts):
            record = calculator_record(index, category, tokenizer, max_tokens=max_tokens)
            if record is None:
                continue
            digest = normalized_hash(record["prompt"])
            if digest in seen:
                continue
            seen.add(digest)
            result[category].append(Candidate(stable_digest(record["source_id"], category), record))
            if len(result[category]) == count:
                break
        if len(result[category]) != count:
            raise ValueError(f"Insufficient valid {category} examples within max_attempts")
    return result


def evaluate_arithmetic(expression: str) -> int | float:
    """Evaluate the small arithmetic grammar without eval, imports or code execution."""
    if not isinstance(expression, str) or len(expression) > 128:
        raise ValueError("calculator expression is invalid or too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("unsupported calculator expression") from exc
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("calculator expression is too complex")
    operators = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
    }

    def visit(node, depth=0):
        if depth > 16:
            raise ValueError("calculator expression is too deep")
        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in operators:
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 16:
                raise ValueError("calculator exponent is too large")
            value = operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            inner = visit(node.operand, depth + 1)
            value = inner if isinstance(node.op, ast.UAdd) else -inner
        else:
            raise ValueError("unsupported calculator expression")
        if type(value) not in {int, float} or not math.isfinite(value) or abs(value) > 1e15:
            raise ValueError("calculator value is outside the supported range")
        return value

    try:
        return visit(tree)
    except (ArithmeticError, OverflowError) as exc:
        raise ValueError("calculator arithmetic is invalid") from exc


def _offered_schemas(system: str) -> dict[str, dict[str, Any]]:
    if not system.startswith("Available tools:\n"):
        raise ValueError("agentic trajectory lacks the compact tool system")
    try:
        definitions = json.loads(system.split("\n", 1)[1])
    except json.JSONDecodeError as exc:
        raise ValueError("offered tools require valid JSON") from exc
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("offered tools require a nonempty list")
    schemas = {}
    for item in definitions:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("offered tool requires a named function")
        name = function["name"]
        parameters = function.get("parameters")
        if (
            not name.strip()
            or name != name.strip()
            or name in schemas
            or not isinstance(parameters, dict)
            or parameters.get("type") != "object"
        ):
            raise ValueError("offered tool schema is incomplete or duplicated")
        if not isinstance(parameters.get("properties", {}), dict) or not isinstance(
            parameters.get("required", []), list
        ):
            raise ValueError("offered tool argument schema is invalid")
        schemas[name] = parameters
    return schemas


def _validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> None:
    required = schema.get("required", [])
    if any(not isinstance(key, str) or key not in arguments for key in required):
        raise ValueError("tool call lacks a required argument")
    properties = schema.get("properties", {})
    for key, value in arguments.items():
        spec = properties.get(key)
        if spec is None:
            if schema.get("additionalProperties") is False:
                raise ValueError("tool call contains an unoffered argument")
            continue
        if not isinstance(spec, dict):
            raise ValueError("offered argument schema is invalid")
        checks = {
            "string": isinstance(value, str),
            "integer": type(value) is int,
            "number": type(value) in {int, float} and math.isfinite(value),
            "boolean": type(value) is bool,
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "null": value is None,
        }
        kind = spec.get("type")
        if kind is not None:
            kinds = kind if isinstance(kind, list) else [kind]
            if (
                not kinds
                or any(not isinstance(item, str) or item not in checks for item in kinds)
                or not any(checks[item] for item in kinds)
            ):
                raise ValueError(f"tool argument {key} does not match its offered type")
        if "enum" in spec and (not isinstance(spec["enum"], list) or value not in spec["enum"]):
            raise ValueError(f"tool argument {key} does not match its offered enum")


def validate_agentic_trajectory(
    trajectory: Any, *, tool_result_validators: Mapping[str, Any] | None = None
) -> None:
    """Validate action/result order, offered arguments, and calculator ground truth.

    Other tool-result semantics belong to caller-provided validators accepting
    (arguments, result_text). A text result alone does not prove factual truth.
    """
    if tool_result_validators is not None and (
        not isinstance(tool_result_validators, Mapping)
        or any(not callable(check) for check in tool_result_validators.values())
    ):
        raise ValueError("tool result validators must be a mapping of callable checks")
    if (
        not isinstance(trajectory, list)
        or len(trajectory) < 5
        or any(not isinstance(row, dict) for row in trajectory)
    ):
        raise ValueError("agentic trajectory is incomplete")
    if trajectory[0].get("role") != "system" or trajectory[1].get("role") != "user":
        raise ValueError("agentic trajectory must start with system and user turns")
    schemas = _offered_schemas(str(trajectory[0].get("content") or ""))
    if trajectory[-1].get("role") != "assistant":
        raise ValueError("agentic trajectory must end with an assistant answer")
    pending = None
    for index, message in enumerate(trajectory):
        role, content = message.get("role"), message.get("content")
        if (
            role not in {"system", "user", "assistant", "tool"}
            or not isinstance(content, str)
            or not content.strip()
        ):
            raise ValueError("agentic trajectory contains an invalid role or empty content")
        if role != "assistant" and (
            LEGACY_MARKERS.search(content) or NATIVE_MARKERS.search(content)
        ):
            raise ValueError("non-assistant trajectory turn contains a control marker")
        if role == "tool":
            if pending is None:
                raise ValueError("tool result has no preceding call")
            if message.get("name", pending["name"]) != pending["name"]:
                raise ValueError("tool result name does not match its call")
            if pending["name"] == "calculator":
                expected = evaluate_arithmetic(pending["arguments"].get("expression"))
                try:
                    observed = Decimal(content.strip())
                except InvalidOperation as exc:
                    raise ValueError("calculator result is not a finite number") from exc
                if not observed.is_finite() or observed != Decimal(str(expected)):
                    raise ValueError("calculator result does not match its expression")
            validator = (tool_result_validators or {}).get(pending["name"])
            if validator is not None:
                if validator(pending["arguments"], content) is False:
                    raise ValueError(f"tool result is invalid for {pending['name']}")
            pending = None
            continue
        if pending is not None:
            raise ValueError("tool call is not followed by a tool result")
        if role != "assistant":
            continue
        call = validate_target_completion(content)
        if call is None:
            if index != len(trajectory) - 1:
                raise ValueError("final answer appears before the end of the trajectory")
        else:
            if call["name"] not in schemas:
                raise ValueError("tool call does not match the offered schema")
            _validate_arguments(call["arguments"], schemas[call["name"]])
            pending = call
    if pending is not None:
        raise ValueError("tool call is not followed by a tool result")


def load_historical_recipe(path: str | Path) -> dict[str, Any]:
    """Read explicit historical budgets and immutable source pins, without fetching."""
    recipe = json.loads(Path(path).read_text())
    if (
        not isinstance(recipe, dict)
        or recipe.get("format") != "complexity-historical-sft-recipe-v1"
        or recipe.get("historical_reference") is not True
        or recipe.get("release_eligible") is not False
    ):
        raise ValueError("expected an explicitly historical SFT recipe")
    tokenizer, sources = recipe.get("tokenizer"), recipe.get("sources")
    if (
        not isinstance(tokenizer, dict)
        or not isinstance(sources, dict)
        or not sources
        or any(not isinstance(spec, dict) for spec in sources.values())
    ):
        raise ValueError("historical recipe requires tokenizer and source contracts")
    pins = [recipe.get("implementation_revision"), tokenizer.get("revision")]
    for spec in sources.values():
        pins.append(spec.get("revision"))
        for name in ("train_sha256", "eval_sha256"):
            if name in spec and not re.fullmatch(r"[0-9a-f]{64}", str(spec[name])):
                raise ValueError("historical source checksum is invalid")
    if "historical_starting_checkpoint" in recipe:
        if not isinstance(recipe["historical_starting_checkpoint"], dict):
            raise ValueError("historical starting checkpoint contract is invalid")
        pins.append(recipe["historical_starting_checkpoint"].get("revision"))
    benchmarks = recipe.get("protected_benchmarks", {})
    if not isinstance(benchmarks, dict) or any(
        not isinstance(spec, dict) for spec in benchmarks.values()
    ):
        raise ValueError("historical benchmark contracts are invalid")
    for spec in benchmarks.values():
        if "revision" in spec:
            pins.append(spec["revision"])
        if "sha256" in spec and not re.fullmatch(r"[0-9a-f]{64}", str(spec["sha256"])):
            raise ValueError("historical benchmark checksum is invalid")
    if any(not isinstance(pin, str) or not re.fullmatch(r"[0-9a-f]{40}", pin) for pin in pins):
        raise ValueError("historical sources require immutable 40-character revisions")
    for split in ("train", "eval"):
        quotas = recipe.get(f"{split}_quotas")
        if (
            not isinstance(quotas, dict)
            or not quotas
            or any(type(count) is not int or count < 1 for count in quotas.values())
        ):
            raise ValueError("historical quotas must be explicit positive integers")
        thinking = recipe.get(f"{split}_think_quotas", {})
        if not isinstance(thinking, dict) or any(
            key not in quotas or type(count) is not int or not 0 <= count <= quotas[key]
            for key, count in thinking.items()
        ):
            raise ValueError("historical thinking budget exceeds its category quota")
    if (
        type(recipe.get("sequence_length")) is not int
        or recipe["sequence_length"] < 1
        or recipe.get("max_tokens_including_eos") != recipe["sequence_length"] + 1
    ):
        raise ValueError("historical recipe requires an exact one-EOS token budget")
    return recipe


def agentic_category(record: dict[str, Any]) -> str | None:
    category = record.get("category")
    if category == "direct":
        return "no_call"
    if category in {"tool_call", "tool_final"}:
        return str(category)
    return None


def arithmetic_example(index: int) -> tuple[str, str, str]:
    _nonnegative_index(index)
    rng = random.Random(500_000 + index)
    kind = index % 10
    a, b = rng.randint(12, 999), rng.randint(3, 199)
    if kind == 0:
        question = f"A warehouse has {a} boxes and receives {b} more. How many boxes are there now?"
        reasoning, answer = f"Add the received boxes: {a} + {b} = {a + b}.", str(a + b)
    elif kind == 1:
        total, removed = a + b, b
        question = f"A shelf held {total} books. {removed} were removed. How many remain?"
        reasoning, answer = f"Subtract the removed books: {total} - {removed} = {a}.", str(a)
    elif kind == 2:
        question = f"There are {b} trays with {a} components on each tray. Find the total."
        reasoning, answer = (
            f"Multiply trays by components per tray: {b} × {a} = {a * b}.",
            str(a * b),
        )
    elif kind == 3:
        quotient, divisor = rng.randint(4, 500), rng.randint(2, 50)
        total = quotient * divisor
        question = (
            f"Split {total} items equally among {divisor} groups. How many items are in each group?"
        )
        reasoning, answer = f"Compute {total} ÷ {divisor} = {quotient}.", str(quotient)
    elif kind == 4:
        percent = rng.choice((5, 10, 15, 20, 25, 30, 40, 50, 60, 75))
        base = rng.randint(2, 200) * 20
        value = base * percent // 100
        question = f"What is {percent}% of {base}?"
        reasoning, answer = (
            f"Convert {percent}% to {percent}/100 and multiply: {base} × {percent}/100 = {value}.",
            str(value),
        )
    elif kind == 5:
        percent = rng.choice((10, 20, 25, 30, 40, 50))
        price = rng.randint(5, 300) * 20
        discount = price * percent // 100
        result = price - discount
        question = (
            f"An item costs ${price} and is discounted by {percent}%. What is the sale price?"
        )
        reasoning = f"The discount is {price} × {percent}/100 = {discount}. Subtract it: {price} - {discount} = {result}."
        answer = f"${result}"
    elif kind == 6:
        start = rng.randint(0, 23 * 60 + 30)
        duration = rng.randint(15, 240)
        end = (start + duration) % (24 * 60)
        question = f"An event starts at {start // 60:02d}:{start % 60:02d} and lasts {duration} minutes. When does it end?"
        reasoning = f"Add {duration} minutes to {start // 60:02d}:{start % 60:02d}; the ending time is {end // 60:02d}:{end % 60:02d}."
        answer = f"{end // 60:02d}:{end % 60:02d}"
    elif kind == 7:
        metres, centimetres = rng.randint(1, 80), rng.randint(0, 99)
        question = f"Convert {metres} metres and {centimetres} centimetres to centimetres."
        value = metres * 100 + centimetres
        reasoning, answer = (
            f"Each metre is 100 cm, so {metres} × 100 + {centimetres} = {value} cm.",
            f"{value} cm",
        )
    elif kind == 8:
        x, y = rng.randint(1, 500), rng.randint(1, 500)
        z = 3 * rng.randint(1, 500) - x - y
        question = f"Find the average of {x}, {y}, and {z}."
        value = (x + y + z) // 3
        reasoning, answer = (
            f"Add the values and divide by 3: ({x} + {y} + {z}) ÷ 3 = {value}.",
            str(value),
        )
    else:
        coefficient, solution, offset = rng.randint(2, 20), rng.randint(2, 100), rng.randint(1, 80)
        total = coefficient * solution + offset
        question = f"Solve for x: {coefficient}x + {offset} = {total}."
        reasoning = f"Subtract {offset}: {coefficient}x = {total - offset}. Divide by {coefficient}: x = {solution}."
        answer = str(solution)
    return question, reasoning, answer


def calculator_values(index: int) -> tuple[str, str]:
    _nonnegative_index(index)
    rng = random.Random(900_000 + index)
    kind = index % 4
    a, b = rng.randint(20, 20_000), rng.randint(2, 500)
    if kind == 0:
        return f"{a}+{b}", str(a + b)
    if kind == 1:
        return f"{a}*{b}", str(a * b)
    if kind == 2:
        c = rng.randint(2, 80)
        return f"({a}+{b})*{c}", str((a + b) * c)
    numerator = a * b
    return f"{numerator}/{b}", str(a)


TOOLS = {
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    "search_knowledge_base": {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the available knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    "date_time": {
        "type": "function",
        "function": {
            "name": "date_time",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}},
        },
    },
}


def tools_system(names: Iterable[str]) -> str:
    definitions = [TOOLS[name] for name in names]
    return "Available tools:\n" + compact_json(definitions)


def repeated_lines(text: str) -> bool:
    lines = [" ".join(line.casefold().split()) for line in text.splitlines() if line.strip()]
    return len(lines) >= 4 and len(set(lines)) / len(lines) < 0.65


def source_identity(repository: str, subset: str, row: dict[str, Any]) -> str:
    explicit = row.get("id") or row.get("uuid")
    identity = str(explicit) if explicit else stable_hash(row)[:32]
    return f"{repository}:{subset}:{identity}"


def parse_messages(messages: Any) -> tuple[str, str, str] | None:
    if not isinstance(messages, list) or not messages:
        return None
    assistant_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], dict)
            and str(messages[index].get("role", "")).lower() == "assistant"
        ),
        None,
    )
    if assistant_index is None:
        return None
    answer = str(messages[assistant_index].get("content") or "").strip()
    if (
        not answer
        or LEGACY_MARKERS.search(answer)
        or NATIVE_MARKERS.search(answer)
        or repeated_lines(answer)
    ):
        return None
    pieces: list[str] = []
    last_user = ""
    for message in messages[:assistant_index]:
        if not isinstance(message, dict):
            return None
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if LEGACY_MARKERS.search(content) or NATIVE_MARKERS.search(content):
            return None
        if role == "system":
            pieces.append(f"<|system|>{content}<|end_of_turn|>")
        elif role == "user":
            pieces.append(f"<|user|>{content}<|end_of_turn|>")
            last_user = content
        elif role == "assistant":
            pieces.append(f"<|assistant|>{final_completion(content)}")
        else:
            return None
    if not pieces or not last_user:
        return None
    return "".join(pieces) + "<|assistant|>", answer, last_user


def extract_python(text: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    code = max(blocks, key=len).strip() if blocks else text.strip()
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    return code


def code_record(
    row: dict[str, Any],
    *,
    repository: str,
    subset: str,
) -> dict[str, Any] | None:
    messages = row.get("messages")
    if isinstance(messages, list):
        messages = [dict(message) if isinstance(message, dict) else message for message in messages]
        assistant_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], dict)
                and str(messages[index].get("role", "")).lower() == "assistant"
            ),
            None,
        )
        if assistant_index is not None:
            content = str(messages[assistant_index].get("content") or "")
            messages[assistant_index]["content"] = LEADING_LEGACY_THINK.sub("", content)
    parsed = parse_messages(messages)
    if parsed is None:
        instruction = str(row.get("instruction") or "").strip()
        answer = LEADING_LEGACY_THINK.sub("", str(row.get("response") or "")).strip()
        if not instruction or not answer:
            return None
        prompt = f"<|user|>{instruction}<|end_of_turn|><|assistant|>"
    else:
        prompt, answer, last_user = parsed
        benchmark_text = last_user
    if parsed is None:
        benchmark_text = instruction
    if extract_python(answer) is None:
        return None
    source_id = source_identity(repository, subset, row)
    return {
        "prompt": prompt,
        "completion": final_completion(answer),
        "source_dataset": repository,
        "source_subset": subset,
        "source_id": source_id,
        "source_conversation_id": source_id,
        "category": "python_code",
        "thinking_supervised": False,
        "_benchmark_check_texts": [benchmark_text],
    }


def generated_reasoning_record(index: int) -> dict[str, Any]:
    _nonnegative_index(index)
    rng = random.Random(20_000_000 + index)
    a, b, c = rng.randint(12, 9_999), rng.randint(2, 199), rng.randint(1, 999)
    kind = index % 5
    if kind == 0:
        question = f"A depot receives {b} crates with {a} components each, then ships {c}. How many remain?"
        answer = a * b - c
        reasoning = f"Multiply the crate count by the components per crate, then subtract the shipment: {b}*{a}-{c}={answer}."
    elif kind == 1:
        total = a * b
        question = f"{total} samples are divided equally among {b} teams. How many samples does each team receive?"
        answer = a
        reasoning = f"Equal division gives {total}/{b}={answer} samples per team."
    elif kind == 2:
        question = f"A counter starts at {a}, increases by {b}, and then decreases by {c}. What is its final value?"
        answer = a + b - c
        reasoning = f"Apply both changes in order: {a}+{b}-{c}={answer}."
    elif kind == 3:
        question = f"Find the perimeter of a rectangle measuring {a} cm by {b} cm."
        answer = 2 * (a + b)
        reasoning = f"A rectangle has two sides of each length, so 2*({a}+{b})={answer} cm."
    else:
        percent = rng.choice((10, 20, 25, 40, 50, 75))
        base = a * 100
        answer = base * percent // 100
        question = f"For inventory batch B-{index}, what is {percent}% of {base}?"
        reasoning = f"Multiply by {percent}/100: {base}*{percent}/100={answer}."
    source_id = f"generated-reasoning-2m:{index:08d}"
    return {
        "prompt": f"<|user|>{question}<|end_of_turn|><|assistant|>",
        "completion": thinking_completion(reasoning, str(answer)),
        "source_dataset": "Complexity-ML/verified-reasoning-generator-2m-v1",
        "source_subset": "deterministic",
        "source_id": source_id,
        "source_conversation_id": source_id,
        "category": "verified_compact_reasoning",
        "thinking_supervised": True,
        "_benchmark_check_texts": [question],
    }


def generated_constraint_record(index: int) -> dict[str, Any]:
    _nonnegative_index(index)
    rng = random.Random(30_000_000 + index)
    left, right = rng.randint(10, 99_999), rng.randint(10, 99_999)
    mode = index % 4
    if mode == 0:
        prompt = f"Return only valid JSON with keys left, right, and total for {left}+{right}."
        answer = compact_json({"left": left, "right": right, "total": left + right})
    elif mode == 1:
        prompt = f"Answer with exactly three words confirming ticket {left}."
        answer = f"Ticket {left} confirmed"
    elif mode == 2:
        prompt = f"Write the numbers {left} and {right} separated only by a comma."
        answer = f"{left},{right}"
    else:
        prompt = f"Return one Markdown bullet stating that batch {left} contains {right} items."
        answer = f"- Batch {left} contains {right} items."
    source_id = f"generated-constraint-2m:{index:08d}"
    return {
        "prompt": f"<|user|>{prompt}<|end_of_turn|><|assistant|>",
        "completion": final_completion(answer),
        "source_dataset": "Complexity-ML/verified-constraints-generator-2m-v1",
        "source_subset": "deterministic",
        "source_id": source_id,
        "source_conversation_id": source_id,
        "category": "constraints_and_formats",
        "thinking_supervised": False,
        "_benchmark_check_texts": [prompt],
    }


def generated_code_record(index: int) -> dict[str, Any]:
    _nonnegative_index(index)
    rng = random.Random(35_000_000 + index)
    value = rng.randint(2, 10_000)
    mode = index % 5
    if mode == 0:
        name = f"scale_{index}"
        factor = rng.randint(2, 20)
        prompt = f"Write a Python function named {name} that multiplies an integer by {factor}."
        code = f"def {name}(value: int) -> int:\n    return value * {factor}"
    elif mode == 1:
        name = f"is_multiple_{index}"
        divisor = rng.randint(2, 19)
        prompt = (
            f"Write a Python function named {name} that returns whether an integer "
            f"is divisible by {divisor}."
        )
        code = f"def {name}(value: int) -> bool:\n    return value % {divisor} == 0"
    elif mode == 2:
        name = f"offset_values_{index}"
        offset = rng.randint(-50, 50)
        prompt = (
            f"Write a Python function named {name} that adds {offset} to every integer in a list."
        )
        code = (
            f"def {name}(values: list[int]) -> list[int]:\n"
            f"    return [value + ({offset}) for value in values]"
        )
    elif mode == 3:
        name = f"count_long_{index}"
        minimum = rng.randint(2, 12)
        prompt = (
            f"Write a Python function named {name} that counts strings whose length "
            f"is at least {minimum}."
        )
        code = (
            f"def {name}(values: list[str]) -> int:\n"
            f"    return sum(len(value) >= {minimum} for value in values)"
        )
    else:
        name = f"bounded_{index}"
        upper = value + rng.randint(1, 500)
        prompt = (
            f"Write a Python function named {name} that clamps an integer to the "
            f"inclusive range [{value}, {upper}]."
        )
        code = f"def {name}(number: int) -> int:\n    return max({value}, min(number, {upper}))"
    if extract_python(code) is None:
        raise AssertionError("deterministic Python generator produced invalid syntax")
    source_id = f"generated-python-2m:{index:08d}"
    return {
        "prompt": f"<|user|>{prompt}<|end_of_turn|><|assistant|>",
        "completion": final_completion(f"```python\n{code}\n```"),
        "source_dataset": "Complexity-ML/verified-python-generator-2m-v1",
        "source_subset": "deterministic",
        "source_id": source_id,
        "source_conversation_id": source_id,
        "category": "python_code",
        "thinking_supervised": False,
        "_benchmark_check_texts": [prompt],
    }


def agentic_record(index: int, subtype: str) -> dict[str, Any]:
    _nonnegative_index(index)
    if subtype not in {
        "single_tool_end_to_end",
        "multi_tool_end_to_end",
        "tool_failure_and_recovery",
        "reasoning_code_and_tools",
    }:
        raise ValueError("unknown agentic trajectory subtype")
    rng = random.Random(40_000_000 + index)
    a, b, c = rng.randint(12, 999), rng.randint(2, 99), rng.randint(1, 500)
    source_id = f"generated-agentic-2m:{subtype}:{index:08d}"
    if subtype == "single_tool_end_to_end":
        mode = index % 3
        if mode == 0:
            result = a * b - c
            expression = f"{a}*{b}-{c}"
            user = (
                f"For request CALC-{index}, a workshop packs {a} boxes of {b} parts, "
                f"then removes {c} parts. How many remain?"
            )
            call = tool_call_completion("calculator", {"expression": expression})
            tool_result, final = str(result), str(result)
            names = ["calculator", "search_knowledge_base"]
        elif mode == 1:
            project = f"Project-{index}-{a}-{b}"
            user = f"What is the verified owner of {project}?"
            call = tool_call_completion("search_knowledge_base", {"query": user})
            tool_result = compact_json({"project": project, "owner": f"Team {c}"})
            final = f"The verified owner of {project} is Team {c}."
            names = ["search_knowledge_base", "date_time"]
        else:
            timezone = "Europe/Paris" if index % 2 else "UTC"
            user = f"For request DT-{index}, what is the current date and time in {timezone}?"
            call = tool_call_completion("date_time", {"timezone": timezone})
            tool_result = f"2031-06-{index % 28 + 1:02d} 14:{index % 60:02d} {timezone}"
            final = tool_result
            names = ["date_time", "calculator"]
        trajectory = [
            {"role": "system", "content": tools_system(names)},
            {"role": "user", "content": user},
            {"role": "assistant", "content": call},
            {"role": "tool", "content": tool_result},
            {"role": "assistant", "content": final_completion(final)},
        ]
    elif subtype == "multi_tool_end_to_end":
        unit_price, quantity = a, b
        user = f"Find the unit price and quantity for order O-{index}, then calculate its total."
        first = tool_call_completion(
            "search_knowledge_base", {"query": f"order O-{index} unit price and quantity"}
        )
        found = compact_json({"unit_price": unit_price, "quantity": quantity})
        expression = f"{unit_price}*{quantity}"
        second = tool_call_completion("calculator", {"expression": expression})
        total = unit_price * quantity
        trajectory = [
            {
                "role": "system",
                "content": tools_system(["search_knowledge_base", "calculator"]),
            },
            {"role": "user", "content": user},
            {"role": "assistant", "content": first},
            {"role": "tool", "content": found},
            {"role": "assistant", "content": second},
            {"role": "tool", "content": str(total)},
            {"role": "assistant", "content": final_completion(f"Order O-{index} totals {total}.")},
        ]
    elif subtype == "tool_failure_and_recovery":
        broad = f"asset {index}-{a}"
        precise = f"asset {index}-{a} in register R-{b}"
        trajectory = [
            {"role": "system", "content": tools_system(["search_knowledge_base"])},
            {"role": "user", "content": f"Who owns {precise}?"},
            {
                "role": "assistant",
                "content": tool_call_completion("search_knowledge_base", {"query": broad}),
            },
            {"role": "tool", "content": "No unique match; provide the register identifier."},
            {
                "role": "assistant",
                "content": tool_call_completion("search_knowledge_base", {"query": precise}),
            },
            {"role": "tool", "content": compact_json({"owner": f"Operator-{c}"})},
            {
                "role": "assistant",
                "content": final_completion(f"{precise} is owned by Operator-{c}."),
            },
        ]
    else:
        expression = f"({a}+{b})*{c}"
        result = (a + b) * c
        reasoning = "The request requires adding the two quantities before multiplication."
        trajectory = [
            {"role": "system", "content": tools_system(["calculator"])},
            {
                "role": "user",
                "content": (
                    f"For request CALC-{index}, combine batches of {a} and {b} items, "
                    f"then make {c} identical sets. How many items are required?"
                ),
            },
            {
                "role": "assistant",
                "content": tool_call_completion(
                    "calculator", {"expression": expression}, reasoning=reasoning
                ),
            },
            {"role": "tool", "content": str(result)},
            {"role": "assistant", "content": final_completion(str(result))},
        ]
    return {
        "trajectory": trajectory,
        "source_dataset": "Complexity-ML/verified-agentic-trajectory-generator-2m-v1",
        "source_subset": subtype,
        "source_id": source_id,
        "source_conversation_id": source_id,
        "category": "complete_agentic_trajectories",
        "trajectory_subtype": subtype,
        "thinking_supervised": subtype == "reasoning_code_and_tools",
    }
