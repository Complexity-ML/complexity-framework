#!/usr/bin/env python3
"""Deterministic promotion gate for clean full-parameter SFT candidates."""

from __future__ import annotations

import argparse
import ast
import json
import re
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    if name != "math":
        raise ImportError(f"regression sandbox only permits math, got {name!r}")
    return __import__(name, *args, **kwargs)


SAFE_BUILTINS = {
    "__import__": _safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


@contextmanager
def _time_limit(seconds: float):
    def expired(signum: int, frame: Any) -> None:
        del signum, frame
        raise TimeoutError("generated code exceeded the regression time limit")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text, flags=re.UNICODE)


def extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.I | re.S)
    return (blocks[0] if blocks else text).strip()


def run_python_cases(text: str, function_name: str, cases: list[dict[str, Any]]) -> str | None:
    code = extract_python(text)
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return f"invalid_python:{error.msg}"
    forbidden = (ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith)
    if any(isinstance(node, forbidden) for node in ast.walk(tree)):
        return "unsafe_python_construct"
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    for node in imports:
        modules = (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [str(node.module)]
        )
        if modules != ["math"]:
            return "unsafe_python_import"
    if any(
        isinstance(node, ast.Attribute) and str(node.attr).startswith("__")
        for node in ast.walk(tree)
    ):
        return "unsafe_dunder_access"
    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    try:
        with _time_limit(1.0):
            exec(compile(tree, "<model-response>", "exec"), namespace, namespace)
    except Exception as error:  # The generated program itself is the test subject.
        return f"python_exec_error:{type(error).__name__}"
    function = namespace.get(function_name)
    if not callable(function):
        return f"missing_function:{function_name}"
    for index, case in enumerate(cases):
        try:
            with _time_limit(1.0):
                actual = function(*case["args"])
        except Exception as error:
            return f"case_{index}_raised:{type(error).__name__}"
        if actual != case["expected"]:
            return f"case_{index}_expected_{case['expected']!r}_got_{actual!r}"
    return None


def check_response(item: dict[str, Any], response: str) -> list[str]:
    checks = item.get("checks", {})
    folded = response.casefold()
    failures: list[str] = []
    if not response.strip():
        return ["empty_response"]
    if "required_all" in checks:
        missing = [value for value in checks["required_all"] if str(value).casefold() not in folded]
        if missing:
            failures.append("missing_all:" + ",".join(map(str, missing)))
    if "required_any" in checks and not any(
        str(value).casefold() in folded for value in checks["required_any"]
    ):
        failures.append("missing_any:" + ",".join(map(str, checks["required_any"])))
    for number in checks.get("required_numbers", []):
        if re.search(rf"(?<!\d){re.escape(str(number))}(?!\d)", response) is None:
            failures.append(f"missing_number:{number}")
    if "max_words" in checks and len(_words(response)) > int(checks["max_words"]):
        failures.append(f"too_many_words:{len(_words(response))}")
    if "exact_bullets" in checks:
        bullets = [line for line in response.splitlines() if re.match(r"^\s*[-*]\s+\S", line)]
        if len(bullets) != int(checks["exact_bullets"]):
            failures.append(f"bullet_count:{len(bullets)}")
        too_long = [
            index for index, line in enumerate(bullets)
            if len(_words(re.sub(r"^\s*[-*]\s+", "", line))) > int(checks["max_words_per_bullet"])
        ]
        if too_long:
            failures.append("long_bullets:" + ",".join(map(str, too_long)))
    if "python_function" in checks:
        failure = run_python_cases(response, checks["python_function"], checks["cases"])
        if failure:
            failures.append(failure)
    return failures


def audit_regression(
    panel: dict[str, Any],
    chat_report: dict[str, Any],
    piqa_report: dict[str, Any],
) -> dict[str, Any]:
    responses = {str(item["id"]): str(item.get("response", "")) for item in chat_report["results"]}
    failures: dict[str, list[str]] = {}
    for item in panel["prompts"]:
        item_id = str(item["id"])
        item_failures = check_response(item, responses.get(item_id, ""))
        if item_failures:
            failures[item_id] = item_failures
    score = float(piqa_report["benchmarks"]["piqa"]["acc_norm"])
    piqa = panel["piqa"]
    threshold = float(piqa["baseline_acc_norm"]) - float(piqa["maximum_absolute_drop"])
    if score < threshold:
        failures["piqa"] = [f"acc_norm:{score:.6f}<threshold:{threshold:.6f}"]
    if chat_report.get("chat_template_applied") is not True:
        failures["chat_template"] = ["official_chat_template_not_applied"]
    return {
        "passed": not failures,
        "panel_id": panel["id"],
        "piqa_acc_norm": score,
        "piqa_threshold": threshold,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--chat-report", type=Path, required=True)
    parser.add_argument("--piqa-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = audit_regression(_read(args.panel), _read(args.chat_report), _read(args.piqa_report))
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
