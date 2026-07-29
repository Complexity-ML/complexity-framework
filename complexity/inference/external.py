"""
External inference backends for Complexity checkpoints.

Complexity trains and exports PyTorch models; production text generation is
intentionally delegated to serving runtimes such as vLLM or SGLang rather than
implemented as a native ``model.generate`` loop in the framework.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ExternalBackendName = Literal["vllm", "sglang"]


@dataclass
class ExternalGenerationConfig:
    """OpenAI-compatible generation parameters for vLLM/SGLang servers."""

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    stop: Optional[List[str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.stop:
            payload["stop"] = self.stop
        payload.update(self.extra)
        return payload


class OpenAICompatibleBackend:
    """Small stdlib client for vLLM/SGLang OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        backend: ExternalBackendName,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, prompt: str, config: Optional[ExternalGenerationConfig] = None) -> str:
        """Generate text through ``/v1/completions``."""
        cfg = config or ExternalGenerationConfig()
        payload = {
            "model": self.model,
            "prompt": prompt,
            **cfg.to_payload(),
        }
        response = self._post_json("/v1/completions", payload)
        try:
            return response["choices"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Invalid {self.backend} completion response: {response!r}") from exc

    def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[ExternalGenerationConfig] = None,
    ) -> str:
        """Generate chat completion through ``/v1/chat/completions``."""
        cfg = config or ExternalGenerationConfig()
        payload = {
            "model": self.model,
            "messages": messages,
            **cfg.to_payload(),
        }
        response = self._post_json("/v1/chat/completions", payload)
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Invalid {self.backend} chat response: {response!r}") from exc

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(f"{self.base_url}{path}", data=data, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.backend} request failed with HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not reach {self.backend} server at {self.base_url}. "
                "Start vLLM or SGLang with an OpenAI-compatible API endpoint first."
            ) from exc


def create_external_backend(
    backend: ExternalBackendName,
    *,
    base_url: str,
    model: str,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
) -> OpenAICompatibleBackend:
    """Create a vLLM/SGLang OpenAI-compatible inference client."""
    if backend not in ("vllm", "sglang"):
        raise ValueError("backend must be 'vllm' or 'sglang'")
    return OpenAICompatibleBackend(
        backend=backend,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
