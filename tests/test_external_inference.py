"""Tests for external vLLM/SGLang inference clients."""

import json

import pytest


def test_external_generation_config_payload_omits_none_top_k():
    from complexity.inference import ExternalGenerationConfig

    config = ExternalGenerationConfig(max_tokens=32, temperature=0.2, top_p=0.8)

    assert config.to_payload() == {
        "max_tokens": 32,
        "temperature": 0.2,
        "top_p": 0.8,
    }


def test_create_external_backend_validates_backend():
    from complexity.inference import create_external_backend

    with pytest.raises(ValueError, match="vllm.*sglang"):
        create_external_backend("torch", base_url="http://localhost:8000", model="m")  # type: ignore[arg-type]


def test_vllm_client_uses_openai_compatible_completion_endpoint(monkeypatch):
    from complexity.inference import ExternalGenerationConfig, create_external_backend
    import complexity.inference.external as external

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"choices": [{"text": " world"}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.headers.get("Authorization")
        return FakeResponse()

    monkeypatch.setattr(external, "urlopen", fake_urlopen)

    client = create_external_backend(
        "vllm",
        base_url="http://localhost:8000",
        model="complexity-model",
        api_key="token",
        timeout=3.0,
    )
    text = client.complete("Hello", ExternalGenerationConfig(max_tokens=4, top_k=16))

    assert text == " world"
    assert captured == {
        "url": "http://localhost:8000/v1/completions",
        "timeout": 3.0,
        "payload": {
            "model": "complexity-model",
            "prompt": "Hello",
            "max_tokens": 4,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 16,
        },
        "authorization": "Bearer token",
    }
