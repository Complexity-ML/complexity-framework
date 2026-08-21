#!/usr/bin/env python3
"""Deploy the PIQA-selected full-SFT root bundle to the TR-Hash Tiny Space."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

SPACE_ID = "Pacific-i64/TR-hash-tiny"
OLD_MODEL = "AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement"
NEW_MODEL = "AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT"
TOKEN_FILE = Path("/workspace/.hf_token")

README = """---
title: TR-Hash Tiny
emoji: 🧭
colorFrom: purple
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-4.0
models:
  - AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT
---

# TR-Hash Tiny · TR-HASH MoE 200M Full SFT

OpenAI-compatible chat API serving the PIQA-selected epoch-2 root weights from
[`AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT`](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT)
with [`Complexity-ML/TR-Hash-i64`](https://github.com/Complexity-ML/TR-Hash-i64).

This is a full-parameter SFT, not a LoRA adapter. All 201.2M parameters were
trained for three epochs on the audited Luciole 16-way mixture. Epoch 2 scores
**68.82% PIQA accuracy** and **69.31% length-normalized accuracy** on the full
1,838-example validation set.

The launcher intentionally downloads only the release files at the model-repo
root. The three large resumable `step_*` training folders are excluded. The
bundled `chat_template.jinja` and tokenizer are loaded beside the single root
`model.safetensors`.

```bash
curl https://pacific-i64-tr-hash-tiny.hf.space/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "tr-hash-moe-200m",
    "messages": [{"role": "user", "content": "Explain why 17 × 23 − 14 equals 377."}],
    "max_tokens": 384,
    "temperature": 0.4,
    "top_k": 30,
    "top_p": 0.85,
    "repetition_penalty": 1.1,
    "stream": false
  }'
```

The CPU deployment uses continuous batching, paged KV caching, prefix caching,
streaming and dynamic INT8 packing of linear layers. Set the Space variable
`CPU_INT8=false` to serve floating-point weights.

Endpoints: `/health`, `/v1/models`, `/v1/completions`,
`/v1/chat/completions`, `/v1/metrics`, `/v1/monitor`, `/v1/experts`.
"""


def download_text(name: str, token: str) -> str:
    path = hf_hub_download(
        SPACE_ID,
        name,
        repo_type="space",
        token=token,
        force_download=True,
    )
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("empty Hugging Face token")
    api = HfApi(token=token)

    dockerfile = download_text("Dockerfile", token)
    if OLD_MODEL not in dockerfile:
        raise RuntimeError("Dockerfile does not target the expected refinement model")
    dockerfile = dockerfile.replace(OLD_MODEL, NEW_MODEL)

    app = download_text("app.py", token)
    if OLD_MODEL not in app:
        raise RuntimeError("app.py does not target the expected refinement model")
    app = app.replace(OLD_MODEL, NEW_MODEL)
    marker = '    "model_config.yaml",\n'
    additions = '    "chat_template.json",\n    "chat_template.jinja",\n'
    if additions not in app:
        if marker not in app:
            raise RuntimeError("MODEL_FILES insertion marker is missing")
        app = app.replace(marker, additions + marker, 1)
    app = app.replace(
        "TR-HASH MoE 200M refinement checkpoint is loading.",
        "TR-HASH MoE 200M full-SFT checkpoint is loading.",
    )

    api.create_commit(
        repo_id=SPACE_ID,
        repo_type="space",
        operations=[
            CommitOperationAdd(
                path_in_repo="Dockerfile", path_or_fileobj=dockerfile.encode()
            ),
            CommitOperationAdd(path_in_repo="app.py", path_or_fileobj=app.encode()),
            CommitOperationAdd(
                path_in_repo="README.md", path_or_fileobj=README.encode()
            ),
        ],
        commit_message="Deploy PIQA-selected full-SFT epoch 2 root model",
    )

    deployed = download_text("app.py", token)
    if NEW_MODEL not in deployed or OLD_MODEL in deployed:
        raise RuntimeError("Space model target verification failed")
    if '"chat_template.jinja"' not in deployed:
        raise RuntimeError("Space root-file filter omits the chat template")
    print(f"Space now targets only root release files from {NEW_MODEL}", flush=True)


if __name__ == "__main__":
    main()
