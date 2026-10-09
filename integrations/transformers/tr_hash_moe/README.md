# TR-HASH Hugging Face Transformers adapter

The canonical adapter lives in `complexity/export/transformers_adapter` and is
included in installed packages. The modules here preserve historical imports.
The two self-contained export files implement the public top-2 GQA family:

- `configuration_tr_hash_moe.py` registers `model_type = "tr_hash_moe"`;
- `modeling_tr_hash_moe.py` implements `TRHashForCausalLM`, legacy and modern
  KV caches, causal generation, the persisted multi-hash route tables, and the
  shared-plus-routed SwiGLU block.

The implementation intentionally keeps the native safetensors names. Exporting
a checkpoint does not transpose, rename, merge, or otherwise rewrite weights.

## Build a local Transformers bundle

```bash
python -m complexity.export.transformers \
  --config /path/to/config.json \
  --weights /path/to/model.safetensors \
  --tokenizer-dir /path/to/tokenizer \
  --output /path/to/hub-bundle
```

The exporter writes a separate local directory and does not publish it.
SafeTensors tensor bytes are retained; the file header adds Transformers
framework metadata, so the exported file hash may differ from its source.
Keep the source checkpoint unchanged.

The resulting directory can be loaded directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "/path/to/hub-bundle"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
)
output = model.generate(
    **tokenizer("Hello", return_tensors="pt"),
    max_new_tokens=32,
)
```

`num_experts_per_tok` is used for architectural top-k routing. It is kept
separate from Transformers' `top_k` sampling parameter so loading the model
does not silently change generation defaults.

The output must be a separate directory from the source checkpoint and tokenizer.
The exporter preserves tensor names and tensor bytes, adds Transformers metadata,
and rejects experimental model options unsupported by this adapter. The local
1B FSDP recipe belongs to the supported configuration family; small CPU parity
checks verify the adapter, rather than downloading or allocating the 1B model.
A compatible external serving runtime is still required for production serving.
