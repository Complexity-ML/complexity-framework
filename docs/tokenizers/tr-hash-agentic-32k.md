---
library_name: transformers
license: cc-by-nc-4.0
language:
- en
- fr
tags:
- tokenizer
- bpe
- byte-level
- tr-hash
- agentic
- tool-use
---

# TR-HASH Tokenizer 32K Agentic

Fresh 32,000-ID ByteLevel BPE tokenizer for the new TR-HASH agentic
pretraining line. It was trained on a balanced, filtered 2B-token sample:

| Bucket | Retained tokens |
|---|---:|
| General | 1,199,999,967 |
| Agentic | 799,999,922 |
| **Total** | **1,999,999,889** |

The general slice combines DCLM, FineWeb-Edu and Cosmopedia. The agentic
slice combines Stack-Edu, filtered FineWeb-Edu, FineMath and InfiWebMath.
Exact source quotas, record counts and input SHA-256 digests are recorded in
`selection_manifest.json`.

## Properties

- Vocabulary size: exactly 32,000 IDs, including special tokens
- Model: ByteLevel BPE
- Declared model context: 2,048 tokens
- Training format: `tr_hash_agentic_reasoning`
- Native chat template with reasoning and tool-use markers

The 20 reserved IDs are:

```text
0  <|begin|>              10 <|tool_result_start|>
1  <|end|>                11 <|tool_result_end|>
2  <|pad|>                12 <|plan_start|>
3  <|unk|>                13 <|plan_end|>
4  <|system|>             14 <|memory_start|>
5  <|user|>               15 <|memory_end|>
6  <|assistant|>          16 <|think_start|>
7  <|end_of_turn|>        17 <|think_end|>
8  <|tool_call_start|>    18 <|final_start|>
9  <|tool_call_end|>      19 <|final_end|>
```

## Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "AETHORIA-AI/TR-HASH-Tokenizer-32K-Agentic"
)

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Plan, execute, and verify the task."}],
    tokenize=False,
    add_generation_prompt=True,
)
```

With `add_generation_prompt=True`, the template ends with
`<|assistant|><|think_start|>`.

## Fixed-suite tokenizer benchmark

The reproducible suite in `benchmark_tiktoken.json` covers English, French,
code, JSON/tool payloads, mathematics and complex emoji sequences. Counts do
not include automatic BOS/EOS insertion.

### Natural text, code, JSON, math and emoji

| Encoding | Vocabulary | Tokens | Characters/token |
|---|---:|---:|---:|
| TR-HASH Agentic 32K | 32,000 | 520 | 2.815 |
| `r50k_base` | 50,257 | 532 | 2.752 |
| `cl100k_base` | 100,277 | 459 | 3.190 |
| `o200k_base` | 200,019 | 418 | 3.502 |

TR-HASH uses 2.3% fewer tokens than `r50k_base` on this suite. The modern
100K and 200K tiktoken vocabularies compress natural text better, at the cost
of much larger embedding tables.

### Including native TR-HASH agentic markers

| Encoding | Tokens | Characters/token |
|---|---:|---:|
| TR-HASH Agentic 32K | 586 | 3.215 |
| `r50k_base` | 681 | 2.767 |
| `cl100k_base` | 581 | 3.243 |
| `o200k_base` | 540 | 3.489 |

Native one-token control markers make TR-HASH 14.0% shorter than `r50k_base`
and within 0.9% of `cl100k_base` on the combined suite. `o200k_base` remains
8.5% shorter. These are fixed-suite measurements, not universal compression
claims.

Exact encode/decode round-trips pass for every category, including accented
French, emoji, flags, skin tones, zero-width-joiner families and all 20 native
markers.

```bash
pip install tokenizers tiktoken
python benchmark_tiktoken.py . --output benchmark_tiktoken.json
```

## Compatibility

This is a **new tokenizer**, not an extension of
`AETHORIA-AI/TR-HASH-Tokenizer-32K`. Token IDs were relearned from scratch.
It is therefore incompatible with all existing TR-HASH checkpoints and with
the previously tokenized 200B corpus.

Use it only for models pretrained from scratch on text retokenized with this
exact tokenizer revision. Do not resize, migrate or pair an existing TR-HASH
checkpoint with this vocabulary.

## Integrity

`agentic_tokenizer_manifest.json` records the format, vocabulary size,
special-token IDs and tokenizer-training contract. `selection_manifest.json`
records the selected source material and its SHA-256 lineage.
