"""Generate from a complexity MLX checkpoint via mlx_lm (KV cache, sampling, etc.).

Wraps the tiktoken o200k_base tokenizer in a minimal HF-compatible adapter so
that mlx_lm.generate accepts it.

Usage:
    PYTHONPATH=/Users/boris/Dev/mlx-lm \\
    python mlx_generate.py <model_dir> --prompt "..." --max-tokens 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiktoken
from mlx_lm.generate import generate
from mlx_lm.utils import load_model


class TiktokenHFAdapter:
    """Minimal HF-tokenizer-like adapter around a tiktoken Encoding.

    Exposes just what mlx_lm.TokenizerWrapper + NaiveStreamingDetokenizer need.
    """

    def __init__(self, encoding: tiktoken.Encoding, eos_token: str = "<|endoftext|>"):
        self._enc = encoding
        self.eos_token = eos_token
        self.eos_token_id = encoding.encode(eos_token, allowed_special={eos_token})[0]
        self.bos_token = None
        self.bos_token_id = None
        self.pad_token = None
        self.pad_token_id = None
        self.chat_template = None
        self.clean_up_tokenization_spaces = False
        self._vocab = None

    def encode(self, text, add_special_tokens=True, **kwargs):
        return self._enc.encode(text, allowed_special="all")

    def decode(self, ids, **kwargs):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        # Filter out-of-vocab ids that tiktoken would reject.
        n = self._enc.n_vocab
        ids = [int(i) for i in ids if 0 <= int(i) < n]
        return self._enc.decode(ids)

    def get_vocab(self):
        if self._vocab is None:
            # BPE tokens via mergeable_ranks; specials separately.
            vocab = {
                tok.decode("utf-8", errors="replace"): rank
                for tok, rank in self._enc._mergeable_ranks.items()
            }
            vocab.update(self._enc._special_tokens)
            self._vocab = vocab
        return self._vocab

    def convert_tokens_to_ids(self, token):
        return self.get_vocab().get(token)


def load_tiktoken(model_dir: Path) -> TiktokenHFAdapter:
    with (model_dir / "tiktoken_config.json").open() as f:
        cfg = json.load(f)
    enc = tiktoken.get_encoding(cfg["encoding_name"])
    return TiktokenHFAdapter(enc, eos_token=cfg.get("eos_token", "<|endoftext|>"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    print(f"Loading model from {args.model_dir} ...")
    model, _ = load_model(args.model_dir)
    tokenizer = load_tiktoken(args.model_dir)
    print(f"Ready. eos_token_id={tokenizer.eos_token_id}\n")

    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p)

    generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        verbose=True,
    )


if __name__ == "__main__":
    main()
