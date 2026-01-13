"""
Data Pipeline - Tokenization and preparation.
"""

import json
from pathlib import Path
from typing import List

from .dataset import Dataset


class DataPipeline:
    """Pipeline for data preparation."""

    def __init__(self, data_path: Path, output_path: Path, tokenizer, **kwargs):
        self.data_path = data_path
        self.output_path = output_path
        self.tokenizer = tokenizer
        self.seq_length = kwargs.get("seq_length", 2048)
        self.pack = kwargs.get("pack_sequences", True)

    def run(self, **kwargs):
        """Execute pipeline."""
        seq_len = kwargs.get("seq_length", self.seq_length)
        pack = kwargs.get("pack_sequences", self.pack)

        print(f"[Pipeline] {self.data_path} -> {self.output_path}")
        print(f"  Seq length: {seq_len}, Pack: {pack}")

        self.output_path.mkdir(parents=True, exist_ok=True)

        ds = Dataset.load(self.data_path)
        tokenized = []

        for i, item in enumerate(ds._data):
            text = item.get("text", "")
            if text:
                tokenized.append(self.tokenizer.encode(text))
            if (i + 1) % 10000 == 0:
                print(f"[Pipeline] Tokenized {i + 1}/{len(ds)}")

        if pack:
            packed = self._pack(tokenized, seq_len)
            print(f"[Pipeline] Packed: {len(tokenized)} -> {len(packed)}")
        else:
            packed = tokenized

        out = self.output_path / "train.jsonl"
        with open(out, "w") as f:
            for seq in packed:
                f.write(json.dumps({"input_ids": seq}) + "\n")

        print(f"[Pipeline] Saved to {out}")

    def _pack(self, tokenized: List[List[int]], seq_len: int) -> List[List[int]]:
        packed, current = [], []
        for tokens in tokenized:
            if len(current) + len(tokens) <= seq_len:
                current.extend(tokens)
            else:
                if current:
                    packed.append(current)
                current = tokens[:seq_len]
        if current:
            packed.append(current)
        return packed
