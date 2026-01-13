"""
Dataset - Load, process, iterate over training data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Iterator, Callable

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader, IterableDataset

from .config import DataConfig


class Dataset:
    """
    Dataset flexible.

    Examples:
        # Simple
        ds = Dataset.load("./train.jsonl")

        # Override
        ds = Dataset.load("./train.jsonl", seq_length=4096, text_column="content")

        # Iterate
        for batch in ds.batches(batch_size=16):
            model(batch["input_ids"])
    """

    def __init__(self, data: List[Dict], tokenizer=None, config: DataConfig = None, **kwargs):
        self._data = data
        self._tokenizer = tokenizer
        self._config = config or DataConfig(**kwargs)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict:
        item = self._data[idx].copy()

        if self._tokenizer:
            text = item.get(self._config.text_column, "")
            if isinstance(text, str):
                tokens = self._tokenizer.encode(text, add_special_tokens=True)
                if len(tokens) > self._config.seq_length:
                    tokens = tokens[:self._config.seq_length]
                item["input_ids"] = torch.tensor(tokens, dtype=torch.long)
                item["attention_mask"] = torch.ones(len(tokens), dtype=torch.long)
                item["labels"] = item["input_ids"].clone()

        return item

    # ==================== Load ====================

    @classmethod
    def load(cls, path: Union[str, Path], tokenizer=None, config: DataConfig = None, **kwargs) -> "Dataset":
        """Load dataset from file or directory."""
        if config:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        else:
            config = DataConfig(**kwargs)

        path = Path(path)

        if path.is_dir():
            data = cls._load_dir(path, kwargs.get("limit"))
        elif path.suffix == ".jsonl":
            data = cls._load_jsonl(path, kwargs.get("limit"))
        elif path.suffix == ".json":
            data = cls._load_json(path, kwargs.get("limit"))
        elif path.suffix in [".txt", ".md"]:
            data = cls._load_text(path, kwargs.get("limit"))
        else:
            raise ValueError(f"Unknown format: {path.suffix}")

        print(f"[Dataset] Loaded {len(data):,} examples from {path}")
        return cls(data, tokenizer=tokenizer, config=config)

    @classmethod
    def from_config(cls, config: Union[Dict, DataConfig], tokenizer=None, **kwargs) -> "Dataset":
        """Load from config dict."""
        if isinstance(config, dict):
            path = config.pop("data_path", config.pop("path", None))
            cfg = DataConfig(**config)
        else:
            path = kwargs.pop("path", None)
            cfg = config

        if not path:
            raise ValueError("data_path required in config")

        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        return cls.load(path, tokenizer=tokenizer, config=cfg)

    @classmethod
    def streaming(cls, path: Union[str, Path], tokenizer=None, **kwargs) -> "StreamingDataset":
        """Streaming dataset for large files."""
        config = DataConfig(**kwargs)
        print(f"[Dataset] Streaming from {path}")
        return StreamingDataset(Path(path), tokenizer=tokenizer, config=config)

    # ==================== Iteration ====================

    def batches(self, **kwargs) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate by batches."""
        loader = self.get_dataloader(**kwargs)
        for batch in loader:
            yield batch

    def get_dataloader(self, **kwargs) -> DataLoader:
        """Get PyTorch DataLoader."""
        batch_size = kwargs.get("batch_size", self._config.batch_size)
        shuffle = kwargs.get("shuffle", self._config.shuffle)
        num_workers = kwargs.get("num_workers", self._config.num_workers)

        return DataLoader(
            _Wrapper(self),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate,
        )

    # ==================== Transform ====================

    def filter(self, fn: Callable[[Dict], bool], **kwargs) -> "Dataset":
        """Filter dataset."""
        new_data = [x for x in self._data if fn(x)]
        print(f"[Dataset] Filtered: {len(self._data)} -> {len(new_data)}")
        return Dataset(new_data, tokenizer=self._tokenizer, config=self._config, **kwargs)

    def map(self, fn: Callable[[Dict], Dict], **kwargs) -> "Dataset":
        """Map function over dataset."""
        new_data = [fn(x) for x in self._data]
        return Dataset(new_data, tokenizer=self._tokenizer, config=self._config, **kwargs)

    def shuffle(self, seed: int = None, **kwargs) -> "Dataset":
        """Shuffle dataset."""
        import random
        data = self._data.copy()
        if seed:
            random.seed(seed)
        random.shuffle(data)
        return Dataset(data, tokenizer=self._tokenizer, config=self._config, **kwargs)

    def split(self, ratio: float = 0.1, seed: int = None, **kwargs) -> tuple["Dataset", "Dataset"]:
        """Split into train/val."""
        import random
        data = self._data.copy()
        if seed:
            random.seed(seed)
        random.shuffle(data)

        idx = int(len(data) * (1 - ratio))
        train = data[:idx]
        val = data[idx:]

        print(f"[Dataset] Split: {len(train)} train, {len(val)} val")
        return (
            Dataset(train, tokenizer=self._tokenizer, config=self._config, **kwargs),
            Dataset(val, tokenizer=self._tokenizer, config=self._config, **kwargs),
        )

    # ==================== Save ====================

    def save(self, path: Union[str, Path], format: str = "jsonl"):
        """Save dataset."""
        path = Path(path)
        if format == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for item in self._data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        print(f"[Dataset] Saved to {path}")

    # ==================== Private ====================

    @staticmethod
    def _load_jsonl(path: Path, limit: int = None) -> List[Dict]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                if line.strip():
                    data.append(json.loads(line))
        return data

    @staticmethod
    def _load_json(path: Path, limit: int = None) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ["data", "examples", "items"]:
                if key in data:
                    data = data[key]
                    break
        return data[:limit] if limit else data

    @staticmethod
    def _load_text(path: Path, limit: int = None) -> List[Dict]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                if line.strip():
                    data.append({"text": line.strip()})
        return data

    @staticmethod
    def _load_dir(path: Path, limit: int = None) -> List[Dict]:
        data = []
        for ext in ["*.jsonl", "*.json", "*.txt"]:
            for f in path.glob(f"**/{ext}"):
                if ext == "*.jsonl":
                    data.extend(Dataset._load_jsonl(f))
                elif ext == "*.json":
                    data.extend(Dataset._load_json(f))
                else:
                    data.extend(Dataset._load_text(f))
                if limit and len(data) >= limit:
                    return data[:limit]
        return data

    def _collate(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        if not batch or "input_ids" not in batch[0]:
            return {}

        ids = [x["input_ids"] for x in batch]
        max_len = max(t.shape[0] for t in ids)

        padded_ids, masks = [], []
        for x in batch:
            t = x["input_ids"]
            m = x.get("attention_mask", torch.ones_like(t))
            pad = max_len - t.shape[0]
            if pad > 0:
                t = torch.cat([t, torch.zeros(pad, dtype=t.dtype)])
                m = torch.cat([m, torch.zeros(pad, dtype=m.dtype)])
            padded_ids.append(t)
            masks.append(m)

        result = {"input_ids": torch.stack(padded_ids), "attention_mask": torch.stack(masks)}

        if "labels" in batch[0]:
            labels = []
            for x in batch:
                l = x["labels"]
                pad = max_len - l.shape[0]
                if pad > 0:
                    l = torch.cat([l, torch.full((pad,), -100, dtype=l.dtype)])
                labels.append(l)
            result["labels"] = torch.stack(labels)

        return result

    def __repr__(self):
        return f"Dataset(examples={len(self._data)}, seq_length={self._config.seq_length})"


class StreamingDataset(IterableDataset):
    """Streaming for large files."""

    def __init__(self, path: Path, tokenizer=None, config: DataConfig = None):
        self._path = path
        self._tokenizer = tokenizer
        self._config = config or DataConfig()

    def __iter__(self):
        files = list(self._path.glob("**/*.jsonl")) if self._path.is_dir() else [self._path]
        for f in files:
            with open(f, "r", encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        item = json.loads(line)
                        yield self._process(item)

    def _process(self, item: Dict) -> Dict:
        if self._tokenizer:
            text = item.get(self._config.text_column, "")
            if text:
                tokens = self._tokenizer.encode(text)
                if len(tokens) > self._config.seq_length:
                    tokens = tokens[:self._config.seq_length]
                item["input_ids"] = torch.tensor(tokens, dtype=torch.long)
                item["attention_mask"] = torch.ones(len(tokens), dtype=torch.long)
        return item

    def batches(self, **kwargs):
        batch_size = kwargs.get("batch_size", self._config.batch_size)
        loader = DataLoader(self, batch_size=batch_size, collate_fn=self._collate)
        for batch in loader:
            yield batch

    def _collate(self, batch):
        if not batch or "input_ids" not in batch[0]:
            return {}
        ids = [x["input_ids"] for x in batch]
        max_len = max(t.shape[0] for t in ids)
        padded, masks = [], []
        for x in batch:
            t, m = x["input_ids"], x.get("attention_mask", torch.ones_like(x["input_ids"]))
            pad = max_len - t.shape[0]
            if pad > 0:
                t = torch.cat([t, torch.zeros(pad, dtype=t.dtype)])
                m = torch.cat([m, torch.zeros(pad, dtype=m.dtype)])
            padded.append(t)
            masks.append(m)
        return {"input_ids": torch.stack(padded), "attention_mask": torch.stack(masks)}


class _Wrapper(TorchDataset):
    def __init__(self, ds: Dataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]
