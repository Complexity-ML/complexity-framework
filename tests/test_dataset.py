"""Tests for complexity.dataset module."""

import pytest
import torch
import tempfile
import json
from pathlib import Path


class TestDataConfig:
    """Test DataConfig."""

    def test_default_config(self):
        from complexity.dataset import DataConfig
        config = DataConfig()
        assert config.seq_length == 2048
        assert config.batch_size == 32
        assert config.text_column == "text"
        assert config.shuffle == True

    def test_custom_config(self):
        from complexity.dataset import DataConfig
        config = DataConfig(seq_length=4096, batch_size=16)
        assert config.seq_length == 4096
        assert config.batch_size == 16


class TestDataset:
    """Test Dataset class."""

    @pytest.fixture
    def sample_jsonl(self, tmp_path):
        """Create a sample JSONL file."""
        data = [
            {"text": "Hello world"},
            {"text": "This is a test"},
            {"text": "Machine learning is great"},
        ]
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return path

    def test_load_jsonl(self, sample_jsonl):
        from complexity.dataset import Dataset
        ds = Dataset.load(sample_jsonl)
        assert len(ds) == 3
        assert ds[0]["text"] == "Hello world"

    def test_filter(self, sample_jsonl):
        from complexity.dataset import Dataset
        ds = Dataset.load(sample_jsonl)
        filtered = ds.filter(lambda x: "test" in x["text"].lower())
        assert len(filtered) == 1

    def test_map(self, sample_jsonl):
        from complexity.dataset import Dataset
        ds = Dataset.load(sample_jsonl)
        mapped = ds.map(lambda x: {"text": x["text"].upper()})
        assert mapped[0]["text"] == "HELLO WORLD"

    def test_split(self, sample_jsonl):
        from complexity.dataset import Dataset
        ds = Dataset.load(sample_jsonl)
        train, val = ds.split(ratio=0.33, seed=42)
        assert len(train) + len(val) == 3

    def test_save(self, sample_jsonl, tmp_path):
        from complexity.dataset import Dataset
        ds = Dataset.load(sample_jsonl)
        out_path = tmp_path / "output.jsonl"
        ds.save(out_path)
        assert out_path.exists()

        # Reload and verify
        ds2 = Dataset.load(out_path)
        assert len(ds2) == len(ds)


class TestStreamingDataset:
    """Test StreamingDataset."""

    @pytest.fixture
    def sample_jsonl(self, tmp_path):
        data = [{"text": f"Line {i}"} for i in range(100)]
        path = tmp_path / "stream.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return path

    def test_streaming(self, sample_jsonl):
        from complexity.dataset import Dataset
        ds = Dataset.streaming(sample_jsonl)
        count = sum(1 for _ in ds)
        assert count == 100


class TestDataPipeline:
    """Test DataPipeline."""

    def test_pipeline_init(self, tmp_path):
        from complexity.dataset import DataPipeline

        # Create dummy tokenizer
        class DummyTokenizer:
            def encode(self, text):
                return list(range(len(text.split())))

        pipeline = DataPipeline(
            data_path=tmp_path,
            output_path=tmp_path / "output",
            tokenizer=DummyTokenizer(),
            seq_length=512,
        )
        assert pipeline.seq_length == 512
