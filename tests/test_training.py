"""Tests for complexity.training module."""

import pytest
import torch


class TestTrainingConfig:
    """Test training configuration."""

    def test_default_config(self):
        """Test default training config."""
        from complexity.training import TrainingConfig

        config = TrainingConfig()

        assert config.max_steps > 0
        assert config.learning_rate > 0

    def test_custom_config(self):
        """Test custom training config."""
        from complexity.training import TrainingConfig

        config = TrainingConfig(
            max_steps=10000,
            learning_rate=1e-4,
            weight_decay=0.1,
            warmup_steps=1000,
        )

        assert config.max_steps == 10000
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.1


class TestTrainer:
    """Test trainer."""

    def test_create_trainer(self):
        """Test creating trainer."""
        from complexity.training import Trainer, TrainingConfig
        from complexity.models import ComplexityModel
        from complexity.config import ModelConfig

        model_config = ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=1000,
        )
        model = ComplexityModel(model_config)

        training_config = TrainingConfig(
            max_steps=100,
            learning_rate=1e-4,
        )

        # Create simple dataloader
        def dummy_dataloader():
            for _ in range(10):
                yield {
                    "input_ids": torch.randint(0, 1000, (4, 32)),
                    "labels": torch.randint(0, 1000, (4, 32)),
                }

        trainer = Trainer(
            model=model,
            config=training_config,
            train_dataloader=dummy_dataloader(),
        )

        assert trainer is not None

    @pytest.mark.skip(reason="Full training test - expensive")
    def test_train_step(self):
        """Test single training step."""
        from complexity.training import Trainer, TrainingConfig
        from complexity.models import ComplexityModel
        from complexity.config import ModelConfig

        model_config = ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=1000,
        )
        model = ComplexityModel(model_config)

        training_config = TrainingConfig(
            max_steps=1,
            learning_rate=1e-4,
        )

        batch = {
            "input_ids": torch.randint(0, 1000, (4, 32)),
            "labels": torch.randint(0, 1000, (4, 32)),
        }

        trainer = Trainer(
            model=model,
            config=training_config,
            train_dataloader=[batch],
        )

        # One step
        metrics = trainer.train()
        assert "loss" in metrics or metrics is not None


class TestMetricsTracker:
    """Test metrics tracking."""

    def test_create_tracker(self):
        """Test creating metrics tracker."""
        from complexity.training import MetricsTracker

        tracker = MetricsTracker()
        assert tracker is not None

    def test_log_metrics(self):
        """Test logging metrics."""
        from complexity.training import MetricsTracker

        tracker = MetricsTracker()
        tracker.log({"loss": 1.5, "accuracy": 0.8}, step=1)

        # Check metrics are stored
        assert len(tracker.history) > 0 or hasattr(tracker, 'metrics')


class TestLRScheduler:
    """Test learning rate scheduler."""

    def test_get_scheduler(self):
        """Test getting LR scheduler."""
        from complexity.training import get_lr_scheduler
        from complexity.models import ComplexityModel
        from complexity.config import ModelConfig

        model_config = ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=1000,
        )
        model = ComplexityModel(model_config)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = get_lr_scheduler(
            optimizer=optimizer,
            scheduler_type="cosine",
            num_warmup_steps=100,
            num_training_steps=1000,
        )

        assert scheduler is not None


class TestCallbacks:
    """Test training callbacks."""

    def test_early_stopping(self):
        """Test early stopping callback."""
        from complexity.training import EarlyStoppingCallback

        callback = EarlyStoppingCallback(
            patience=5,
            min_delta=0.01,
        )

        assert callback.patience == 5

    @pytest.mark.skip(reason="Requires wandb")
    def test_wandb_callback(self):
        """Test WandB callback."""
        from complexity.training import WandBCallback

        callback = WandBCallback(
            project="test-project",
            name="test-run",
        )

        assert callback is not None

    @pytest.mark.skip(reason="Requires tensorboard")
    def test_tensorboard_callback(self):
        """Test TensorBoard callback."""
        from complexity.training import TensorBoardCallback

        callback = TensorBoardCallback(
            log_dir="./logs",
        )

        assert callback is not None


class TestAPITrainer:
    """Test API trainer wrapper."""

    def test_trainer_config(self):
        """Test API trainer config."""
        from complexity.api.trainer import TrainerConfig

        config = TrainerConfig(
            max_steps=5000,
            batch_size=16,
            lr=1e-4,
        )

        assert config.max_steps == 5000
        assert config.batch_size == 16
        assert config.lr == 1e-4

    def test_trainer_config_conversion(self):
        """Test converting to internal config."""
        from complexity.api.trainer import TrainerConfig

        config = TrainerConfig(
            max_steps=5000,
            eval_steps=250,
            save_steps=500,
            lr=1e-4,
        )

        internal = config.to_training_config()

        assert internal.max_steps == 5000
        assert internal.eval_every_n_steps == 250
        assert internal.save_every_n_steps == 500
