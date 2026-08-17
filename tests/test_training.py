"""Tests for complexity.training module."""

from contextlib import contextmanager

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

    def test_evaluate_runs_without_backward(self, tmp_path):
        """Evaluation should compute loss without touching gradients."""
        from complexity.training import Trainer, TrainingConfig

        model = torch.nn.Linear(1, 1)
        training_config = TrainingConfig(
            max_steps=1,
            learning_rate=1e-4,
            precision="fp32",
            use_fsdp=False,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "logs"),
        )

        batch = {"x": torch.ones(1, 1)}

        def compute_loss(m, b):
            return m(b["x"]).sum()

        trainer = Trainer(
            model=model,
            config=training_config,
            train_dataloader=[batch],
            eval_dataloader=[batch],
            compute_loss=compute_loss,
        )

        eval_loss = trainer.evaluate()

        assert isinstance(eval_loss, float)
        assert all(p.grad is None for p in model.parameters())

    def test_gradient_accumulation_uses_no_sync_until_optimizer_boundary(self):
        from complexity.training import Trainer

        class Model:
            def __init__(self):
                self.no_sync_calls = 0

            @contextmanager
            def no_sync(self):
                self.no_sync_calls += 1
                yield

        trainer = Trainer.__new__(Trainer)
        trainer.model = Model()

        with trainer._gradient_sync_context(should_sync=False):
            pass
        with trainer._gradient_sync_context(should_sync=True):
            pass

        assert trainer.model.no_sync_calls == 1

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

    def test_supervisor_progress_uses_line_mode_outside_tty(self, monkeypatch):
        from complexity.training import TqdmCallback
        import tqdm

        captured = {}

        class FakeProgress:
            def close(self):
                pass

        def fake_tqdm(**kwargs):
            captured.update(kwargs)
            return FakeProgress()

        monkeypatch.setattr(tqdm, "tqdm", fake_tqdm)
        monkeypatch.setattr(
            "complexity.training.callbacks.is_main_process", lambda: True
        )
        monkeypatch.setattr("complexity.training.callbacks.sys.stderr.isatty", lambda: False)
        callback = TqdmCallback(total_steps=100, desc="supervisor-test")
        callback.close()

        assert "file" not in captured  # defaults to stderr, unbuffered when piped
        assert callback.line_mode is True
        assert captured["disable"] is True
        assert captured["dynamic_ncols"] is True
        # mininterval=0: one log line per real training step, no throttled
        # gaps in the step count (each __call__ already fires once per
        # optimizer step, not per micro-batch, so this can't flood the log).
        # miniters=1 is required alongside it: tqdm's dynamic_miniters
        # defaults to on whenever miniters is left at 0/None, and it
        # auto-inflates miniters based on observed call rate — silently
        # re-throttling to "every other step" (or worse as steps/s rises)
        # even with mininterval=0 explicitly set.
        assert captured["mininterval"] == 0
        assert captured["miniters"] == 1

    def test_on_resume_syncs_the_bar_to_the_resumed_step(self, monkeypatch):
        """Regression guard: after a process restart with --resume auto,
        tqdm's own counter starts at 0 regardless of where global_step
        actually resumed from, so the displayed N/total silently diverged
        from the real step (same loss/lr logged, but a bar that looked
        like it had restarted from scratch)."""
        from complexity.training import TqdmCallback

        monkeypatch.setattr(
            "complexity.training.callbacks.is_main_process", lambda: True
        )
        monkeypatch.setattr("complexity.training.callbacks.sys.stderr.isatty", lambda: True)
        callback = TqdmCallback(total_steps=247_946, desc="resume-test")
        assert callback.pbar.n == 0

        callback.on_resume(760)

        assert callback.pbar.n == 760
        callback.close()

    def test_call_refreshes_the_bar_exactly_once_per_step(self, monkeypatch):
        """Regression guard: set_postfix() refreshes by default and update()
        refreshes again. Outside a real tty (piped to a log file) each
        refresh becomes its own line instead of an in-place \\r redraw, so
        this used to print 2-3 lines per training step."""
        from unittest.mock import MagicMock

        from complexity.training import TqdmCallback

        monkeypatch.setattr(
            "complexity.training.callbacks.is_main_process", lambda: True
        )
        monkeypatch.setattr(
            "complexity.training.moe_telemetry.global_expert_shares",
            lambda model, num_experts=None: ([], None),
        )

        callback = TqdmCallback.__new__(TqdmCallback)
        callback.tokens_per_step = None
        callback.line_mode = False
        callback.pbar = MagicMock()
        callback.pbar.format_dict = {}

        trainer = MagicMock()
        trainer.scheduler.get_last_lr.return_value = [1e-4]
        trainer.optimizer = MagicMock(spec=[])

        callback(trainer, step=1, loss=2.0)

        callback.pbar.set_postfix.assert_called_once()
        assert callback.pbar.set_postfix.call_args.kwargs["refresh"] is False
        callback.pbar.update.assert_called_once_with(1)
        callback.pbar.refresh.assert_not_called()

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
