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

    @staticmethod
    def _make_fake_moe_model(num_experts=4):
        from types import SimpleNamespace

        class FakeMoEModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(num_experts=num_experts, hidden_size=4)
                self.embed = torch.nn.Embedding(10, 4)
                self.gate_proj_w = torch.nn.Parameter(torch.randn(num_experts, 4, 4))
                self.dense = torch.nn.Linear(4, 4)
                self.norm = torch.nn.LayerNorm(4)

            def forward(self, x):
                return self.dense(self.norm(self.embed(x)))

        return FakeMoEModel()

    def test_plain_adamw_leaves_experts_in_the_base_group_by_default(self, tmp_path):
        """expert_lr_pack defaults to False: an existing run (like the live
        200M pretrain, which checkpointed under the old 2-group layout) must
        resume with the exact same param-group structure, or optimizer
        state_dict resume breaks. Experts stay in the base group at the
        plain learning_rate/weight_decay — no ×expert_lr_scale — unless the
        flag is explicitly opted into."""
        from complexity.training import Trainer, TrainingConfig

        model = self._make_fake_moe_model()
        training_config = TrainingConfig(
            max_steps=1,
            learning_rate=1e-3,
            weight_decay=0.1,
            expert_lr_scale=2.0,
            expert_weight_decay=0.005,
            optimizer_type="adamw",
            precision="fp32",
            use_fsdp=False,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "logs"),
        )

        trainer = Trainer(
            model=model,
            config=training_config,
            train_dataloader=[{"x": torch.zeros(1, dtype=torch.long)}],
        )

        assert len(trainer.optimizer.param_groups) == 2
        base_group = next(
            g for g in trainer.optimizer.param_groups
            if any(p is model.gate_proj_w for p in g["params"])
        )
        assert base_group["weight_decay"] == pytest.approx(0.1)
        assert base_group["lr"] == pytest.approx(1e-3)  # base learning_rate, not ×expert_lr_scale

    def test_plain_adamw_gives_expert_params_their_own_lr_pack_when_opted_in(self, tmp_path):
        """With expert_lr_pack=True, plain "adamw" gets an expert LR pack
        (like muon_tr/adam_tr): expert params land in their own group at
        learning_rate × expert_lr_scale with expert_weight_decay, instead of
        sharing the dense hidden group's settings."""
        from complexity.training import Trainer, TrainingConfig

        model = self._make_fake_moe_model()
        training_config = TrainingConfig(
            max_steps=1,
            learning_rate=1e-3,
            weight_decay=0.1,
            expert_lr_scale=2.0,
            expert_weight_decay=0.005,
            expert_lr_pack=True,
            optimizer_type="adamw",
            precision="fp32",
            use_fsdp=False,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "logs"),
        )

        trainer = Trainer(
            model=model,
            config=training_config,
            train_dataloader=[{"x": torch.zeros(1, dtype=torch.long)}],
        )

        expert_groups = [g for g in trainer.optimizer.param_groups if g.get("lr") == pytest.approx(2e-3)]
        assert len(expert_groups) == 1
        assert expert_groups[0]["weight_decay"] == pytest.approx(0.005)
        assert expert_groups[0]["params"][0] is model.gate_proj_w

        other_params = {
            id(p)
            for g in trainer.optimizer.param_groups
            if g is not expert_groups[0]
            for p in g["params"]
        }
        assert id(model.gate_proj_w) not in other_params

    def test_interrupted_training_saves_only_interrupted_not_also_final(self, tmp_path):
        """Regression guard: `self._save_checkpoint(tag="final")` used to sit
        unindented after the try/except KeyboardInterrupt block, so it ran
        on every exit path — a SIGTERM-triggered interrupt saved BOTH
        "interrupted_N" and "final_N" at the same step, doubling checkpoint
        disk usage and mislabeling a killed run as having finished. Live
        instance hit this: a single supervisor restart wrote 2x 26GB
        checkpoints for 56 steps of progress."""
        from complexity.training import Trainer, TrainingConfig

        model = torch.nn.Linear(1, 1)
        training_config = TrainingConfig(
            max_steps=100,
            learning_rate=1e-4,
            precision="fp32",
            use_fsdp=False,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "logs"),
        )

        def raising_dataloader():
            raise KeyboardInterrupt("simulated SIGTERM")
            yield  # pragma: no cover - makes this a generator

        def compute_loss(m, b):
            return m(b["x"]).sum()

        trainer = Trainer(
            model=model,
            config=training_config,
            train_dataloader=raising_dataloader(),
            compute_loss=compute_loss,
        )

        saved_tags = []
        trainer._save_checkpoint = lambda tag="step": saved_tags.append(tag)

        trainer.train()

        assert saved_tags == ["interrupted"]

    def test_canary_param_diagnostic_prints_a_single_line(self, capsys):
        """Regression guard: the post-step-1 canary diagnostic used to print
        one line per param plus a header and a summary line (6+ lines for a
        handful of params). Consolidated to one line so it doesn't stand out
        as a multi-line block against the rest of the log, which is one
        line per event everywhere else."""
        from complexity.training import Trainer

        model = torch.nn.Linear(4, 4)
        trainer = Trainer.__new__(Trainer)
        trainer.model = model
        trainer.is_main = True
        trainer._init_snapshot = {}
        trainer._update_check_done = False

        trainer._snapshot_canary_params()
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
                p.grad = torch.ones_like(p)

        trainer._check_params_updated()

        out = capsys.readouterr().out.strip("\n")
        lines = out.splitlines()

        assert len(lines) == 1
        assert lines[0].startswith("[param-update check] post-step-1:")
        assert "OK(delta=" in lines[0]

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

    @staticmethod
    def _wsd_trainer_stub(max_steps, warmup_steps, wsd_decay_ratio, global_step):
        """A bare Trainer with a real WSD scheduler built for max_steps, as if
        just restored by scheduler.load_state_dict() during resume -- enough
        to exercise _resync_wsd_schedule_bounds without a full training loop."""
        from complexity.training import Trainer, TrainingConfig
        from complexity.training.scheduler import get_lr_scheduler

        config = TrainingConfig(
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            wsd_decay_ratio=wsd_decay_ratio,
            lr_scheduler="wsd",
            learning_rate=1e-3,
        )
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        scheduler = get_lr_scheduler(optimizer, config=config, num_training_steps=max_steps)

        trainer = Trainer.__new__(Trainer)
        trainer.config = config
        trainer.scheduler = scheduler
        trainer.global_step = global_step
        trainer.is_main = True
        return trainer

    def test_resync_wsd_schedule_bounds_extends_decay_start_for_a_bigger_max_steps(self):
        """The checkpointed schedule (built for the old, shorter max_steps)
        must move its decay-start milestone and cosine T_max out to match a
        larger max_steps set on resume -- otherwise the stale, periodic
        CosineAnnealingLR would climb back toward peak LR once stepped past
        its original (now wrong) T_max."""
        old_max_steps = 1000
        trainer = self._wsd_trainer_stub(
            max_steps=old_max_steps, warmup_steps=100, wsd_decay_ratio=0.2, global_step=200,
        )
        old_milestones = list(trainer.scheduler._milestones)
        old_t_max = trainer.scheduler._schedulers[2].T_max

        trainer.config.max_steps = 5000  # extend the run

        trainer._resync_wsd_schedule_bounds()

        new_milestones = trainer.scheduler._milestones
        new_t_max = trainer.scheduler._schedulers[2].T_max
        assert new_milestones != old_milestones
        assert new_t_max != old_t_max
        # warmup boundary untouched; only the stable/decay split moves out.
        assert new_milestones[0] == old_milestones[0] == 100
        remaining = 5000 - 100
        expected_stable = int(remaining * 0.8)
        assert new_milestones[1] == 100 + expected_stable
        assert new_t_max == remaining - expected_stable

    def test_resync_wsd_schedule_bounds_is_a_noop_when_max_steps_is_unchanged(self):
        trainer = self._wsd_trainer_stub(
            max_steps=1000, warmup_steps=100, wsd_decay_ratio=0.2, global_step=200,
        )
        before = (list(trainer.scheduler._milestones), trainer.scheduler._schedulers[2].T_max)

        trainer._resync_wsd_schedule_bounds()

        after = (list(trainer.scheduler._milestones), trainer.scheduler._schedulers[2].T_max)
        assert before == after

    def test_resync_wsd_schedule_bounds_refuses_once_decay_phase_already_started(self):
        """Once global_step is at/past the (old, loaded) decay-start milestone,
        the cosine sub-scheduler has already been stepped into -- overwriting
        its T_max at that point would produce a discontinuous LR curve instead
        of a clean extension, so this must refuse rather than silently do the
        wrong thing."""
        trainer = self._wsd_trainer_stub(
            max_steps=1000, warmup_steps=100, wsd_decay_ratio=0.2, global_step=900,
        )
        trainer.config.max_steps = 5000

        with pytest.raises(RuntimeError, match="decay-start milestone"):
            trainer._resync_wsd_schedule_bounds()

    def test_resync_wsd_schedule_bounds_skips_non_wsd_schedulers(self):
        from complexity.training import Trainer, TrainingConfig
        from complexity.training.scheduler import get_lr_scheduler

        config = TrainingConfig(max_steps=1000, warmup_steps=100, lr_scheduler="cosine", learning_rate=1e-3)
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        scheduler = get_lr_scheduler(optimizer, config=config, num_training_steps=1000)

        trainer = Trainer.__new__(Trainer)
        trainer.config = config
        trainer.scheduler = scheduler
        trainer.global_step = 200
        trainer.is_main = True

        trainer.config.max_steps = 5000
        trainer._resync_wsd_schedule_bounds()  # must not raise or touch a cosine scheduler

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

    def test_supervisor_progress_uses_default_stderr_tqdm(self, monkeypatch):
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
        callback = TqdmCallback(total_steps=100, desc="supervisor-test")
        callback.close()

        assert "file" not in captured  # defaults to stderr, unbuffered when piped
        assert captured["disable"] is False
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
