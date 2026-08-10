"""Tests for TrainingConfig validation and default values.

Verifies that:
* All fields have the expected default values.
* `__post_init__` raises `ValueError` for every invalid field.
* Valid edge-case configs (e.g. resample_period=0, lr_schedule=None) pass.
"""
import pytest
from underPINN.core.config import TrainingConfig


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_epochs_default(self):
        assert TrainingConfig().epochs == 1000

    def test_lr_default(self):
        assert TrainingConfig().lr == 1e-3

    def test_batch_r_default(self):
        assert TrainingConfig().batch_r == 4096

    def test_resample_period_off_by_default(self):
        assert TrainingConfig().resample_period == 0

    def test_resample_k_default(self):
        assert TrainingConfig().resample_k == 1.0

    def test_n_scan_steps_default(self):
        assert TrainingConfig().n_scan_steps == 1

    def test_out_dir_default(self):
        assert TrainingConfig().out_dir == ""

    def test_callbacks_default_empty(self):
        assert TrainingConfig().callbacks == []

    def test_candidate_sampler_default_none(self):
        assert TrainingConfig().candidate_sampler is None

    def test_lr_schedule_default_none(self):
        assert TrainingConfig().lr_schedule is None


# ---------------------------------------------------------------------------
# Validation — each invalid field should raise ValueError
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_negative_epochs(self):
        with pytest.raises(ValueError, match="epochs"):
            TrainingConfig(epochs=-1)

    def test_zero_epochs(self):
        with pytest.raises(ValueError, match="epochs"):
            TrainingConfig(epochs=0)

    def test_float_epochs(self):
        with pytest.raises(ValueError, match="epochs"):
            TrainingConfig(epochs=1.5)

    def test_zero_lr(self):
        with pytest.raises(ValueError, match="lr"):
            TrainingConfig(lr=0.0)

    def test_negative_lr(self):
        with pytest.raises(ValueError, match="lr"):
            TrainingConfig(lr=-1e-3)

    def test_zero_batch_r(self):
        with pytest.raises(ValueError, match="batch_r"):
            TrainingConfig(batch_r=0)

    def test_zero_n_scan_steps(self):
        with pytest.raises(ValueError, match="n_scan_steps"):
            TrainingConfig(n_scan_steps=0)

    def test_negative_resample_period(self):
        with pytest.raises(ValueError, match="resample_period"):
            TrainingConfig(resample_period=-1)

    def test_zero_resample_k(self):
        with pytest.raises(ValueError, match="resample_k"):
            TrainingConfig(resample_k=0.0)

    def test_negative_resample_k(self):
        with pytest.raises(ValueError, match="resample_k"):
            TrainingConfig(resample_k=-1.0)

    def test_negative_seed(self):
        with pytest.raises(ValueError, match="seed"):
            TrainingConfig(seed=-1)

    def test_negative_save_restart_every(self):
        with pytest.raises(ValueError, match="save_restart_every"):
            TrainingConfig(save_restart_every=-1)


# ---------------------------------------------------------------------------
# Valid edge cases
# ---------------------------------------------------------------------------

class TestValidEdgeCases:
    def test_minimal_config(self):
        cfg = TrainingConfig(epochs=1, lr=1e-10, batch_r=1, batch_i=1, batch_b=1,
                             log_every=1, n_scan_steps=1, seed=0)
        assert cfg.epochs == 1

    def test_resample_disabled(self):
        cfg = TrainingConfig(resample_period=0)
        assert cfg.resample_period == 0

    def test_resample_enabled(self):
        cfg = TrainingConfig(resample_period=10, resample_k=2.0, resample_candidates=500)
        assert cfg.resample_period == 10
        assert cfg.resample_k == 2.0
        assert cfg.resample_candidates == 500

    def test_with_callbacks(self):
        from underPINN.callbacks.early_stopping import EarlyStopping
        from underPINN.callbacks.logging import ConsoleLogger
        cfg = TrainingConfig(callbacks=[ConsoleLogger(), EarlyStopping(patience=50)])
        assert len(cfg.callbacks) == 2

    def test_with_lr_schedule(self):
        import optax
        sched = optax.cosine_decay_schedule(1e-3, 1000)
        cfg = TrainingConfig(lr=1e-3, lr_schedule=sched)
        assert cfg.lr_schedule is sched

    def test_save_restart_zero_allowed(self):
        cfg = TrainingConfig(save_restart_every=0)
        assert cfg.save_restart_every == 0
