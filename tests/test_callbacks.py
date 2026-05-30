"""Tests for callback implementations.

Covers EarlyStopping and ConsoleLogger behaviour: triggering, patience,
monitor key selection, reset, and output formatting.
"""
import logging
import pytest

from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.callbacks.logging import ConsoleLogger


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------
# EarlyStopping raises StopIteration when _wait >= patience.
# _wait resets to 0 on any call where loss < best - min_delta.
# _wait increments to 1, 2, ..., patience → fires on the patience-th bad epoch.

class TestEarlyStopping:
    def test_does_not_stop_while_improving(self):
        cb = EarlyStopping(patience=5, min_delta=0.0)
        for i in range(20):
            cb.on_epoch_end(i, {"loss": 1.0 / (i + 1)})
        # Must reach here without StopIteration

    def test_stops_after_patience_exhausted(self):
        # patience=3: fires when _wait reaches 3
        cb = EarlyStopping(patience=3, min_delta=0.0)
        cb.on_epoch_end(0, {"loss": 1.0})   # improves → _best=1.0 _wait=0
        cb.on_epoch_end(1, {"loss": 0.5})   # improves → _best=0.5 _wait=0
        cb.on_epoch_end(2, {"loss": 0.5})   # stale → _wait=1
        cb.on_epoch_end(3, {"loss": 0.5})   # stale → _wait=2
        with pytest.raises(StopIteration):
            cb.on_epoch_end(4, {"loss": 0.5})  # stale → _wait=3 ≥ patience → stop

    def test_min_delta_respected(self):
        # patience=2, min_delta=0.1 → improvement must be > 0.1 from best
        cb = EarlyStopping(patience=2, min_delta=0.1)
        cb.on_epoch_end(0, {"loss": 1.0})   # improves (1.0 < inf-0.1) → _best=1.0 _wait=0
        cb.on_epoch_end(1, {"loss": 0.95})  # 0.95 < 1.0-0.1=0.9? NO → _wait=1
        with pytest.raises(StopIteration):
            cb.on_epoch_end(2, {"loss": 0.95})  # still no → _wait=2 ≥ patience → stop

    def test_improvement_resets_counter(self):
        # patience=2; improvement at epoch 2 must reset _wait to 0
        cb = EarlyStopping(patience=2, min_delta=0.0)
        cb.on_epoch_end(0, {"loss": 1.0})   # _best=1.0 _wait=0
        cb.on_epoch_end(1, {"loss": 1.0})   # stale → _wait=1
        cb.on_epoch_end(2, {"loss": 0.5})   # improves → _best=0.5 _wait=0 (counter reset)
        cb.on_epoch_end(3, {"loss": 0.5})   # stale → _wait=1
        with pytest.raises(StopIteration):
            cb.on_epoch_end(4, {"loss": 0.5})  # stale → _wait=2 ≥ patience → stop

    def test_monitor_custom_key(self):
        # Watch "pde", not "loss"
        cb = EarlyStopping(patience=2, monitor="pde")
        cb.on_epoch_end(0, {"loss": 1.0, "pde": 1.0})   # pde improves → _best=1.0
        cb.on_epoch_end(1, {"loss": 0.01, "pde": 1.0})  # pde stale → _wait=1
        with pytest.raises(StopIteration):
            cb.on_epoch_end(2, {"loss": 0.001, "pde": 1.0})  # pde stale → _wait=2 → stop

    def test_missing_monitor_key_uses_inf(self):
        """When the monitor key is absent, loss is treated as +inf (never improves)."""
        cb = EarlyStopping(patience=2)
        # _best starts at inf; inf is not < inf-0 = inf, so every call is stale
        cb.on_epoch_end(0, {})   # stale → _wait=1
        with pytest.raises(StopIteration):
            cb.on_epoch_end(1, {})   # stale → _wait=2 ≥ patience → stop

    def test_on_train_end_no_error(self):
        cb = EarlyStopping()
        if hasattr(cb, "on_train_end"):
            cb.on_train_end({"loss": 0.1})


# ---------------------------------------------------------------------------
# ConsoleLogger — uses logging module (not print), capture via caplog fixture
# ---------------------------------------------------------------------------

class TestConsoleLogger:
    def test_fires_at_log_every(self, caplog):
        cb = ConsoleLogger(log_every=5)
        with caplog.at_level(logging.INFO, logger="underPINN.training"):
            for i in range(10):
                cb.on_epoch_end(i, {"loss": 0.1})
        # Epochs 0, 5 (and possibly 9 depending on log_every) should generate a log
        fired_epochs = [int(r.message.split()[1]) for r in caplog.records
                        if "Epoch" in r.message]
        assert 0 in fired_epochs or 5 in fired_epochs

    def test_no_log_between_intervals(self, caplog):
        cb = ConsoleLogger(log_every=100)
        with caplog.at_level(logging.INFO, logger="underPINN.training"):
            for i in range(50):
                cb.on_epoch_end(i, {"loss": 0.5})
        epoch_logs = [r for r in caplog.records if "Epoch" in r.message]
        # Only epoch 0 should fire (the first one)
        for r in epoch_logs:
            ep = int(r.message.split()[1])
            assert ep == 0

    def test_on_train_end_logs(self, caplog):
        cb = ConsoleLogger(log_every=1)
        with caplog.at_level(logging.INFO, logger="underPINN.training"):
            cb.on_train_end({"loss": 1.23e-4})
        assert any("Training complete" in r.message or "complete" in r.message.lower()
                   for r in caplog.records)

    def test_does_not_crash_on_empty_logs(self):
        cb = ConsoleLogger(log_every=1)
        cb.on_epoch_end(0, {})   # should not raise
