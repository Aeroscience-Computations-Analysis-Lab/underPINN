"""Mid-training restart system for underPINN.

Saves a compact snapshot every N epochs so an interrupted run can resume
exactly where it left off, without restarting from epoch 0.

How it works
------------
* A snapshot is written to ``<out_dir>/restart/`` every ``save_every`` epochs.
* On the *next* run the snapshot is detected automatically.
* The config is hashed (MD5).  If the config changed between runs the
  snapshot is silently ignored and training starts fresh.
* When training finishes normally, :meth:`done` marks the snapshot as
  complete — a subsequent re-run starts fresh rather than re-resuming.

Snapshot layout (``<out_dir>/restart/``)::

    params.msgpack    — Flax-serialised model parameters
    opt_state.msgpack — Flax-serialised optimizer state
    hists.npz         — loss history arrays (loss, pde, ic, bc, …)
    meta.json         — epoch, cfg_hash, done flag

Usage — custom training loop
----------------------------
::

    from underPINN.utils.restart import RestartManager

    rm = RestartManager(out_dir, save_every=500, cfg=cfg)

    start_ep, params, opt_state, hists = rm.maybe_restore(params, opt_state)
    loss_hist = hists.get("loss_hist", [])

    for ep in range(start_ep, total_epochs):
        params, opt_state, loss = step(params, opt_state, ...)
        loss_hist.append(float(loss))
        rm.maybe_save(ep, params, opt_state, {"loss_hist": loss_hist})

    rm.done()

Usage — FBPINNSolver via TrainingConfig
---------------------------------------
::

    from underPINN.core.config import TrainingConfig
    config = TrainingConfig(
        epochs=10_000,
        out_dir="outputs/burgers",   # ← enables automatic restarts
        save_restart_every=500,
    )
    solver.train(*data, config=config)
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _hash_cfg(cfg) -> str:
    """Return a stable MD5 hex-digest of *cfg*.

    Supports SimpleNamespace (YAML-loaded config used by the ``resume`` CLI),
    plain dicts, and Python dataclasses.  The ``resume`` CLI is the canonical
    caller; solvers should pass ``cfg=None`` to skip hash checking and rely
    solely on the ``done`` flag.
    """
    try:
        from underPINN.config.loader import _ns_to_dict
        d = _ns_to_dict(cfg)
    except Exception:
        import dataclasses
        import types
        if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
            d = dataclasses.asdict(cfg)
        elif isinstance(cfg, types.SimpleNamespace):
            d = vars(cfg)
        elif isinstance(cfg, dict):
            d = cfg
        else:
            d = {}
    text = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# RestartManager
# ---------------------------------------------------------------------------

class RestartManager:
    """Periodic mid-training checkpointer with config-based resume detection.

    Parameters
    ----------
    out_dir    : str | Path
        Experiment output directory.  Snapshots are written to
        ``<out_dir>/restart/``.
    save_every : int
        Snapshot interval in epochs (default 500).
    cfg        : optional config object
        Used to compute a config hash.  If the hash does not match the
        saved snapshot the snapshot is ignored (config changed → fresh start).
    """

    _SUBDIR = "restart"
    _PARAMS = "params.msgpack"
    _OPT    = "opt_state.msgpack"
    _HISTS  = "hists.npz"
    _META   = "meta.json"

    def __init__(self, out_dir, save_every: int = 500, cfg=None):
        self._dir       = pathlib.Path(out_dir) / self._SUBDIR
        self.save_every = max(1, int(save_every))
        self._cfg_hash  = _hash_cfg(cfg) if cfg is not None else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_checkpoint(self) -> bool:
        """Return True iff a usable snapshot exists for the current config."""
        meta = self._read_meta()
        if meta is None:
            return False
        if meta.get("done", False):
            return False                          # completed cleanly → start fresh
        if self._cfg_hash and meta.get("cfg_hash") != self._cfg_hash:
            return False                          # config changed → start fresh
        return (self._dir / self._PARAMS).exists() and meta.get("epoch", -1) >= 0

    def maybe_restore(
        self,
        params,
        opt_state,
    ) -> Tuple[int, Any, Any, Dict[str, List[float]]]:
        """Attempt to restore from snapshot.

        Returns
        -------
        (start_epoch, params, opt_state, hists)

        *hists* is a dict with keys like ``"loss_hist"``, ``"pde_hist"``, etc.
        All values are lists of floats.

        If no valid snapshot is found, returns
        ``(0, original_params, original_opt_state, {})``.
        """
        if not self.has_checkpoint():
            return 0, params, opt_state, {}

        try:
            from flax import serialization

            meta  = self._read_meta()
            epoch = int(meta["epoch"])

            # ── params ────────────────────────────────────────────────────
            with open(self._dir / self._PARAMS, "rb") as f:
                params = serialization.from_bytes(params, f.read())

            # ── opt_state ─────────────────────────────────────────────────
            opt_path = self._dir / self._OPT
            if opt_path.exists():
                with open(opt_path, "rb") as f:
                    opt_state = serialization.from_bytes(opt_state, f.read())

            # ── loss histories ─────────────────────────────────────────────
            hists: Dict[str, List[float]] = {}
            hists_path = self._dir / self._HISTS
            if hists_path.exists():
                npz = np.load(hists_path)
                for key in npz.files:
                    hists[key] = npz[key].tolist()

            last_loss = (f"{hists['loss_hist'][-1]:.3e}"
                         if hists.get("loss_hist") else "n/a")
            print(f"\n  [Restart] Resuming from epoch {epoch + 1}  "
                  f"(last saved loss {last_loss})\n")
            return epoch + 1, params, opt_state, hists

        except Exception as exc:
            print(f"\n  [Restart] Snapshot found but unreadable ({exc})."
                  f"  Starting from scratch.\n")
            return 0, params, opt_state, {}

    def maybe_save(
        self,
        epoch: int,
        params,
        opt_state,
        hists: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """Save snapshot if this epoch aligns with ``save_every``."""
        if (epoch + 1) % self.save_every == 0:
            self._write(epoch, params, opt_state, hists or {}, done=False)

    def done(self) -> None:
        """Mark training as complete.

        After this call ``has_checkpoint()`` returns False, so the next
        identical invocation starts fresh (not resumes).
        """
        meta = self._read_meta() or {}
        meta["done"] = True
        self._write_meta(meta)
        print("  [Restart] Training complete — snapshot marked as done.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(
        self,
        epoch: int,
        params,
        opt_state,
        hists: Dict[str, List[float]],
        done: bool,
    ) -> None:
        from flax import serialization

        self._dir.mkdir(parents=True, exist_ok=True)

        with open(self._dir / self._PARAMS, "wb") as f:
            f.write(serialization.to_bytes(params))
        with open(self._dir / self._OPT, "wb") as f:
            f.write(serialization.to_bytes(opt_state))

        if hists:
            np.savez(self._dir / self._HISTS,
                     **{k: np.array(v, dtype=np.float32) for k, v in hists.items()})

        meta = {
            "epoch":    epoch,
            "cfg_hash": self._cfg_hash,
            "done":     done,
        }
        self._write_meta(meta)

    def _read_meta(self) -> Optional[dict]:
        p = self._dir / self._META
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def _write_meta(self, meta: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / self._META).write_text(json.dumps(meta, indent=2))
