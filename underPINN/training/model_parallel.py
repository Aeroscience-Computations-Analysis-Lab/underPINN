"""Model (subdomain) parallelism for FBPINN networks.

What this does
--------------
Data parallelism (:mod:`underPINN.training.parallel`) copies the whole network
to every device and splits the *points*. Model parallelism here does the
opposite for FBPINN: it splits the *subdomain networks* across devices -- with S
subdomains on D devices, each device holds and evaluates only S/D of them, on
every point -- and XLA sums the per-device contributions. Per-device parameter
memory, activation memory, and compute all scale with S/D, which is what makes
FBPINNs with many subdomains tractable.

Why this is the only model parallelism underPINN offers
--------------------------------------------------------
Its other networks are far too small to split usefully (an MLP is ~0.2-0.6 MB;
the largest shipped FNO ~38 MB). Splitting one across devices adds a collective
per layer, multiplied by the nested ``jacfwd``/``hessian`` passes a PDE
residual needs -- communication would outweigh compute. FBPINN's subdomains are
independent networks joined only by a weighted sum, so they split cleanly.

It does not change the optimisation
-----------------------------------
:class:`~underPINN.nn.fbpinn.StackedFBPINN` is the same function as
:class:`~underPINN.nn.fbpinn.FBPINN` (weights stacked rather than kept in a
Python list) and initialises bit-for-bit identically. Sharding the stacked axis
is a layout choice inside a global-view ``jax.jit``, so the loss, every
gradient and the training trajectory match single-device training to float32
rounding.

Any number of devices
---------------------
When S does not divide by D the model is padded with up to D-1 *dummy*
subdomains, multiplied by an exact 0 mask: they add nothing to the output and
get exactly zero gradient, so they never change. (Parameters persist across
steps, and an unevenly split array that crosses a ``jit`` boundary comes back
fully replicated -- the same reason data parallelism pads batches.)

Usage
-----
Wrap the model *before* building the PDE (the PDE keeps a reference to it)::

    mp = model_parallel_from_cfg(cfg.training)   # inert unless model_parallel: true
    model = mp.wrap(FBPINN(...))                  # StackedFBPINN, padded for D
    pde = MyPDE(model, ...)
    ...
    params, opt_state = mp.prepare(params, opt_state)   # after any restore
    ...
    save_checkpoint(mp.export(model, params), ...)       # ordinary FBPINN layout

Solvers do the ``prepare`` step themselves when given
``TrainingConfig(model_parallel=True)``. Combining data and model parallelism
in one run is not supported yet; enable one or the other.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from underPINN.nn.fbpinn import FBPINN, StackedFBPINN
from underPINN.training.parallel import _check_n_devices, _select_devices

__all__ = [
    "ModelParallel",
    "make_model_parallel",
    "model_parallel_from_cfg",
    "model_parallel_kwargs",
]

_STACKED_KEY = "subnets"     # StackedFBPINN's vmapped submodule name


def _is_stacked(path) -> bool:
    """True for leaves under a StackedFBPINN's stacked ``subnets`` -- in the
    params and in any optimizer state that mirrors them (Adam's mu / nu)."""
    return any(getattr(k, "key", None) == _STACKED_KEY for k in path)


@dataclass(frozen=True)
class ModelParallel:
    """A device mesh for splitting FBPINN subdomains across devices.

    ``active=False`` gives an inert instance: :meth:`wrap` returns the model
    unchanged and :meth:`prepare` / :meth:`export` return their input, so a
    training loop can call them unconditionally.
    """

    mesh: Mesh
    axis_name: str = "model"
    active: bool = True

    @property
    def n_devices(self) -> int:
        return int(self.mesh.devices.size)

    @property
    def devices(self) -> tuple:
        return tuple(self.mesh.devices.flatten())

    @property
    def enabled(self) -> bool:
        return self.active and self.n_devices > 1

    @property
    def stacked_sharding(self) -> NamedSharding:
        """Split along the leading (subdomain) axis."""
        return NamedSharding(self.mesh, P(self.axis_name))

    @property
    def replicated_sharding(self) -> NamedSharding:
        return NamedSharding(self.mesh, P())

    # ── model ────────────────────────────────────────────────────────────

    def wrap(self, model):
        """The model to train: a :class:`StackedFBPINN` padded so its
        subdomain count divides the device count, or *model* itself when
        inert. Call before constructing the PDE, which keeps a reference."""
        if not self.enabled:
            return model
        if isinstance(model, (FBPINN, StackedFBPINN)):
            return StackedFBPINN.from_fbpinn(model, multiple=self.n_devices)
        raise TypeError(
            f"model parallelism splits FBPINN subdomain networks across "
            f"devices; got {type(model).__name__}. Other underPINN networks "
            f"are too small to split usefully -- use data parallelism "
            f"(training.data_parallel) for them.")

    def export(self, model, params) -> dict:
        """*params* in the ordinary FBPINN layout (dummy subdomains dropped),
        so a checkpoint from a model-parallel run is interchangeable with one
        from a normal run. Identity for a non-stacked model."""
        if isinstance(model, StackedFBPINN):
            return model.export_params(jax.device_get(params))
        return params

    # ── placement ────────────────────────────────────────────────────────

    def place(self, tree) -> Any:
        """Shard stacked-subdomain leaves along the model axis; replicate the
        rest (e.g. an optimizer's step count)."""
        def put(path, leaf):
            if _is_stacked(path) and getattr(leaf, "ndim", 0) >= 1:
                return jax.device_put(leaf, self.stacked_sharding)
            return jax.device_put(leaf, self.replicated_sharding)
        return jax.tree_util.tree_map_with_path(put, tree)

    def prepare(self, params, state, model=None):
        """Validate the layout, then shard params and optimizer state.

        Call after any restart-restore. Returns ``(params, state)``, unchanged
        when inert. Pass the (wrapped) *model* so the report can tell real
        subdomains from masked padding, and warn when some devices would hold
        only padding.
        """
        if not self.enabled:
            return params, state
        stacked = [leaf for path, leaf in
                   jax.tree_util.tree_flatten_with_path(params)[0]
                   if _is_stacked(path)]
        if not stacked:
            keys = list(params.get("params", {})) if isinstance(params, dict) else []
            if any(str(k).startswith("subnets_") for k in keys):
                raise ValueError(
                    "model_parallel: these are plain FBPINN params. Wrap the "
                    "model first -- `model = mp.wrap(FBPINN(...))` -- before "
                    "building the PDE and initialising, so params use the "
                    "stacked layout that can be split across devices.")
            raise ValueError(
                "model_parallel applies to FBPINN networks only, and found no "
                "stacked subdomain weights in these params. Use data "
                "parallelism (training.data_parallel) for other networks.")
        n_total = stacked[0].shape[0]
        if n_total % self.n_devices:
            raise ValueError(
                f"model_parallel: {n_total} stacked subdomains do not divide "
                f"across {self.n_devices} devices. Build the model with "
                f"mp.wrap(...) from this same ModelParallel, which pads it.")
        n_real = (model.n_subdomains if isinstance(model, StackedFBPINN)
                  else n_total)
        print(f"[model_parallel] {n_real} subdomain networks split across "
              f"{self.n_devices} {self.devices[0].platform} devices "
              f"({n_total // self.n_devices} per device"
              + (f", {n_total - n_real} masked padding)" if n_total > n_real
                 else ")"))
        if n_real < self.n_devices:
            import warnings
            warnings.warn(
                f"model_parallel: only {n_real} real subdomain(s) for "
                f"{self.n_devices} devices, so {self.n_devices - n_real} "
                f"device(s) hold only masked padding and do no useful work. "
                f"Use model_parallel_devices <= {n_real}, more subdomains, or "
                f"data parallelism instead.", UserWarning, stacklevel=2)
        return self.place(params), self.place(state)


def make_model_parallel(devices: Sequence | None = None,
                        axis_name: str = "model",
                        active: bool = True,
                        n_devices: int | None = None) -> ModelParallel:
    """Build a :class:`ModelParallel` over the first *n_devices* visible
    devices (all when ``None``). Inert with ``active=False``."""
    devs = _select_devices(devices, active, n_devices, mode="model_parallel",
                           what="model parallelism")
    mesh = Mesh(np.array(devs), axis_names=(axis_name,))
    return ModelParallel(mesh=mesh, axis_name=axis_name, active=active)


def model_parallel_kwargs(train_cfg) -> dict:
    """Read ``training.model_parallel`` / ``training.model_parallel_devices``,
    ready to splat into ``TrainingConfig`` or a solver's ``train()``."""
    from underPINN.config.loader import cfg_get
    n_devices = cfg_get(train_cfg, "model_parallel_devices", default=None)
    _check_n_devices(n_devices, key="model_parallel_devices")
    flag = bool(cfg_get(train_cfg, "model_parallel", default=False))
    if flag and bool(cfg_get(train_cfg, "data_parallel", default=False)):
        raise ValueError(
            "training.data_parallel and training.model_parallel are both on; "
            "combining them in one run is not supported yet -- enable one.")
    return {"model_parallel": flag, "model_parallel_devices": n_devices}


def model_parallel_from_cfg(train_cfg, axis_name: str = "model") -> ModelParallel:
    """Build a :class:`ModelParallel` from a YAML ``training:`` block::

        training:
          model_parallel: true
          model_parallel_devices: 4    # optional; default = all visible

    Inert when ``model_parallel`` is absent or false.
    """
    kw = model_parallel_kwargs(train_cfg)
    return make_model_parallel(axis_name=axis_name,
                               active=kw["model_parallel"],
                               n_devices=kw["model_parallel_devices"])
