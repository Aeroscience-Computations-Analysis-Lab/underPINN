"""Data-parallel PINN training across multiple devices.

What this does
--------------
Collocation / IC / BC points are split along their leading axis across devices;
network parameters and optimizer state are *replicated*. Each device evaluates
the PDE residual on its own shard, and XLA's SPMD partitioner inserts the
cross-device all-reduce when the per-term mean is taken. The result is
mathematically the same gradient a single device would compute on the full
batch -- this is data parallelism, not a different algorithm.

Why gradients stay exact
------------------------
``jax.jit`` over ``NamedSharding``-annotated arrays is a *global-view*
computation: the program means exactly what it would mean unsharded, and the
sharding only decides where the work runs. Every underPINN loss term is a mean
over its own point set::

    pde_loss = mean_i r(x_i)^2        over N_r collocation points
    ic_loss  = mean_j (u(x_j) - u_j)^2 over N_i initial-condition points
    ...

and the SPMD partitioner lowers ``jnp.mean`` over a ``P('data')`` axis to a
cross-device all-reduce-sum divided by the *global* count. So the sharded
gradient equals the single-device full-batch gradient, up to float32
reduction-order noise (which is why the tests compare with a tolerance rather
than bit-for-bit).

This is different from the older ``pmap`` style, where each device takes a
*local* mean and the results are ``pmean``-ed: there, unequal shard sizes would
genuinely mis-weight the result. That failure mode does not arise here.

Any number of devices
---------------------
A batch is split into equal per-device slices, so a batch of N points over D
devices needs N divisible by D. When it is not, :meth:`DataParallel.shard`
pads the batch to the next multiple of D by repeating its first ``(-N) % D``
rows (at most D-1 points), so every device count works with every batch size.

* **Divisible batch** -- no padding; the gradient equals single-device
  training exactly (up to float32 reduction order).
* **Non-divisible batch** -- the padded rows are counted twice in that step's
  mean. For randomly drawn mini-batches the duplicated rows are themselves
  random points, so the gradient remains an unbiased estimate; for a fixed
  full-batch set it is a fixed reweighting of at most D-1 points. Padding is
  reported at start-up by :meth:`DataParallel.prepare`, with the nearest sizes
  that would avoid it. :meth:`DataParallel.require_divisible` restores the
  strict check for callers that need exact equivalence.

Paired arrays of the same length (points and their targets / RBA state) are
padded with the same rows, so they stay aligned.

Why padding rather than an uneven split: JAX can split an uneven array only
*inside* a single compiled program. An array that crosses a ``jit`` boundary
unevenly sharded comes back fully *replicated* -- correct, but with nothing
split and no error -- so splitting exactly would mean restructuring every
training step.

Single-device reality check
---------------------------
On one GPU this buys nothing: there is no second device to split work across,
and the sharding annotations collapse to no-ops. Data parallelism pays off when
a *global batch* is spread over several devices. It does not reduce per-device
memory for the model itself -- parameters are replicated, so a model that does
not fit on one device still will not fit. Sharding the parameters themselves
(tensor / FSDP-style parallelism) is a different mechanism and is not what this
module does.

Testing without a multi-GPU box
-------------------------------
JAX can fake a multi-device host, which is enough to verify sharding
correctness::

    XLA_FLAGS=--xla_force_host_platform_device_count=8 python -m pytest \\
        tests/test_parallel.py

Those virtual devices prove the gradients and shapes are right; they say
nothing about speedup, which needs real accelerators.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

__all__ = [
    "DataParallel",
    "data_parallel_from_cfg",
    "data_parallel_kwargs",
    "make_data_parallel",
    "pad_to_multiple",
]


@dataclass(frozen=True)
class DataParallel:
    """A device mesh plus the shardings a data-parallel step needs.

    Parameters
    ----------
    mesh      : the 1-D :class:`jax.sharding.Mesh` to split batches over.
    axis_name : name of the mesh axis batches are sharded along.
    active    : ``False`` builds an *inert* instance whose ``shard``,
                ``shard_all``, ``replicate`` and ``prepare`` return their input
                unchanged. This lets a training loop call them unconditionally
                -- no ``if data_parallel:`` branch at every call site -- while
                still behaving exactly like the original code when the flag is
                off, even on a multi-device machine.

    Notes
    -----
    Instances are frozen and cheap to pass into a solver. Build one with
    :func:`make_data_parallel` or :func:`data_parallel_from_cfg` rather than
    calling this directly.
    """

    mesh: Mesh
    axis_name: str = "data"
    active: bool = True

    # ── introspection ────────────────────────────────────────────────────

    @property
    def n_devices(self) -> int:
        """Number of devices batches are split across."""
        return int(self.mesh.devices.size)

    @property
    def devices(self) -> tuple:
        return tuple(self.mesh.devices.flatten())

    @property
    def enabled(self) -> bool:
        """Whether batches are actually split.

        ``False`` when the instance is inert (``active=False``) or there is
        only one device -- in both cases every placement method is a no-op.
        """
        return self.active and self.n_devices > 1

    # ── shardings ────────────────────────────────────────────────────────

    def batch_sharding(self, axis: int = 0) -> NamedSharding:
        """Split along *axis* (the point axis), replicate every other axis.

        ``axis=0`` for a plain ``(batch, ...)`` array. Use ``axis=1`` for the
        ``(n_scan_steps, batch, ...)`` stacks the ``lax.scan`` path builds --
        splitting axis 0 there would distribute the *scan steps* across
        devices instead of the points within each step.
        """
        if axis < 0:
            raise ValueError(f"axis must be >= 0, got {axis}")
        return NamedSharding(self.mesh, P(*([None] * axis), self.axis_name))

    @property
    def replicated_sharding(self) -> NamedSharding:
        """Full copy on every device -- used for params and optimizer state."""
        return NamedSharding(self.mesh, P())

    # ── placement ────────────────────────────────────────────────────────

    def shard(self, array, axis: int = 0) -> Any:
        """Place *array* with *axis* split across devices (identity if inert).

        If ``array.shape[axis]`` does not divide by the device count, the array
        is first padded to the next multiple by repeating its first rows (see
        the module docstring), so any device count works.
        """
        if not self.enabled:
            return array
        if array.shape[axis] % self.n_devices:
            array, _ = pad_to_multiple(array, self.n_devices, axis=axis)
        return jax.device_put(array, self.batch_sharding(axis))

    def shard_all(self, arrays: Iterable, axis: int = 0) -> tuple:
        """:meth:`shard` applied to each array in *arrays*."""
        return tuple(self.shard(a, axis) for a in arrays)

    def replicate(self, tree) -> Any:
        """Place every leaf of *tree* on all devices (identity if inert)."""
        if not self.enabled:
            return tree
        return jax.device_put(tree, self.replicated_sharding)

    def prepare(self, params, state, sizes: Mapping[str, int]):
        """One-call setup for a training loop: report padding, then replicate.

        Call this *after* any restart-restore, so that parameters restored from
        a snapshot (plain single-device arrays) are re-replicated rather than
        left on one device. Returns ``(params, state)`` -- unchanged when inert.

        Parameters
        ----------
        sizes : mapping of name -> the *effective* per-step point count, i.e.
                what the loop actually draws, not what the config requests.
                Several loops use ``min(batch_bc, N_inlet)``, so the sharded
                batch can be smaller than the configured one. Used only to
                report which batches will be padded -- :meth:`shard` pads
                whatever it is given, so an incomplete mapping is harmless.
        """
        if not self.enabled:
            return params, state
        d = self.n_devices
        print(f"[data_parallel] each batch split across {d} "
              f"{self.devices[0].platform} devices; params replicated")
        padded = {k: v for k, v in sizes.items() if v % d}
        if padded:
            print(f"[data_parallel] batches not divisible by {d} are padded by "
                  f"repeating up to {d - 1} of their own points:")
            for k, v in padded.items():
                lo, hi = (v // d) * d, (v // d + 1) * d
                alt = f"{lo} or {hi}" if lo else f"{hi}"
                print(f"    {k} = {v} -> {hi}  (+{hi - v}; use {alt} for an "
                      f"exact split)")
        return self.replicate(params), self.replicate(state)

    # ── validation ───────────────────────────────────────────────────────

    def require_divisible(self, sizes: Mapping[str, int]) -> None:
        """Raise unless every count in *sizes* splits evenly across devices.

        Training no longer needs this -- :meth:`shard` pads non-divisible
        batches. Call it when a run must be *exactly* equivalent to
        single-device training, i.e. when no batch may be padded. It reports in
        the caller's terms: which size is wrong and what to use instead.
        """
        if not self.enabled:
            return
        d = self.n_devices
        bad = {k: v for k, v in sizes.items() if v % d != 0}
        if not bad:
            return
        lines = [
            f"Data-parallel training over {d} devices requires every point "
            f"count to be divisible by {d}: each batch is split along its "
            f"leading axis, and JAX cannot shard a dimension that does not "
            f"divide the device count.",
            "",
            "Offending sizes:",
        ]
        for k, v in sorted(bad.items()):
            lines.append(
                f"  {k} = {v}  ->  use {(v // d) * d} or {(v // d + 1) * d}")
        raise ValueError("\n".join(lines))


def _check_n_devices(n_devices) -> None:
    """Reject anything but ``None`` or a positive integer.

    ``bool`` is refused explicitly: ``True`` is an ``int`` in Python, so
    ``data_parallel_devices: true`` would otherwise be read as "1 device".
    """
    if n_devices is None:
        return
    if (isinstance(n_devices, bool)
            or not isinstance(n_devices, (int, np.integer)) or n_devices < 1):
        raise ValueError(
            f"data_parallel_devices must be a positive integer, or omitted to "
            f"use every visible device; got {n_devices!r}.")


def make_data_parallel(devices: Sequence | None = None,
                       axis_name: str = "data",
                       active: bool = True,
                       n_devices: int | None = None) -> DataParallel:
    """Build a :class:`DataParallel` over *devices* (default: all visible).

    Works unchanged on one device, where it is an explicit no-op -- so solver
    code does not need a separate single-device branch. With ``active=False``
    the result is inert even on a multi-device machine (see
    :class:`DataParallel`); its mesh is a single device, so merely constructing
    it never involves the others.

    Parameters
    ----------
    n_devices : use only the first *n_devices* of *devices* (all of them when
                ``None``). Any count works; batches that do not divide by it
                are padded (see :meth:`DataParallel.shard`). Asking for more
                than are visible is an error rather than a silent cap. To
                choose *which* GPUs, restrict visibility with
                ``CUDA_VISIBLE_DEVICES``.

    Examples
    --------
    >>> dp = make_data_parallel()                      # doctest: +SKIP
    >>> dp = make_data_parallel(n_devices=4)           # doctest: +SKIP
    >>> dp.require_divisible({"batch_r": 4096})        # doctest: +SKIP
    """
    _check_n_devices(n_devices)
    devs = list(devices) if devices is not None else jax.devices()
    if not devs:
        raise RuntimeError("No JAX devices visible; cannot build a mesh.")
    if not active:
        if n_devices is not None:
            import warnings
            warnings.warn(
                f"data_parallel_devices={n_devices} is set but data parallelism "
                f"is off, so training runs on a single device. Set "
                f"data_parallel: true to use {n_devices} devices.",
                UserWarning, stacklevel=2)
        devs = devs[:1]
    elif n_devices is not None:
        if n_devices > len(devs):
            raise ValueError(
                f"data_parallel_devices={n_devices}, but only {len(devs)} "
                f"device(s) are visible: {[str(d) for d in devs]}. Lower the "
                f"count, or check CUDA_VISIBLE_DEVICES.")
        devs = devs[:n_devices]
    mesh = Mesh(np.array(devs), axis_names=(axis_name,))
    return DataParallel(mesh=mesh, axis_name=axis_name, active=active)


def data_parallel_kwargs(train_cfg) -> dict:
    """Read the data-parallel settings from a YAML ``training:`` block.

    Returns ``{"data_parallel": bool, "data_parallel_devices": int | None}``,
    ready to splat into :class:`~underPINN.core.config.TrainingConfig` or a
    solver's ``train()``, so every example parses these keys the same way::

        training:
          data_parallel: true
          data_parallel_devices: 4     # optional; default = all visible
    """
    from underPINN.config.loader import cfg_get
    n_devices = cfg_get(train_cfg, "data_parallel_devices", default=None)
    _check_n_devices(n_devices)
    return {
        "data_parallel": bool(cfg_get(train_cfg, "data_parallel", default=False)),
        "data_parallel_devices": n_devices,
    }


def data_parallel_from_cfg(train_cfg, axis_name: str = "data") -> DataParallel:
    """Build a :class:`DataParallel` from a YAML ``training:`` block.

    Reads ``training.data_parallel`` (default ``False``) and the optional
    ``training.data_parallel_devices``, so an example turns it on with config
    alone and no code change::

        training:
          data_parallel: true
          data_parallel_devices: 4     # optional; default = all visible

    Returns an inert instance when ``data_parallel`` is absent or false, which
    makes the result safe to use unconditionally inside a training loop.
    """
    kw = data_parallel_kwargs(train_cfg)
    return make_data_parallel(axis_name=axis_name, active=kw["data_parallel"],
                              n_devices=kw["data_parallel_devices"])


def pad_to_multiple(array, multiple: int, axis: int = 0):
    """Pad *array* along *axis* up to a multiple of *multiple* by repeating rows.

    Returns ``(padded, n_original)``. This is an escape hatch for counts that
    cannot be made divisible; note that it is **not** loss-neutral -- the
    repeated rows are real points as far as the mean is concerned, so they are
    double-counted. Prefer choosing a divisible batch size. Use this only when
    the duplicated weight is acceptable (e.g. a large batch where a handful of
    repeated points is below the sampling noise floor).
    """
    if multiple < 1:
        raise ValueError(f"multiple must be >= 1, got {multiple}")
    n = array.shape[axis]
    rem = n % multiple
    if rem == 0 or n == 0:
        return array, n
    pad = multiple - rem
    take = jnp.arange(n, n + pad) % n           # wrap around and repeat
    extra = jnp.take(array, take, axis=axis)
    return jnp.concatenate([array, extra], axis=axis), n
