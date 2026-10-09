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

Divisibility
------------
Each point count must divide evenly across devices -- not for correctness, but
because ``jax.device_put`` refuses to shard a leading dimension that does not
divide the mesh axis. :meth:`DataParallel.require_divisible` checks this before
any tracing and names the offending config field with usable values, instead of
leaving the caller to decode JAX's generic shape error. Use
:func:`pad_to_multiple` only if a non-divisible count is truly unavoidable.

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
        """Place *array* with *axis* split across devices (identity if inert)."""
        if not self.enabled:
            return array
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
        """One-call setup for a training loop: validate, then replicate.

        Call this *after* any restart-restore, so that parameters restored from
        a snapshot (plain single-device arrays) are re-replicated rather than
        left on one device. Returns ``(params, state)`` -- unchanged when inert.

        Parameters
        ----------
        sizes : mapping of name -> the *effective* per-step point count, i.e.
                what the loop actually draws, not what the config requests.
                Several loops use ``min(batch_bc, N_inlet)``, so the sharded
                batch can be smaller than the configured one.
        """
        if not self.enabled:
            return params, state
        self.require_divisible(sizes)
        print(f"[data_parallel] each batch split across {self.n_devices} "
              f"{self.devices[0].platform} devices; params replicated")
        return self.replicate(params), self.replicate(state)

    # ── validation ───────────────────────────────────────────────────────

    def require_divisible(self, sizes: Mapping[str, int]) -> None:
        """Raise unless every count in *sizes* splits evenly across devices.

        JAX would reject an uneven split anyway, at ``device_put`` time, with a
        generic shape error. This check runs before any tracing and reports in
        the caller's terms: which config field is wrong and what to use
        instead.
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


def make_data_parallel(devices: Sequence | None = None,
                       axis_name: str = "data",
                       active: bool = True) -> DataParallel:
    """Build a :class:`DataParallel` over *devices* (default: all visible).

    Works unchanged on one device, where it is an explicit no-op -- so solver
    code does not need a separate single-device branch. With ``active=False``
    the result is inert even on a multi-device machine (see
    :class:`DataParallel`); its mesh is a single device, so merely constructing
    it never involves the others.

    Examples
    --------
    >>> dp = make_data_parallel()                      # doctest: +SKIP
    >>> dp.require_divisible({"batch_r": 4096})        # doctest: +SKIP
    """
    devs = list(devices) if devices is not None else jax.devices()
    if not devs:
        raise RuntimeError("No JAX devices visible; cannot build a mesh.")
    if not active:
        devs = devs[:1]
    mesh = Mesh(np.array(devs), axis_names=(axis_name,))
    return DataParallel(mesh=mesh, axis_name=axis_name, active=active)


def data_parallel_from_cfg(train_cfg, axis_name: str = "data") -> DataParallel:
    """Build a :class:`DataParallel` from a YAML ``training:`` block.

    Reads ``training.data_parallel`` (default ``False``), so an example turns
    it on with one line of config and no code change::

        training:
          data_parallel: true

    Returns an inert instance when the key is absent or false, which makes the
    result safe to use unconditionally inside a training loop.
    """
    from underPINN.config.loader import cfg_get
    flag = bool(cfg_get(train_cfg, "data_parallel", default=False))
    return make_data_parallel(axis_name=axis_name, active=flag)


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
    if rem == 0:
        return array, n
    pad = multiple - rem
    take = jnp.arange(n, n + pad) % n           # wrap around and repeat
    extra = jnp.take(array, take, axis=axis)
    return jnp.concatenate([array, extra], axis=axis), n
