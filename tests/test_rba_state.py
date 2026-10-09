"""RBA running-sum bookkeeping in LDCSolver and RANSSolver.

Both solvers keep a per-collocation-point running sum (residual-based
adaptivity) that weights each point's residual by its own history. Every epoch
they shuffle the points, walk them in mini-batches, and feed each batch its
slice of the running sums -- so the sums must stay attached to the *points*
across the shuffle. A regression here is silent: training still runs and the
loss still falls, the weights are just someone else's.

The step function is replaced by a stub that records what it is handed and
returns per-point values that identify the point (each collocation point's x
coordinate is its own index). On every later epoch, each point must arrive
carrying exactly the value computed for *it* -- or the initial 0 if it has not
been visited yet (when n_col is not a multiple of the batch size, a different
remainder is skipped each epoch).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_OFFSETS = (0.0, 0.25, 0.5, 0.75, 0.875)   # distinguishes the RBA components


def _tagged_points(n_col: int) -> jnp.ndarray:
    """(n_col, 2) points whose x coordinate is the point's own index."""
    return jnp.stack([jnp.arange(n_col, dtype=jnp.float32),
                      jnp.full(n_col, 0.5, jnp.float32)], axis=1)


def _check_alignment(calls, n_comp: int, steps: int):
    """Every point must carry its own RBA value (or 0 before first visit)."""
    seen: set[int] = set()
    checked = 0
    for i, (col, r_batch) in enumerate(calls):
        ids = np.asarray(col)[:, 0].astype(int)
        for k in range(n_comp):
            got = np.asarray(r_batch[k])
            want = np.array([pid + _OFFSETS[k] if pid in seen else 0.0
                             for pid in ids])
            np.testing.assert_allclose(
                got, want, atol=1e-6,
                err_msg=f"RBA component {k} misaligned in step {i} "
                        f"(epoch {i // steps}): points {ids.tolist()} received "
                        f"{got.tolist()}")
            checked += 1
        if (i + 1) % steps == 0:            # epoch boundary: commit visits
            for c, _r in calls[i + 1 - steps:i + 1]:
                seen.update(np.asarray(c)[:, 0].astype(int).tolist())
    assert checked > 0
    return seen


# ---------------------------------------------------------------------------
# LDCSolver
# ---------------------------------------------------------------------------

def _ldc_solver(n_col: int):
    import optax
    from underPINN.nn.mlp import MLP
    from underPINN.pde.navier_stokes import NavierStokesPDE
    from underPINN.solver.ldc_solver import LDCInputWrapper, LDCSolver

    model = MLP(layers=[2, 8, 3])
    solver = LDCSolver(model, NavierStokesPDE(model, Re=100.0),
                       optimizer=optax.adam(1e-3))
    solver.init(jax.random.PRNGKey(0))
    inputs = LDCInputWrapper(col=_tagged_points(n_col),
                             inlet=jnp.zeros((6, 2)), noslip=jnp.zeros((6, 2)))
    calls = []

    def stub(params, state, rsum, col, inlet, noslip, is_init):
        calls.append((col, rsum))
        tag = jnp.asarray(col)[:, 0]
        new = tuple(tag + o for o in _OFFSETS[:3])
        z = jnp.float32(0.0)
        return params, state, new, z, (z, z, z)

    solver._step = stub
    return solver, inputs, calls


@pytest.mark.parametrize("n_col, batch", [(12, 4), (10, 4)],
                         ids=["divisible", "remainder"])
def test_ldc_rba_sums_stay_with_their_points(n_col, batch):
    solver, inputs, calls = _ldc_solver(n_col)
    solver.train(inputs, epochs=4, batch_size=batch, seed=0)
    steps = n_col // batch
    assert len(calls) == 4 * steps
    _check_alignment(calls, n_comp=3, steps=steps)


# ---------------------------------------------------------------------------
# RANSSolver
# ---------------------------------------------------------------------------

def _rans_solver(n_col: int):
    import optax
    from underPINN.nn.mlp import MLP
    from underPINN.pde.k_epsilon import KEpsilonPDE
    from underPINN.solver.rans_solver import RANSInputWrapper, RANSSolver

    model = MLP(layers=[2, 8, 5])
    solver = RANSSolver(model, KEpsilonPDE(model, Re=1000.0), optax.adam(1e-3))
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))
    z2 = jnp.zeros((6, 2))
    inputs = RANSInputWrapper(col=_tagged_points(n_col), inlet=z2, noslip=z2,
                              outlet=z2, data_x=z2, data_u=jnp.zeros((6, 5)))
    calls = []

    def stub(params, state, rsum, col, inlet, noslip, outlet, data_x, data_u,
             is_init):
        calls.append((col, rsum))
        tag = jnp.asarray(col)[:, 0]
        new = tuple(tag + o for o in _OFFSETS)
        z = jnp.float32(0.0)
        return params, state, new, z, (z, z, z, z, z)

    solver.step_fn = stub
    return solver, params, inputs, calls


@pytest.mark.parametrize("n_col, batch", [(12, 4), (10, 4)],
                         ids=["divisible", "remainder"])
def test_rans_rba_sums_stay_with_their_points(n_col, batch):
    solver, params, inputs, calls = _rans_solver(n_col)
    solver.train(params, inputs, epochs=4, batch_size=batch, seed=0,
                 save_restart_every=0)
    steps = n_col // batch
    assert len(calls) == 4 * steps
    _check_alignment(calls, n_comp=5, steps=steps)
