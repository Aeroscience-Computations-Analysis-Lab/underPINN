"""Correctness tests for data-parallel training (underPINN.training.parallel).

The claim under test is narrow and important: sharding collocation points
across devices must produce the *same gradient* as computing the full batch on
one device. Data parallelism is only useful if it is numerically a no-op.

These tests need more than one device. JAX can fake that on a CPU host, but the
flag must be set before JAX initialises its backend, so run this file in its
own process::

    UNDERPINN_REQUIRE_MULTI_DEVICE=1 \\
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        python -m pytest tests/test_parallel.py

CI does exactly this as a separate step. Inside a plain ``pytest tests/`` run
there is one device and the multi-device tests skip -- so a green full-suite
run is *not* evidence that sharding works; the dedicated step is.

This file deliberately does not set ``XLA_FLAGS`` itself. JAX reads it once,
when the backend first initialises, so a test module that set it would change
the device topology for whichever modules happened to be collected after it --
behaviour that depends on test ordering is not behaviour worth having.

``UNDERPINN_REQUIRE_MULTI_DEVICE=1`` turns the skips into failures, so a
misconfigured run (flag ignored, one device) cannot pass green.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402

from underPINN.losses.loss import PINNLoss                         # noqa: E402
from underPINN.nn.mlp import MLP                                   # noqa: E402
from underPINN.pde.burgers import BurgersPDE                       # noqa: E402
from underPINN.training.parallel import (                          # noqa: E402
    DataParallel,
    make_data_parallel,
    pad_to_multiple,
)

_N_DEV = jax.device_count()
_REQUIRE = os.environ.get("UNDERPINN_REQUIRE_MULTI_DEVICE") == "1"

if _REQUIRE and _N_DEV < 2:
    raise RuntimeError(
        f"UNDERPINN_REQUIRE_MULTI_DEVICE=1 but JAX sees {_N_DEV} device(s). "
        "XLA_FLAGS=--xla_force_host_platform_device_count=N did not take "
        "effect -- it must be set before the JAX backend initialises, i.e. in "
        "the environment of a fresh process.")

requires_multi_device = pytest.mark.skipif(
    _N_DEV < 2,
    reason=(f"needs >=2 JAX devices, saw {_N_DEV}; rerun this file alone with "
            "XLA_FLAGS=--xla_force_host_platform_device_count=8"),
)


# ---------------------------------------------------------------------------
# A real (small) PINN problem, so the test exercises the production loss path
# ---------------------------------------------------------------------------

def _problem(n_r: int = 64, n_i: int = 16, n_b: int = 16, seed: int = 0):
    """Burgers collocation/IC/BC sets plus the real PINNLoss closure."""
    rng = np.random.default_rng(seed)
    x_r = jnp.array(rng.uniform(-1.0, 1.0, n_r).astype("f4"))
    t_r = jnp.array(rng.uniform(0.0, 1.0, n_r).astype("f4"))
    x_i = jnp.array(np.linspace(-1, 1, n_i, dtype="f4"))
    u_i = jnp.array(-np.sin(np.pi * np.asarray(x_i)).astype("f4"))
    x_b = jnp.array(np.tile([-1.0, 1.0], n_b // 2).astype("f4"))
    t_b = jnp.array(rng.uniform(0.0, 1.0, n_b).astype("f4"))
    u_b = jnp.zeros(n_b, dtype="f4")

    model = MLP(layers=[2, 16, 16, 1])
    pde = BurgersPDE(model, nu=0.01)
    loss = PINNLoss(model, pde, ic_weight=100.0, bc_weight=10.0)
    params = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 2)))

    def loss_of(p, batch):
        return loss(p, *batch)

    batch = (x_r, t_r, x_i, u_i, x_b, t_b, u_b)
    return params, batch, loss_of


def _grad(params, batch, loss_of):
    (val, _aux), g = jax.value_and_grad(
        lambda p: loss_of(p, batch), has_aux=True)(params)
    return float(val), g


# ---------------------------------------------------------------------------
# Mesh construction — works on any device count
# ---------------------------------------------------------------------------

class TestMeshConstruction:
    def test_builds_over_all_visible_devices(self):
        dp = make_data_parallel()
        assert isinstance(dp, DataParallel)
        assert dp.n_devices == jax.device_count()
        assert len(dp.devices) == dp.n_devices

    def test_enabled_reflects_device_count(self):
        dp = make_data_parallel()
        assert dp.enabled == (jax.device_count() > 1)

    def test_single_device_subset_is_a_no_op(self):
        dp = make_data_parallel(devices=jax.devices()[:1])
        assert dp.n_devices == 1
        assert dp.enabled is False
        # A no-op mesh must still accept and return usable arrays.
        x = jnp.arange(7.0)
        assert jnp.allclose(dp.shard(x), x)


# ---------------------------------------------------------------------------
# Divisibility guard — early, field-named error instead of JAX's shape error
# ---------------------------------------------------------------------------

class TestDivisibilityGuard:
    @requires_multi_device
    def test_rejects_uneven_batch_and_names_the_field(self):
        dp = make_data_parallel()
        d = dp.n_devices
        with pytest.raises(ValueError) as exc:
            dp.require_divisible({"batch_r": d * 4 + 1})
        msg = str(exc.value)
        assert "batch_r" in msg
        assert str(d) in msg
        # must suggest concrete usable values, not just complain
        assert str(d * 4) in msg

    @requires_multi_device
    def test_accepts_divisible_batch(self):
        dp = make_data_parallel()
        dp.require_divisible({"batch_r": dp.n_devices * 8,
                              "batch_i": dp.n_devices * 2})

    def test_single_device_accepts_anything(self):
        dp = make_data_parallel(devices=jax.devices()[:1])
        dp.require_divisible({"batch_r": 4097})     # no constraint to violate

    @requires_multi_device
    def test_jax_itself_rejects_uneven_sharding(self):
        """Pins the behaviour parallel.py's docstring relies on.

        The divisibility guard is documented as a *usability* improvement over
        JAX's own error, not a correctness safeguard -- because JAX refuses an
        uneven split outright rather than mis-weighting it. If a future JAX
        starts accepting uneven shards, this test fails and that docstring
        must be revisited.
        """
        dp = make_data_parallel()
        with pytest.raises(ValueError):
            dp.shard(jnp.arange(float(dp.n_devices * 4 + 1)))


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

class TestPlacement:
    @requires_multi_device
    def test_shard_splits_leading_axis_across_devices(self):
        dp = make_data_parallel()
        d = dp.n_devices
        x = jnp.arange(float(8 * d))
        xs = dp.shard(x)
        shards = xs.addressable_shards
        assert len(shards) == d
        assert all(s.data.shape == (8,) for s in shards)
        assert jnp.allclose(xs, x)                  # values unchanged

    @requires_multi_device
    def test_replicate_puts_full_copy_on_every_device(self):
        dp = make_data_parallel()
        params, _batch, _ = _problem()
        rep = dp.replicate(params)
        leaf = jax.tree_util.tree_leaves(rep)[0]
        full = jax.tree_util.tree_leaves(params)[0].shape
        assert all(s.data.shape == full for s in leaf.addressable_shards)

    @requires_multi_device
    def test_shard_all_handles_a_tuple_of_batches(self):
        dp = make_data_parallel()
        d = dp.n_devices
        arrays = (jnp.arange(float(4 * d)), jnp.arange(float(4 * d)))
        out = dp.shard_all(arrays)
        assert len(out) == 2
        assert all(len(a.addressable_shards) == d for a in out)


# ---------------------------------------------------------------------------
# The central claim: sharded gradient == single-device gradient
# ---------------------------------------------------------------------------

class TestGradientEquivalence:
    @requires_multi_device
    def test_sharded_gradient_matches_single_device(self):
        dp = make_data_parallel()
        d = dp.n_devices
        # every point count divisible by the device count
        params, batch, loss_of = _problem(n_r=16 * d, n_i=4 * d, n_b=4 * d)

        ref_loss, ref_grad = _grad(params, batch, loss_of)

        sharded_batch = dp.shard_all(batch)
        par_params = dp.replicate(params)
        par_loss, par_grad = _grad(par_params, sharded_batch, loss_of)

        assert np.isfinite(ref_loss) and ref_loss > 0
        # float32 all-reduce reorders the sum, so compare with a tolerance
        # rather than bit-for-bit.
        assert par_loss == pytest.approx(ref_loss, rel=1e-5), (
            f"loss diverged: single={ref_loss!r} sharded={par_loss!r}")

        ref_flat = jax.tree_util.tree_leaves(ref_grad)
        par_flat = jax.tree_util.tree_leaves(par_grad)
        assert len(ref_flat) == len(par_flat)
        for i, (a, b) in enumerate(zip(ref_flat, par_flat)):
            assert a.shape == b.shape
            np.testing.assert_allclose(
                np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-6,
                err_msg=f"gradient leaf {i} differs between sharded and "
                        f"single-device execution")

    @requires_multi_device
    def test_gradient_is_not_trivially_zero(self):
        """Guards the test above: an all-zero gradient would match trivially."""
        dp = make_data_parallel()
        d = dp.n_devices
        params, batch, loss_of = _problem(n_r=16 * d, n_i=4 * d, n_b=4 * d)
        _, g = _grad(dp.replicate(params), dp.shard_all(batch), loss_of)
        norms = [float(jnp.linalg.norm(x))
                 for x in jax.tree_util.tree_leaves(g)]
        assert max(norms) > 1e-8, "gradient is ~0; equivalence test is vacuous"

    @requires_multi_device
    def test_optimizer_step_keeps_params_replicated(self):
        """Replicated params + all-reduced grads must stay identical per device."""
        import optax
        dp = make_data_parallel()
        d = dp.n_devices
        params, batch, loss_of = _problem(n_r=16 * d, n_i=4 * d, n_b=4 * d)

        opt = optax.adam(1e-3)
        par_params = dp.replicate(params)
        state = dp.replicate(opt.init(params))
        sharded = dp.shard_all(batch)

        @jax.jit
        def step(p, s):
            (_v, _a), g = jax.value_and_grad(
                lambda q: loss_of(q, sharded), has_aux=True)(p)
            upd, s = opt.update(g, s)
            return optax.apply_updates(p, upd), s

        new_params, _ = step(par_params, state)
        for leaf in jax.tree_util.tree_leaves(new_params):
            datas = [np.asarray(s.data) for s in leaf.addressable_shards]
            for other in datas[1:]:
                np.testing.assert_array_equal(
                    datas[0], other,
                    err_msg="params diverged across devices; the gradient "
                            "all-reduce is missing or the params were sharded")


# ---------------------------------------------------------------------------
# End-to-end: FBPINNSolver(data_parallel=True) trains exactly like single-device
# ---------------------------------------------------------------------------

def _train_solver(data_parallel: bool, n_scan: int, epochs: int = 12,
                  batch: int | None = None):
    """Train a real FBPINNSolver on Burgers; return (solver, loss_hist)."""
    from underPINN.core.config import TrainingConfig
    from underPINN.solver.fbpinn import FBPINNSolver

    d = jax.device_count()
    bs = batch if batch is not None else 8 * d
    params, (x_r, t_r, x_i, u_i, x_b, t_b, u_b), _ = _problem(
        n_r=32 * d, n_i=8 * d, n_b=8 * d)
    model = MLP(layers=[2, 16, 16, 1])
    pde = BurgersPDE(model, nu=0.01)
    loss = PINNLoss(model, pde, ic_weight=100.0, bc_weight=10.0)

    solver = FBPINNSolver(model, pde, loss=loss)
    solver.init(jax.random.PRNGKey(0))
    cfg = TrainingConfig(epochs=epochs, lr=1e-3, batch_r=bs, batch_i=bs,
                         batch_b=bs, seed=0, n_scan_steps=n_scan,
                         data_parallel=data_parallel, log_every=10_000)
    solver.train(x_r, t_r, x_i, u_i, x_b, t_b, u_b, config=cfg)
    return solver, np.asarray(solver.loss_hist)


class TestFBPINNSolverDataParallel:
    @requires_multi_device
    @pytest.mark.parametrize("n_scan", [1, 4], ids=["python_loop", "lax_scan"])
    def test_loss_trajectory_matches_single_device(self, n_scan):
        """Same seed => same batches => same updates, on both training paths.

        This is the claim that makes data_parallel safe to switch on: it must
        not change what the optimiser does, only where the work runs.
        """
        _, ref = _train_solver(False, n_scan)
        _, par = _train_solver(True, n_scan)
        assert ref.shape == par.shape and ref.size > 0
        assert np.all(np.isfinite(par))
        # Float32 all-reduce reorders sums; Adam then compounds that noise over
        # steps, so tolerance is looser than for a single gradient.
        np.testing.assert_allclose(
            par, ref, rtol=1e-3, atol=1e-5,
            err_msg=f"data-parallel loss trajectory diverged (n_scan={n_scan})")

    @requires_multi_device
    def test_trained_params_end_up_replicated(self):
        solver, _ = _train_solver(True, n_scan=1)
        for leaf in jax.tree_util.tree_leaves(solver.params):
            shards = [np.asarray(s.data) for s in leaf.addressable_shards]
            assert len(shards) == jax.device_count()
            for other in shards[1:]:
                np.testing.assert_array_equal(shards[0], other)

    @requires_multi_device
    def test_inference_works_after_parallel_training(self):
        """Replicated params must still apply to an ordinary unsharded input."""
        solver, _ = _train_solver(True, n_scan=1)
        pts = jnp.stack([jnp.linspace(-1, 1, 7), jnp.full(7, 0.5)], axis=1)
        out = solver.model.apply(solver.params, pts)
        assert out.shape == (7, 1)
        assert np.all(np.isfinite(np.asarray(out)))

    @requires_multi_device
    def test_indivisible_batch_raises_named_error_before_training(self):
        d = jax.device_count()
        with pytest.raises(ValueError, match="batch_r"):
            _train_solver(True, n_scan=1, batch=8 * d + 1)

    def test_flag_is_a_no_op_on_one_device(self):
        """data_parallel=True must not break single-device training."""
        if jax.device_count() != 1:
            pytest.skip("checks the single-device path specifically")
        _, hist = _train_solver(True, n_scan=1, batch=13)  # not divisible: fine
        assert hist.size > 0 and np.all(np.isfinite(hist))


# ---------------------------------------------------------------------------
# Every PINN solver supports data_parallel (not just FBPINNSolver)
# ---------------------------------------------------------------------------

def _steady_solver():
    from underPINN.losses.steady_loss import SteadyLoss
    from underPINN.pde.heat import SteadyHeatPDE
    from underPINN.solver.steady_solver import SteadySolver

    model = MLP(layers=[2, 8, 1])
    pde = SteadyHeatPDE(model)
    solver = SteadySolver(model, pde, SteadyLoss(model, pde))
    solver.init(jax.random.PRNGKey(0))
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    data = (jax.random.uniform(k1, (40, 2)), jax.random.uniform(k2, (20, 2)),
            jnp.zeros(20))
    return solver, data


def _ode_solver():
    from underPINN.losses.ode_loss import ODELoss
    from underPINN.pde.ode import ExponentialDecayODE
    from underPINN.solver.ode_solver import ODESolver

    model = MLP(layers=[1, 16, 16, 1])
    pde = ExponentialDecayODE(model, lam=1.0)
    solver = ODESolver(model, pde, ODELoss(model, pde))
    solver.init(jax.random.PRNGKey(0))
    return solver


class TestSolverDataParallel:
    @requires_multi_device
    def test_steady_solver_trajectory_matches(self):
        from underPINN.core.config import TrainingConfig

        def run(on):
            solver, data = _steady_solver()
            solver.train(*data, config=TrainingConfig(
                epochs=6, batch_r=8 * _N_DEV, batch_b=2 * _N_DEV,
                data_parallel=on))
            return np.asarray(solver.loss_hist)

        np.testing.assert_allclose(run(True), run(False), rtol=1e-3, atol=1e-6)

    @requires_multi_device
    @pytest.mark.parametrize("n_scan", [1, 3], ids=["python_loop", "lax_scan"])
    def test_ode_solver_trajectory_matches(self, n_scan):
        from underPINN.core.config import TrainingConfig
        t_r = jnp.linspace(0.0, 2.0, 8 * _N_DEV)

        def run(on):
            solver = _ode_solver()
            solver.train(t_r, jnp.array([0.0]), jnp.array([1.0]),
                         config=TrainingConfig(epochs=6, n_scan_steps=n_scan,
                                               data_parallel=on))
            return np.asarray(solver.loss_hist)

        ref, par = run(False), run(True)
        assert ref.size == 6
        np.testing.assert_allclose(par, ref, rtol=1e-3, atol=1e-6)

    @requires_multi_device
    def test_ode_scan_step_really_splits_t_r(self):
        """The lax.scan path used to close over t_r. jit bakes a closed-over
        array into the program as a full per-device constant -- no split, no
        all-reduce -- so a trajectory match alone could not detect it. This
        inspects the compiled program instead."""
        import optax
        dp = make_data_parallel()
        solver = _ode_solver()
        solver.opt = optax.adam(1e-3)
        t_r = jnp.linspace(0.0, 2.0, 8 * _N_DEV)
        scan_step = solver._build_scan_step(t_r, jnp.array([0.0]),
                                            jnp.array([1.0]), jnp.zeros(1))
        params = dp.replicate(solver.params)
        state = dp.replicate(solver.opt.init(solver.params))
        hlo = scan_step.lower(params, state, jnp.zeros(3),
                              dp.shard(t_r)).compile().as_text()
        assert "all-reduce" in hlo, (
            "compiled ODE scan step has no all-reduce: t_r is not being split "
            "across devices (closure capture regression?)")

    @requires_multi_device
    def test_rans_solver_trajectory_matches(self):
        """K-Epsilon's real geometry, FBPINN model and k-eps PDE, with
        synthetic observations standing in for the external CFD dataset the
        example needs. Uses RANSSolver's legacy-kwarg path, as the example
        does."""
        import importlib.util
        import optax
        from underPINN.nn.fbpinn import FBPINN
        from underPINN.pde.k_epsilon import KEpsilonPDE
        from underPINN.solver.rans_solver import RANSInputWrapper, RANSSolver

        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            "_keps", os.path.join(here, "..", "examples", "PINNs",
                                  "K-Epsilon", "turbulence.py"))
        kmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kmod)

        n = 8 * _N_DEV
        x_col, x_in, x_out, x_ns = kmod._build_bfs_geometry(n * 2, 16, 16, 32,
                                                            seed=0)
        rng = np.random.default_rng(0)
        data_x = x_col[rng.choice(len(x_col), 24, replace=False)]
        data_u = rng.uniform(0.1, 1.0, (24, 5)).astype(np.float32)
        inputs = RANSInputWrapper(
            col=jnp.array(x_col), inlet=jnp.array(x_in),
            noslip=jnp.array(x_ns), outlet=jnp.array(x_out),
            data_x=jnp.array(data_x), data_u=jnp.array(data_u))

        def run(on):
            xs_min = jnp.array([[0.0, 0.0], [12.0, 0.0]])
            xs_max = jnp.array([[12.0, 2.0], [35.0, 2.0]])
            model = FBPINN(layers=[2, 16, 16, 5],
                           shifts=jnp.array([[6.0, 1.0], [24.0, 1.0]]),
                           xs_min=xs_min, xs_max=xs_max,
                           smins=jnp.ones_like(xs_min),
                           smaxs=jnp.ones_like(xs_max),
                           out_transform=kmod.k_eps_positivity)
            pde = KEpsilonPDE(model, Re=10000.0)
            solver = RANSSolver(model, pde, optax.adam(1e-3))
            params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))
            _p, hist = solver.train(params, inputs, epochs=3, batch_size=n,
                                    seed=0, save_restart_every=0,
                                    data_parallel=on)
            return np.asarray(hist)

        ref, par = run(False), run(True)
        assert ref.size == 3 and np.all(np.isfinite(par))
        np.testing.assert_allclose(par, ref, rtol=2e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# pad_to_multiple
# ---------------------------------------------------------------------------

class TestPadToMultiple:
    def test_divisible_input_is_returned_untouched(self):
        x = jnp.arange(12.0)
        out, n = pad_to_multiple(x, 4)
        assert n == 12
        assert out.shape == (12,)
        assert jnp.allclose(out, x)

    def test_pads_up_to_next_multiple(self):
        x = jnp.arange(10.0)
        out, n = pad_to_multiple(x, 4)
        assert n == 10
        assert out.shape == (12,)
        assert jnp.allclose(out[:10], x)

    def test_padding_repeats_existing_rows(self):
        x = jnp.arange(5.0)
        out, _ = pad_to_multiple(x, 4)
        assert out.shape == (8,)
        # wrapped repeats, so every padded value came from the original
        assert set(np.asarray(out[5:]).tolist()) <= set(np.asarray(x).tolist())

    def test_works_on_2d_along_axis_0(self):
        x = jnp.arange(10.0).reshape(5, 2)
        out, n = pad_to_multiple(x, 4)
        assert n == 5
        assert out.shape == (8, 2)
        assert jnp.allclose(out[:5], x)

    def test_rejects_bad_multiple(self):
        with pytest.raises(ValueError):
            pad_to_multiple(jnp.arange(4.0), 0)
