"""Model (subdomain) parallelism for FBPINN: StackedFBPINN + ModelParallel.

The claim under test: splitting an FBPINN's subdomain networks across devices
changes where work runs, never the function, its gradient, or its
initialisation. Multi-device tests need a fresh process with virtual devices::

    UNDERPINN_REQUIRE_MULTI_DEVICE=1 \\
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        python -m pytest tests/test_model_parallel.py
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import numpy as np                                                # noqa: E402
import optax                                                      # noqa: E402
import pytest                                                     # noqa: E402

from underPINN.nn.fbpinn import (                                  # noqa: E402
    FBPINN, StackedFBPINN, stack_fbpinn_params, unstack_fbpinn_params)
from underPINN.training.model_parallel import (                    # noqa: E402
    ModelParallel, make_model_parallel, model_parallel_kwargs)

_N_DEV = jax.device_count()
if os.environ.get("UNDERPINN_REQUIRE_MULTI_DEVICE") == "1" and _N_DEV < 2:
    raise RuntimeError(f"UNDERPINN_REQUIRE_MULTI_DEVICE=1 but JAX sees {_N_DEV} "
                       "device(s); set XLA_FLAGS in a fresh process.")
multi = pytest.mark.skipif(_N_DEV < 2, reason=f"needs >=2 devices, saw {_N_DEV}")
leaves = jax.tree_util.tree_leaves


def _fbpinn_2d(nx=3, ny=2, layers=(2, 16, 16, 1)):
    """nx*ny overlapping subdomains on [0,1]^2 (exercises multi-dim windows)."""
    ex, ey = np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1)
    ox, oy = 0.3 * (ex[1] - ex[0]), 0.3 * (ey[1] - ey[0])
    shifts, lo, hi = [], [], []
    for i in range(nx):
        for j in range(ny):
            l = [ex[i] - ox, ey[j] - oy]
            h = [ex[i + 1] + ox, ey[j + 1] + oy]
            lo.append(l); hi.append(h)
            shifts.append([(l[0] + h[0]) / 2, (l[1] + h[1]) / 2])
    A = lambda v: jnp.array(v, jnp.float32)
    S = nx * ny
    return FBPINN(layers=list(layers), shifts=A(shifts), xs_min=A(lo),
                  xs_max=A(hi), smins=jnp.full((S, 2), 0.3 * ox, jnp.float32),
                  smaxs=jnp.full((S, 2), 0.3 * ox, jnp.float32))


_X = jax.random.uniform(jax.random.PRNGKey(7), (48, 2))
_KEY = jax.random.PRNGKey(3)


def _u(model, params):
    return lambda z: model.apply(params, z[None, :])[0, 0]


def _pde_loss(model):
    """Poisson-like residual: needs the Hessian through every subdomain net."""
    def loss(params):
        u = jax.vmap(_u(model, params))(_X)
        lap = jax.vmap(lambda z: jnp.trace(jax.hessian(_u(model, params))(z)))(_X)
        return jnp.mean((lap + u) ** 2)
    return loss


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-30))


# ---------------------------------------------------------------------------
# StackedFBPINN is the same model (runs on any device count)
# ---------------------------------------------------------------------------

class TestStackedFBPINN:
    def test_init_is_bit_identical_to_fbpinn(self):
        ref = _fbpinn_2d()
        st = StackedFBPINN.from_fbpinn(ref)
        want = stack_fbpinn_params(ref.init(_KEY, _X[:1]))
        got = st.init(_KEY, _X[:1])
        for a, b in zip(leaves(want), leaves(got)):
            assert jnp.array_equal(a, b)

    @pytest.mark.parametrize("multiple", [1, 4, 5], ids=["unpadded", "pad2", "pad4"])
    def test_same_function_value_and_derivatives(self, multiple):
        ref = _fbpinn_2d()
        ref_p = ref.init(_KEY, _X[:1])
        st = StackedFBPINN.from_fbpinn(ref, multiple=multiple)
        st_p = st.init(_KEY, _X[:1])
        assert st.n_subdomains == 6 and st.shifts.shape[0] % multiple == 0
        for f in (lambda m, p: jax.vmap(_u(m, p))(_X),
                  lambda m, p: jax.vmap(jax.grad(_u(m, p)))(_X),
                  lambda m, p: jax.vmap(jax.hessian(_u(m, p)))(_X)):
            assert _rel(f(ref, ref_p), f(st, st_p)) < 5e-6

    def test_layout_round_trip_and_export(self):
        ref = _fbpinn_2d()
        p = ref.init(_KEY, _X[:1])
        st = StackedFBPINN.from_fbpinn(ref, multiple=4)          # 6 -> 8
        back = st.export_params(st.init(_KEY, _X[:1]))
        assert sorted(back["params"]) == sorted(p["params"])     # dummies dropped
        for a, b in zip(leaves(p), leaves(back)):
            assert jnp.array_equal(a, b)
        assert unstack_fbpinn_params(stack_fbpinn_params(p)).keys() == p.keys()
        # exported params load straight into the plain model
        np.testing.assert_allclose(ref.apply(back, _X), ref.apply(p, _X))

    def test_dummy_subdomains_contribute_nothing(self):
        st = StackedFBPINN.from_fbpinn(_fbpinn_2d(), multiple=4)  # 2 dummies
        p = st.init(_KEY, _X[:1])
        poisoned = jax.tree_util.tree_map(
            lambda a: a.at[6:].set(1e3), p)        # wild dummy weights
        np.testing.assert_array_equal(st.apply(p, _X), st.apply(poisoned, _X))

    def test_rejects_non_fbpinn(self):
        from underPINN.nn.mlp import MLP
        with pytest.raises(TypeError, match="FBPINN"):
            StackedFBPINN.from_fbpinn(MLP(layers=[2, 4, 1]))


class TestConfig:
    def test_inert_instance_is_a_no_op(self):
        mp = make_model_parallel(active=False)
        model = _fbpinn_2d()
        assert mp.wrap(model) is model and not mp.enabled
        params = {"params": {"subnets_0": jnp.ones(3)}}
        assert mp.prepare(params, "state") == (params, "state")

    @pytest.mark.parametrize("bad", [0, -1, True, 2.5])
    def test_rejects_bad_device_counts(self, bad):
        with pytest.raises(ValueError, match="model_parallel_devices"):
            make_model_parallel(n_devices=bad)

    def test_data_and_model_parallel_together_is_refused(self):
        import types
        tr = types.SimpleNamespace(data_parallel=True, model_parallel=True)
        with pytest.raises(ValueError, match="not supported yet"):
            model_parallel_kwargs(tr)

    def test_training_config_validation(self):
        from underPINN.core.config import TrainingConfig
        with pytest.raises(ValueError, match="model_parallel_devices"):
            TrainingConfig(model_parallel=True, model_parallel_devices=0)
        with pytest.raises(ValueError, match="not supported yet"):
            TrainingConfig(data_parallel=True, model_parallel=True)
        TrainingConfig(model_parallel=True, model_parallel_devices=3)   # valid


# ---------------------------------------------------------------------------
# Splitting across devices
# ---------------------------------------------------------------------------

_COUNTS = [d for d in (2, 3, 4, 5, 8) if d <= _N_DEV] or [2]


class TestModelParallel:
    @multi
    @pytest.mark.parametrize("D", _COUNTS)
    def test_params_and_opt_state_are_split(self, D):
        mp = make_model_parallel(n_devices=D)
        model = mp.wrap(_fbpinn_2d())                 # 6 subdomains
        S = model.shifts.shape[0]
        assert S % D == 0 and model.n_subdomains == 6
        opt = optax.adam(1e-3)
        params = model.init(_KEY, _X[:1])
        params, state = mp.prepare(params, opt.init(params))
        for leaf in leaves(params):
            assert {s.data.shape[0] for s in leaf.addressable_shards} == {S // D}
        mu = state[0].mu
        for leaf in leaves(mu):                       # Adam moments split too
            assert {s.data.shape[0] for s in leaf.addressable_shards} == {S // D}
        count = state[0].count                        # scalar: replicated
        assert len(count.addressable_shards) == D and count.shape == ()

    @multi
    @pytest.mark.parametrize("D", _COUNTS)
    def test_gradient_matches_single_device(self, D):
        ref = _fbpinn_2d()
        ref_p = ref.init(_KEY, _X[:1])
        ref_l, ref_g = jax.value_and_grad(_pde_loss(ref))(ref_p)

        mp = make_model_parallel(n_devices=D)
        model = mp.wrap(ref)
        params, _ = mp.prepare(model.init(_KEY, _X[:1]), ())
        step = jax.jit(jax.value_and_grad(_pde_loss(model)))
        loss, grad = step(params)

        assert abs(float(loss) - float(ref_l)) / float(ref_l) < 5e-6
        real = model.export_params(grad)            # drops dummy subdomains
        for a, b in zip(leaves(ref_g), leaves(real)):
            assert _rel(a, b) < 5e-5
        n_real = model.n_subdomains
        for leaf in leaves(grad):                   # dummies: exactly zero
            assert bool(jnp.all(leaf[n_real:] == 0))
        hlo = step.lower(params).compile().as_text()
        assert "all-reduce" in hlo, "subdomains not actually split"

    @multi
    def test_optimizer_never_moves_dummy_subdomains(self):
        mp = make_model_parallel(n_devices=min(4, _N_DEV))   # 6 -> pad to 8
        model = mp.wrap(_fbpinn_2d())
        if model.n_pad == 0:
            pytest.skip("this device count needs no padding")
        opt = optax.adam(1e-2)
        p0 = model.init(_KEY, _X[:1])
        params, state = mp.prepare(p0, opt.init(p0))

        @jax.jit
        def step(p, s):
            g = jax.grad(_pde_loss(model))(p)
            u, s = opt.update(g, s)
            return optax.apply_updates(p, u), s

        for _ in range(3):
            params, state = step(params, state)
        n = model.n_subdomains
        for before, after in zip(leaves(p0), leaves(params)):
            np.testing.assert_array_equal(np.asarray(before)[n:],
                                          np.asarray(after)[n:])

    @multi
    def test_prepare_explains_unwrapped_fbpinn_params(self):
        mp = make_model_parallel(n_devices=2)
        plain = _fbpinn_2d().init(_KEY, _X[:1])
        with pytest.raises(ValueError, match=r"mp\.wrap"):
            mp.prepare(plain, ())

    @multi
    def test_prepare_and_wrap_refuse_non_fbpinn(self):
        from underPINN.nn.mlp import MLP
        mp = make_model_parallel(n_devices=2)
        with pytest.raises(TypeError, match="data parallelism"):
            mp.wrap(MLP(layers=[2, 4, 1]))
        mlp_params = MLP(layers=[2, 4, 1]).init(_KEY, _X[:1])
        with pytest.raises(ValueError, match="FBPINN networks only"):
            mp.prepare(mlp_params, ())

    @multi
    def test_export_gives_a_plain_fbpinn_checkpoint(self):
        ref = _fbpinn_2d()
        mp = make_model_parallel(n_devices=min(4, _N_DEV))
        model = mp.wrap(ref)
        params, _ = mp.prepare(model.init(_KEY, _X[:1]), ())
        exported = mp.export(model, params)
        assert sorted(exported["params"]) == [f"subnets_{i}" for i in range(6)]
        np.testing.assert_allclose(ref.apply(exported, _X),
                                   np.asarray(model.apply(params, _X)),
                                   rtol=1e-5, atol=1e-7)

    def test_is_a_model_parallel_instance(self):
        assert isinstance(make_model_parallel(), ModelParallel)


# ---------------------------------------------------------------------------
# Solvers: training with model_parallel follows the single-device trajectory
# ---------------------------------------------------------------------------

def _fbpinn_solver_run(D):
    """Burgers through FBPINNSolver with a 6-subdomain FBPINN."""
    from underPINN.core.config import TrainingConfig
    from underPINN.losses.loss import PINNLoss
    from underPINN.pde.burgers import BurgersPDE
    from underPINN.solver.fbpinn import FBPINNSolver

    mp = make_model_parallel(active=D is not None, n_devices=D)
    model = mp.wrap(_fbpinn_2d())
    pde = BurgersPDE(model, nu=0.01)
    solver = FBPINNSolver(model, pde,
                          loss=PINNLoss(model, pde, ic_weight=100.0, bc_weight=10.0))
    solver.init(_KEY)
    rng = np.random.default_rng(0)
    f = lambda n, lo=0.0, hi=1.0: jnp.array(rng.uniform(lo, hi, n).astype("f4"))
    x_r, t_r, x_i = f(64), f(64), f(16)
    data = (x_r, t_r, x_i, jnp.sin(jnp.pi * x_i), f(16), f(16), jnp.zeros(16))
    solver.train(*data, config=TrainingConfig(
        epochs=6, batch_r=32, batch_i=8, batch_b=8, seed=0, log_every=10_000,
        model_parallel=D is not None, model_parallel_devices=D))
    return solver, np.asarray(solver.loss_hist)


_SOLVER_COUNTS = [d for d in (2, 4, 5) if d <= _N_DEV] or [2]    # even + padded


class TestSolvers:
    @multi
    @pytest.mark.parametrize("D", _SOLVER_COUNTS)
    def test_fbpinn_solver_trajectory_matches(self, D):
        _, ref = _fbpinn_solver_run(None)
        solver, par = _fbpinn_solver_run(D)
        np.testing.assert_allclose(par, ref, rtol=1e-4, atol=1e-6)
        S = solver.model.shifts.shape[0]
        for leaf in leaves(solver.params):            # stays split after training
            assert {s.data.shape[0] for s in leaf.addressable_shards} == {S // D}

    @multi
    @pytest.mark.parametrize("D", _SOLVER_COUNTS)
    def test_ldc_solver_trajectory_matches(self, D):
        """Includes LDCSolver's per-point RBA state."""
        from underPINN.core.config import TrainingConfig
        from underPINN.pde.navier_stokes import NavierStokesPDE
        from underPINN.solver.ldc_solver import LDCInputWrapper, LDCSolver

        def run(D):
            mp = make_model_parallel(active=D is not None, n_devices=D)
            model = mp.wrap(_fbpinn_2d(layers=(2, 16, 16, 3)))
            solver = LDCSolver(model, NavierStokesPDE(model, Re=100.0),
                               optimizer=optax.adam(1e-3))
            solver.init(_KEY)
            k = jax.random.split(jax.random.PRNGKey(5), 3)
            inputs = LDCInputWrapper(col=jax.random.uniform(k[0], (32, 2)),
                                     inlet=jax.random.uniform(k[1], (16, 2)),
                                     noslip=jax.random.uniform(k[2], (16, 2)))
            solver.train(inputs, config=TrainingConfig(
                epochs=3, batch_r=16, seed=0,
                model_parallel=D is not None, model_parallel_devices=D))
            return np.asarray(solver.loss_hist)

        np.testing.assert_allclose(run(D), run(None), rtol=1e-4, atol=1e-6)

    @multi
    @pytest.mark.parametrize("D", [d for d in (2, 3) if d <= _N_DEV] or [2])
    def test_rans_solver_trajectory_matches(self, D):
        """K-Epsilon's real FBPINN (3 subdomains, positivity transform) via
        RANSSolver's legacy-kwarg path, as the example calls it; synthetic
        observations stand in for the external CFD dataset."""
        import importlib.util
        from underPINN.pde.k_epsilon import KEpsilonPDE
        from underPINN.solver.rans_solver import RANSInputWrapper, RANSSolver

        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            "_keps_mp", os.path.join(here, "..", "examples", "PINNs",
                                     "K-Epsilon", "turbulence.py"))
        kmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kmod)
        x_col, x_in, x_out, x_ns = kmod._build_bfs_geometry(32, 8, 8, 16, seed=0)
        rng = np.random.default_rng(0)
        inputs = RANSInputWrapper(
            col=jnp.array(x_col), inlet=jnp.array(x_in), noslip=jnp.array(x_ns),
            outlet=jnp.array(x_out), data_x=jnp.array(x_col[:12]),
            data_u=jnp.array(rng.uniform(0.1, 1.0, (12, 5)).astype("f4")))

        def run(D):
            mp = make_model_parallel(active=D is not None, n_devices=D)
            xs_min = jnp.array([[0.0, 0.0], [12.0, 0.0], [24.0, 0.0]])
            xs_max = jnp.array([[12.0, 2.0], [24.0, 2.0], [35.0, 2.0]])
            model = mp.wrap(FBPINN(
                layers=[2, 16, 16, 5],
                shifts=jnp.array([[6.0, 1.0], [18.0, 1.0], [30.0, 1.0]]),
                xs_min=xs_min, xs_max=xs_max, smins=jnp.ones_like(xs_min),
                smaxs=jnp.ones_like(xs_max), out_transform=kmod.k_eps_positivity))
            solver = RANSSolver(model, KEpsilonPDE(model, Re=10000.0),
                                optax.adam(1e-3))
            params = model.init(_KEY, jnp.ones((1, 2)))
            _p, hist = solver.train(params, inputs, epochs=3, batch_size=16,
                                    seed=0, save_restart_every=0,
                                    model_parallel=D is not None,
                                    model_parallel_devices=D)
            return np.asarray(hist)

        np.testing.assert_allclose(run(D), run(None), rtol=1e-4, atol=1e-6)

    @multi
    def test_solver_refuses_model_parallel_for_non_fbpinn(self):
        from underPINN.core.config import TrainingConfig
        from underPINN.losses.steady_loss import SteadyLoss
        from underPINN.nn.mlp import MLP
        from underPINN.pde.heat import SteadyHeatPDE
        from underPINN.solver.steady_solver import SteadySolver
        model = MLP(layers=[2, 8, 1])
        pde = SteadyHeatPDE(model)
        solver = SteadySolver(model, pde, SteadyLoss(model, pde))
        solver.init(_KEY)
        with pytest.raises(ValueError, match="FBPINN networks only"):
            solver.train(_X, _X[:16], jnp.zeros(16), config=TrainingConfig(
                epochs=1, batch_r=16, batch_b=8, model_parallel=True,
                model_parallel_devices=2))

    @multi
    def test_warns_when_devices_would_hold_only_padding(self):
        one = FBPINN(layers=[2, 8, 1], shifts=jnp.array([[0.5, 0.5]]),
                     xs_min=jnp.zeros((1, 2)), xs_max=jnp.ones((1, 2)),
                     smins=jnp.full((1, 2), 0.4), smaxs=jnp.full((1, 2), 0.4))
        mp = make_model_parallel(n_devices=2)
        model = mp.wrap(one)                       # 1 real + 1 dummy
        params = model.init(_KEY, _X[:1])
        with pytest.warns(UserWarning, match="only masked padding"):
            mp.prepare(params, (), model=model)
