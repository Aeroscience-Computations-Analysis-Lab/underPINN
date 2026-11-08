"""Tests for the PIKAN networks (underPINN.nn.kan)."""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from underPINN.nn.factory import build_model, network_config
from underPINN.nn.kan import (BSplineKAN, ChebyKAN, bspline_basis, bspline_knots,
                              chebyshev_basis, refine_bspline_kan)

NETS = [
    ("chebykan", lambda: ChebyKAN(layers=[2, 8, 8, 3], degree=4)),
    ("kan", lambda: BSplineKAN(layers=[2, 8, 8, 3], grid_size=5, spline_order=3)),
]


def _init(model, d=2):
    return model.init(jax.random.PRNGKey(0), jnp.ones((1, d)))


# ── bases ────────────────────────────────────────────────────────────────────

def test_chebyshev_matches_numpy():
    t = np.linspace(-1, 1, 41)
    T = np.asarray(chebyshev_basis(jnp.asarray(t), 6))
    for k in range(7):
        ref = np.polynomial.chebyshev.chebval(t, [0] * k + [1])
        np.testing.assert_allclose(T[:, k], ref, atol=1e-5)


def test_chebyshev_derivative_finite_at_endpoints():
    # cos(k*arccos t) would give inf/nan here; the recurrence must not.
    d2 = jax.vmap(jax.hessian(lambda t: chebyshev_basis(t, 5)[5]))(jnp.array([-1.0, 1.0]))
    # T_5''(±1) = ±k^2(k^2-1)/3 = ±200
    np.testing.assert_allclose(np.asarray(d2), [-200.0, 200.0], rtol=1e-4)


@pytest.mark.parametrize("G,k", [(5, 3), (8, 2), (3, 1)])
def test_bspline_partition_of_unity(G, k):
    x = jnp.linspace(-1, 1, 101, endpoint=False)
    B = bspline_basis(x, bspline_knots(G, k), k)
    assert B.shape == (101, G + k)
    np.testing.assert_allclose(np.asarray(B.sum(-1)), 1.0, atol=1e-5)
    assert float(B.min()) >= -1e-7


# ── networks ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,make", NETS)
def test_shape_and_jit(name, make):
    m = make()
    p = _init(m)
    x = jax.random.uniform(jax.random.PRNGKey(1), (17, 2), minval=-1, maxval=1)
    out = jax.jit(m.apply)(p, x)
    assert out.shape == (17, 3)
    assert np.all(np.isfinite(np.asarray(out)))


@pytest.mark.parametrize("name,make", NETS)
def test_hessian_matches_finite_difference(name, make):
    """Second derivatives -- what PDE residuals use -- are correct."""
    # float64 so the finite-difference reference itself is trustworthy
    # (float32 central differences are too noisy at this curvature).
    m = make()
    try:                                   # jax >= 0.7-ish
        x64 = jax.enable_x64(True)
    except (AttributeError, TypeError):    # older jax
        from jax.experimental import enable_x64
        x64 = enable_x64()
    with x64:
        p = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float64), _init(m))
        f = lambda z: m.apply(p, z[None])[0, 0]
        z = jnp.array([0.137, -0.291], jnp.float64)    # away from spline knots
        H = np.asarray(jax.hessian(f)(z))
        h = 1e-5
        fd = np.stack([(np.asarray(jax.grad(f)(z + h * e)) - np.asarray(jax.grad(f)(z - h * e)))
                       / (2 * h) for e in jnp.eye(2, dtype=jnp.float64)])
    np.testing.assert_allclose(H, H.T, atol=1e-10)
    np.testing.assert_allclose(H, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("name,make", NETS)
def test_param_gradients_flow(name, make):
    m = make()
    p = _init(m)
    x = jax.random.uniform(jax.random.PRNGKey(2), (32, 2), minval=-1, maxval=1)
    g = jax.grad(lambda q: jnp.mean(m.apply(q, x) ** 2))(p)
    norms = [float(jnp.linalg.norm(l)) for l in jax.tree_util.tree_leaves(g)]
    assert all(np.isfinite(norms)) and sum(norms) > 0


# ── factory ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cfg,cls,attrs", [
    ({"type": "chebykan", "layers": [2, 4, 1], "degree": 7}, ChebyKAN, {"degree": 7}),
    ({"type": "cpikan", "layers": [2, 4, 1]}, ChebyKAN, {"degree": 5}),
    ({"type": "kan", "layers": [2, 4, 1], "grid_size": 9, "spline_order": 2},
     BSplineKAN, {"grid_size": 9, "spline_order": 2}),
])
def test_factory_builds(cfg, cls, attrs):
    m = build_model(cfg)
    assert isinstance(m, cls)
    for k, v in attrs.items():
        assert getattr(m, k) == v


def test_network_config_round_trip():
    from types import SimpleNamespace
    nc = SimpleNamespace(type="kan", layers=[2, 4, 1], grid_size=7, spline_order=3,
                         grid_range=[-1.0, 1.0])
    meta = network_config(SimpleNamespace(network=nc))
    assert meta == {"type": "kan", "layers": [2, 4, 1], "grid_size": 7,
                    "spline_order": 3, "grid_range": [-1.0, 1.0]}
    assert build_model(meta).grid_size == 7


# ── grid extension ───────────────────────────────────────────────────────────

def test_refine_bspline_is_exact_for_nested_grids():
    old = BSplineKAN(layers=[2, 6, 1], grid_size=4)
    new = BSplineKAN(layers=[2, 6, 1], grid_size=8)
    p_old = _init(old)
    p_new = refine_bspline_kan(p_old, 4, 8)
    assert p_new["params"]["layer_0"]["spline_coef"].shape[-1] == 8 + 3
    x = jax.random.uniform(jax.random.PRNGKey(3), (200, 2), minval=-2, maxval=2)
    np.testing.assert_allclose(np.asarray(new.apply(p_new, x)),
                               np.asarray(old.apply(p_old, x)), atol=2e-4)


def test_refine_bspline_rejects_wrong_grid():
    p = _init(BSplineKAN(layers=[2, 4, 1], grid_size=4))
    with pytest.raises(ValueError, match="spline_coef"):
        refine_bspline_kan(p, 5, 10)


# ── end to end: a PIKAN trains through the real solver (data-parallel on) ─────

@pytest.mark.parametrize("cfg", [{"type": "chebykan", "layers": [2, 16, 16, 1], "degree": 4},
                                 {"type": "kan", "layers": [2, 16, 16, 1], "grid_size": 5}])
def test_burgers_trains(cfg):
    from underPINN.core.config import TrainingConfig
    from underPINN.losses.loss import PINNLoss
    from underPINN.pde.burgers import BurgersPDE
    from underPINN.solver.fbpinn import FBPINNSolver

    model = build_model(cfg)
    pde = BurgersPDE(model, nu=0.01)
    solver = FBPINNSolver(model, pde, loss=PINNLoss(model, pde))
    solver.init(jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    x_r = jnp.asarray(rng.uniform(-1, 1, 512), jnp.float32)
    t_r = jnp.asarray(rng.uniform(0, 1, 512), jnp.float32)
    x_i = jnp.linspace(-1, 1, 64)
    u_i = -jnp.sin(jnp.pi * x_i)
    t_b = jnp.asarray(rng.uniform(0, 1, 32), jnp.float32)
    x_b = jnp.concatenate([-jnp.ones(32), jnp.ones(32)])
    t_b = jnp.concatenate([t_b, t_b])
    tc = TrainingConfig(epochs=200, lr=3e-3, batch_r=128, batch_i=32, batch_b=32,
                        callbacks=[], data_parallel=jax.device_count() > 1)
    solver.train(x_r, t_r, x_i, u_i, x_b, t_b, jnp.zeros(64), config=tc)
    h = solver.loss_hist
    assert np.all(np.isfinite(h))
    assert np.mean(h[-20:]) < 0.5 * np.mean(h[:5])
