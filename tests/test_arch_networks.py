"""Tests for PirateNet, PINNsFormer, PINNMamba, the failure-mode PDEs and the
PINNLoss periodic / initial-velocity options."""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from underPINN.losses.loss import PINNLoss
from underPINN.nn.factory import build_model
from underPINN.nn.pirate import PirateNet, pirate_output_init
from underPINN.nn.sequence import MambaBlock, PINNMamba, PINNsFormer, pseudo_sequence
from underPINN.pde.failure_modes import AllenCahnPDE, ConvectionPDE, ReactionPDE

SMALL = {
    "piratenet": {"type": "piratenet", "width": 16, "n_blocks": 2},
    "pinnsformer": {"type": "pinnsformer", "d_model": 8, "heads": 2, "d_ff": 16, "d_hidden": 16},
    "pinnmamba": {"type": "pinnmamba", "d_model": 8, "n_layers": 1, "d_state": 4, "d_hidden": 16},
}


def _x64():
    try:
        return jax.enable_x64(True)
    except (AttributeError, TypeError):
        from jax.experimental import enable_x64
        return enable_x64()


@pytest.mark.parametrize("name", sorted(SMALL))
def test_shape_and_factory(name):
    m = build_model(SMALL[name])
    x = jax.random.uniform(jax.random.PRNGKey(0), (7, 2), minval=-1, maxval=1)
    p = m.init(jax.random.PRNGKey(1), x[:1])
    out = jax.jit(m.apply)(p, x)
    assert out.shape == (7, 1) and np.all(np.isfinite(np.asarray(out)))


@pytest.mark.parametrize("name", sorted(SMALL))
def test_hessian_matches_finite_difference(name):
    m = build_model(SMALL[name])
    with _x64():
        p = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float64),
                                   m.init(jax.random.PRNGKey(1), jnp.zeros((1, 2))))
        f = lambda z: m.apply(p, z[None])[0, 0]
        z = jnp.array([0.21, -0.37], jnp.float64)
        H = np.asarray(jax.hessian(f)(z))
        h = 1e-5
        fd = np.stack([(np.asarray(jax.grad(f)(z + h * e)) - np.asarray(jax.grad(f)(z - h * e))) / (2 * h)
                       for e in jnp.eye(2, dtype=jnp.float64)])
    np.testing.assert_allclose(H, H.T, atol=1e-9)
    np.testing.assert_allclose(H, fd, rtol=1e-4, atol=1e-6)


def test_pseudo_sequence():
    x = jnp.array([[0.5, 0.1], [-0.2, 0.3]])
    s = pseudo_sequence(x, 3, 0.01)
    np.testing.assert_allclose(np.asarray(s[:, :, 1]), [[0.1, 0.11, 0.12], [0.3, 0.31, 0.32]], atol=1e-7)
    np.testing.assert_allclose(np.asarray(s[:, :, 0]), [[0.5] * 3, [-0.2] * 3])
    b = pseudo_sequence(x, 3, 0.01, backward=True)
    np.testing.assert_allclose(np.asarray(b[:, -1]), np.asarray(x), atol=1e-7)   # last token = the point


def test_mamba_is_causal():
    blk = MambaBlock(d_model=8, d_state=4)
    u = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 8))
    p = blk.init(jax.random.PRNGKey(1), u)
    y = blk.apply(p, u)
    u2 = u.at[:, 4:].add(1.0)                      # change only the future
    y2 = blk.apply(p, u2)
    np.testing.assert_allclose(np.asarray(y[:, :4]), np.asarray(y2[:, :4]), atol=1e-6)
    assert float(jnp.abs(y[:, 4:] - y2[:, 4:]).max()) > 1e-4


@pytest.mark.parametrize("cls,kw", [(PINNsFormer, dict(d_model=8, d_ff=16, d_hidden=16)),
                                    (PINNMamba, dict(d_model=8, n_layers=1, d_state=4, d_hidden=16))])
def test_sequence_context_is_used(cls, kw):
    """The read-out token must depend on the rest of the window."""
    x = jax.random.uniform(jax.random.PRNGKey(0), (5, 2), minval=-1, maxval=1)
    m1, m2 = cls(dt=0.05, **kw), cls(dt=0.10, **kw)
    p = m1.init(jax.random.PRNGKey(1), x[:1])
    assert float(jnp.abs(m1.apply(p, x) - m2.apply(p, x)).max()) > 1e-5


def test_pirate_starts_linear_and_ls_init_is_exact():
    m = PirateNet(width=32, n_blocks=3)
    p = m.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
    x = jax.random.uniform(jax.random.PRNGKey(1), (200, 2), minval=-1, maxval=1)
    # alpha = 0  =>  features are exactly the Fourier embedding
    feats = m.apply(p, x, method=PirateNet.features)
    emb = m.apply(p, x, method=lambda mod, z: mod.embed(z))
    np.testing.assert_allclose(np.asarray(feats), np.asarray(emb), atol=1e-6)
    # a target in the span of the features is fitted exactly
    w = jax.random.normal(jax.random.PRNGKey(2), (32,))
    target = feats @ w + 0.3
    p2 = pirate_output_init(p, feats, target, head_path=("params", "head"), ridge=1e-10)
    np.testing.assert_allclose(np.asarray(m.apply(p2, x)[:, 0]), np.asarray(target), atol=2e-3)


class _Exact(nn.Module):
    fn: callable

    @nn.compact
    def __call__(self, xt):
        self.param("dummy", nn.initializers.zeros, ())
        return self.fn(xt[:, 0], xt[:, 1])[:, None]


def _res(pde_cls, fn, **kw):
    m = _Exact(fn)
    p = m.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
    xt = jax.random.uniform(jax.random.PRNGKey(3), (50, 2)) * jnp.array([6.0, 1.0])
    return np.asarray(pde_cls(m, **kw).residual(p, xt))


def test_convection_residual_vanishes_on_exact():
    assert np.abs(_res(ConvectionPDE, lambda x, t: jnp.sin(x - 50.0 * t), beta=50.0)).max() < 1e-3


def test_reaction_residual_vanishes_on_exact():
    pde = ReactionPDE(None, rho=5.0)
    assert np.abs(_res(ReactionPDE, lambda x, t: pde.exact(x, t), rho=5.0)).max() < 1e-4


def test_allen_cahn_residual_formula():
    # u = x^2 t:  u_t - eps u_xx + kappa (u^3 - u) = x^2 - 2 eps t + kappa((x^2 t)^3 - x^2 t)
    m = _Exact(lambda x, t: x ** 2 * t)
    p = m.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
    xt = jax.random.uniform(jax.random.PRNGKey(3), (20, 2), minval=-1, maxval=1)
    x, t = xt[:, 0], xt[:, 1]
    expect = x ** 2 - 2e-4 * t + 5.0 * ((x ** 2 * t) ** 3 - x ** 2 * t)
    np.testing.assert_allclose(np.asarray(AllenCahnPDE(m).residual(p, xt)), np.asarray(expect),
                               rtol=1e-5, atol=1e-6)


def test_pinnloss_periodic_and_velocity_terms():
    # u = sin(x) cos(t): 2π-periodic in x, u_t(x, 0) = 0
    m = _Exact(lambda x, t: jnp.sin(x) * jnp.cos(t))
    p = m.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
    pde = ConvectionPDE(m, beta=1.0)
    x_i = jnp.linspace(0, 6, 9)
    args = (jnp.ones(4), jnp.ones(4), x_i, jnp.sin(x_i),
            jnp.zeros(5), jnp.linspace(0, 1, 5), jnp.full(5, 123.0))   # u_b ignored when periodic
    _, (_, ic, bc, _) = PINNLoss(m, pde, ic_velocity_weight=1.0, periodic_length=2 * np.pi)(p, *args)
    assert float(ic) < 1e-10 and float(bc) < 1e-10
    # a non-periodic period makes the BC term non-zero; an evolving IC velocity likewise
    _, (_, _, bc2, _) = PINNLoss(m, pde, periodic_length=1.0)(p, *args)
    m2 = _Exact(lambda x, t: jnp.sin(x) * jnp.cos(t) + t)
    p2 = m2.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
    _, (_, ic2, _, _) = PINNLoss(m2, pde, ic_velocity_weight=1.0)(p2, *args)
    assert float(bc2) > 1e-3 and abs(float(ic2) - 1.0) < 1e-5
    # defaults leave the original Dirichlet behaviour untouched
    _, (_, _, bc3, _) = PINNLoss(m, pde)(p, *args)
    assert float(bc3) > 1.0
