"""Tests for MLP and FourierMLP neural-network modules.

Verifies shape correctness, parameter initialisation, and that the models
are compatible with JAX transformations (jit, grad, vmap).
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import pytest

from underPINN.nn.mlp import MLP, FourierMLP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init(model, in_shape, seed=0):
    key = jax.random.PRNGKey(seed)
    x = jnp.ones(in_shape)
    params = model.init(key, x)
    return params


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class TestMLP:
    def test_output_shape_1d_out(self):
        model = MLP(layers=[2, 16, 16, 1])
        params = _init(model, (1, 2))
        out = model.apply(params, jnp.ones((5, 2)))
        assert out.shape == (5, 1)

    def test_output_shape_multiout(self):
        model = MLP(layers=[3, 32, 4])
        params = _init(model, (1, 3))
        out = model.apply(params, jnp.ones((8, 3)))
        assert out.shape == (8, 4)

    def test_single_hidden_layer(self):
        model = MLP(layers=[1, 8, 1])
        params = _init(model, (1, 1))
        out = model.apply(params, jnp.ones((3, 1)))
        assert out.shape == (3, 1)

    def test_deep_network(self):
        model = MLP(layers=[2, 64, 64, 64, 64, 1])
        params = _init(model, (1, 2))
        out = model.apply(params, jnp.ones((10, 2)))
        assert out.shape == (10, 1)

    def test_params_not_none(self):
        model = MLP(layers=[2, 8, 1])
        params = _init(model, (1, 2))
        leaves = jax.tree_util.tree_leaves(params)
        assert len(leaves) > 0

    def test_different_seeds_give_different_params(self):
        model = MLP(layers=[2, 8, 1])
        p1 = _init(model, (1, 2), seed=0)
        p2 = _init(model, (1, 2), seed=1)
        # Biases are initialised to zero by default — compare kernel weights instead
        kernels_1 = [l for l in jax.tree_util.tree_leaves(p1) if l.ndim > 1]
        kernels_2 = [l for l in jax.tree_util.tree_leaves(p2) if l.ndim > 1]
        assert len(kernels_1) > 0
        assert not jnp.allclose(kernels_1[0], kernels_2[0])

    def test_jit_compatible(self):
        model = MLP(layers=[2, 8, 1])
        params = _init(model, (1, 2))

        @jax.jit
        def forward(p, x):
            return model.apply(p, x)

        out = forward(params, jnp.ones((4, 2)))
        assert out.shape == (4, 1)

    def test_grad_compatible(self):
        model = MLP(layers=[2, 8, 1])
        params = _init(model, (1, 2))

        def loss(p):
            return jnp.mean(model.apply(p, jnp.ones((4, 2))) ** 2)

        grads = jax.grad(loss)(params)
        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0

    def test_output_changes_with_params(self):
        model = MLP(layers=[2, 8, 1])
        p1 = _init(model, (1, 2), seed=0)
        p2 = _init(model, (1, 2), seed=1)
        x = jnp.ones((3, 2))
        assert not jnp.allclose(model.apply(p1, x), model.apply(p2, x))

    def test_batch_independence(self):
        """Output for a single point should match when batched."""
        model = MLP(layers=[2, 8, 1])
        params = _init(model, (1, 2))
        x_single = jnp.array([[0.5, 0.3]])
        x_batch  = jnp.array([[0.5, 0.3], [0.1, 0.2]])
        out_single = model.apply(params, x_single)
        out_batch  = model.apply(params, x_batch)
        assert jnp.allclose(out_single[0], out_batch[0], atol=1e-6)


# ---------------------------------------------------------------------------
# FourierMLP
# ---------------------------------------------------------------------------

class TestFourierMLP:
    def test_output_shape(self):
        model = FourierMLP(layers=[2, 32, 1], n_fourier=8)
        params = _init(model, (1, 2))
        out = model.apply(params, jnp.ones((6, 2)))
        assert out.shape == (6, 1)

    def test_fourier_b_in_params(self):
        model = FourierMLP(layers=[2, 16, 1], n_fourier=4)
        params = _init(model, (1, 2))
        # Check that fourier_B exists somewhere in the param tree
        flat = str(jax.tree_util.tree_map(lambda x: x.shape, params))
        assert "fourier_B" in flat

    def test_jit_compatible(self):
        model = FourierMLP(layers=[2, 16, 1], n_fourier=4)
        params = _init(model, (1, 2))

        @jax.jit
        def fwd(p, x):
            return model.apply(p, x)

        out = fwd(params, jnp.ones((3, 2)))
        assert out.shape == (3, 1)

    def test_grad_compatible(self):
        model = FourierMLP(layers=[2, 16, 1], n_fourier=4)
        params = _init(model, (1, 2))

        def loss(p):
            return jnp.mean(model.apply(p, jnp.ones((4, 2))) ** 2)

        grads = jax.grad(loss)(params)
        assert len(jax.tree_util.tree_leaves(grads)) > 0
