"""Tests for RAR-D resampling — both packed and split interfaces.

Covers:
* Shape preservation (output == input shape).
* Bootstrap and custom-sampler modes.
* Correct packed-API call to pde.residual(params, xy).
* Split shim packs/unpacks correctly for 1-D and 2-D spatial arrays.
* High-residual points get higher sampling probability.
* rar_d_resample_split produces same result as manually calling rar_d_resample.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import pytest

from underPINN.training.resample import rar_d_resample, rar_d_resample_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ConstantPDE:
    """Fake PDE that returns a user-supplied residual value at every point."""
    def __init__(self, value=1.0):
        self._value = value

    def residual(self, params, xy):
        return jnp.full(xy.shape[0], self._value)


class _SpikedPDE:
    """PDE whose residual is large only near a specific x-coordinate."""
    def __init__(self, spike_x=0.5, spike_val=100.0, base_val=1e-4):
        self._spike_x  = spike_x
        self._spike_val = spike_val
        self._base_val  = base_val

    def residual(self, params, xy):
        # xy[:, 0] is the first coordinate; spike near spike_x
        dist = jnp.abs(xy[:, 0] - self._spike_x)
        return jnp.where(dist < 0.05, self._spike_val, self._base_val)


# ---------------------------------------------------------------------------
# Packed interface — rar_d_resample
# ---------------------------------------------------------------------------

class TestRarDResamplePacked:
    def setup_method(self):
        self.key = jax.random.PRNGKey(0)
        self.pde = _ConstantPDE(1.0)

    def _xy(self, N=20, D=2):
        return jax.random.uniform(jax.random.PRNGKey(1), (N, D))

    def test_output_shape_preserved(self):
        xy = self._xy(20, 2)
        xy_new = rar_d_resample(self.pde, None, xy, key=self.key)
        assert xy_new.shape == xy.shape

    def test_output_shape_1d_time_only(self):
        xy = self._xy(15, 1)
        xy_new = rar_d_resample(self.pde, None, xy, key=self.key)
        assert xy_new.shape == xy.shape

    def test_output_shape_3d_coords(self):
        xy = self._xy(10, 3)
        xy_new = rar_d_resample(self.pde, None, xy, key=self.key)
        assert xy_new.shape == xy.shape

    def test_bootstrap_mode_stays_in_domain(self):
        """Bootstrap points must be drawn from existing set."""
        xy = self._xy(30, 2)
        xy_new = rar_d_resample(self.pde, None, xy, key=self.key, n_candidates=100)
        # All outputs should be exact copies of rows from xy (bootstrap resamples)
        # Check that each new row is present in the original set
        # Note: may have duplicates — that's fine for bootstrap
        for i in range(xy_new.shape[0]):
            assert jnp.any(jnp.all(xy_new[i] == xy, axis=1))

    def test_custom_sampler_called(self):
        call_log = []
        def sampler(n, key):
            call_log.append(n)
            return jax.random.uniform(key, (n, 2))

        xy = self._xy(10, 2)
        rar_d_resample(self.pde, None, xy, candidate_sampler=sampler, key=self.key)
        assert len(call_log) == 1

    def test_custom_sampler_n_candidates(self):
        called_n = []
        def sampler(n, key):
            called_n.append(n)
            return jax.random.uniform(key, (n, 2))

        xy = self._xy(10, 2)
        rar_d_resample(self.pde, None, xy, n_candidates=999,
                       candidate_sampler=sampler, key=self.key)
        assert called_n[0] == 999

    def test_high_residual_region_sampled_more(self):
        """Points near the spike should appear more often in the resampled set."""
        spike_pde = _SpikedPDE(spike_x=0.5, spike_val=1000.0, base_val=1e-6)

        rng = jax.random.PRNGKey(77)
        # Candidate pool: uniform in [0,1]
        def sampler(n, key):
            return jax.random.uniform(key, (n, 2))

        xy = jax.random.uniform(rng, (200, 2))
        xy_new = rar_d_resample(spike_pde, None, xy,
                                n_candidates=5000, candidate_sampler=sampler,
                                key=rng, k=2.0)

        # Expect many points near x=0.5 (spike)
        near_spike = jnp.sum(jnp.abs(xy_new[:, 0] - 0.5) < 0.1)
        assert int(near_spike) > 20, f"Expected many points near spike, got {int(near_spike)}"

    def test_pde_residual_called_with_packed_array(self):
        """residual must receive an (N, D) packed array, not separate x,t."""
        call_log = []

        class _RecordingPDE:
            def residual(self, params, xy):
                call_log.append(xy.shape)
                return jnp.ones(xy.shape[0])

        xy = self._xy(12, 3)
        rar_d_resample(_RecordingPDE(), None, xy, key=self.key)
        assert len(call_log) == 1
        assert len(call_log[0]) == 2   # shape is (n_candidates, 3)
        assert call_log[0][1] == 3     # D is preserved


# ---------------------------------------------------------------------------
# Split interface — rar_d_resample_split
# ---------------------------------------------------------------------------

class TestRarDResampleSplit:
    def setup_method(self):
        self.key = jax.random.PRNGKey(5)
        self.pde = _ConstantPDE(1.0)
        rng = jax.random.PRNGKey(11)
        k1, k2 = jax.random.split(rng)
        self.x_r = jax.random.uniform(k1, (20,))
        self.t_r = jax.random.uniform(k2, (20,))

    def test_output_shapes_preserved_1d(self):
        x_new, t_new = rar_d_resample_split(
            self.pde, None, self.x_r, self.t_r, key=self.key)
        assert x_new.shape == self.x_r.shape
        assert t_new.shape == self.t_r.shape

    def test_output_shapes_preserved_2d_spatial(self):
        x_2d = jax.random.uniform(jax.random.PRNGKey(3), (20, 2))
        t_1d = jax.random.uniform(jax.random.PRNGKey(4), (20,))
        x_new, t_new = rar_d_resample_split(
            self.pde, None, x_2d, t_1d, key=self.key)
        assert x_new.shape == x_2d.shape
        assert t_new.shape == t_1d.shape

    def test_split_sampler_wrapping(self):
        """candidate_sampler returning (x, t) pair should work via the shim."""
        call_count = [0]
        def pair_sampler(n, key):
            call_count[0] += 1
            k1, k2 = jax.random.split(key)
            return jax.random.uniform(k1, (n,)), jax.random.uniform(k2, (n,))

        x_new, t_new = rar_d_resample_split(
            self.pde, None, self.x_r, self.t_r,
            candidate_sampler=pair_sampler, key=self.key)
        assert call_count[0] == 1
        assert x_new.shape == self.x_r.shape

    def test_equivalent_to_packed_with_manual_pack(self):
        """Split interface must give same result as manually packed interface."""
        # Use same key and no custom sampler (bootstrap)
        x_new, t_new = rar_d_resample_split(
            self.pde, None, self.x_r, self.t_r, k=1.5, key=self.key)

        # Manually pack and call packed interface
        xy = jnp.stack([self.x_r, self.t_r], axis=1)
        xy_new = rar_d_resample(self.pde, None, xy, k=1.5, key=self.key)

        assert jnp.allclose(x_new, xy_new[:, 0], atol=1e-6)
        assert jnp.allclose(t_new, xy_new[:, 1], atol=1e-6)


# ---------------------------------------------------------------------------
# Multi-output PDE (residual returns (N, K))
# ---------------------------------------------------------------------------

class TestMultiOutputResiudal:
    def test_vector_residual_uses_l2_norm(self):
        """For a multi-component residual the weighting uses the L2 magnitude."""
        class _VectorPDE:
            def residual(self, params, xy):
                # large residual near x=0, small elsewhere
                big = (xy[:, 0] < 0.1).astype(jnp.float32) * 100.0
                return jnp.stack([big, jnp.zeros_like(big)], axis=1)  # (N, 2)

        rng = jax.random.PRNGKey(99)
        xy = jax.random.uniform(rng, (100, 2))
        xy_new = rar_d_resample(_VectorPDE(), None, xy,
                                n_candidates=1000, key=rng, k=1.0)
        # Most resampled points should be near x=0
        near_zero = jnp.sum(xy_new[:, 0] < 0.2)
        assert int(near_zero) > 20
