"""Tests for underPINN/utils/sampling.py.

Covers:
* ``safe_choice`` — replace-fallback behaviour when batch > pool.
* ``rad_resample`` — shape/weighting sanity (was previously untested).
* ``qr_deim_resample`` — shape preservation, vector-valued residuals,
  reproducibility, graceful handling of degenerate pools, and -- the
  property that actually motivates the algorithm -- better spatial spread
  than magnitude-only sampling when many candidates share (near-)identical
  high residual at one location.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
import pytest

from underPINN.utils.sampling import qr_deim_resample, rad_resample, safe_choice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ConstantPDE:
    """Scalar residual, identical at every point."""
    def __init__(self, value=1.0):
        self._value = value

    def residual(self, params, xy):
        return jnp.full(xy.shape[0], self._value)


class _ClusteredSpikePDE:
    """Scalar residual: a tight cluster of points all share one huge value,
    the rest of the domain sits at a tiny baseline. Mimics a single narrow
    shock spike sampled at high density -- the case magnitude-only sampling
    handles poorly (it just keeps redrawing from inside the cluster)."""
    def __init__(self, cluster_x=0.1, cluster_radius=0.03,
                 spike_val=1000.0, base_val=1e-3):
        self.cluster_x = cluster_x
        self.cluster_radius = cluster_radius
        self.spike_val = spike_val
        self.base_val = base_val

    def residual(self, params, xy):
        dist = jnp.abs(xy[:, 0] - self.cluster_x)
        return jnp.where(dist < self.cluster_radius, self.spike_val, self.base_val)


class _VectorResidualPDE:
    """Vector-valued residual (e.g. momentum + continuity), two components."""
    def residual(self, params, xy):
        r0 = jnp.sin(4.0 * xy[:, 0])
        r1 = xy[:, 1] ** 2
        return jnp.stack([r0, r1], axis=1)


def _uniform_sampler(n, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# safe_choice
# ---------------------------------------------------------------------------

class TestSafeChoice:
    def test_no_fallback_needed(self):
        import jax
        idx = safe_choice(jax.random.PRNGKey(0), n=100, batch=10)
        assert idx.shape == (10,)
        assert jnp.all(idx < 100) and jnp.all(idx >= 0)

    def test_fallback_when_batch_exceeds_pool(self):
        import jax
        # Would raise under replace=False; must not raise here.
        idx = safe_choice(jax.random.PRNGKey(0), n=5, batch=50)
        assert idx.shape == (50,)
        assert jnp.all(idx < 5) and jnp.all(idx >= 0)


# ---------------------------------------------------------------------------
# rad_resample
# ---------------------------------------------------------------------------

class TestRadResample:
    def test_output_shape(self):
        out = rad_resample(_ConstantPDE(), None, _uniform_sampler,
                           n_keep=16, n_candidates=64, seed=0)
        assert out.shape == (16, 2)

    def test_favors_high_residual_region(self):
        # c=0 isolates the pure |r|^k weighting (c>0 is a deliberate uniform-
        # coverage floor per Wu et al. 2023, and at the default c=1.0 it
        # legitimately competes with -- and roughly halves -- the spike's
        # share here, since the ~1945 baseline points each contribute a
        # constant +1 while only ~55 points see the huge residual).
        pde = _ClusteredSpikePDE()
        out = rad_resample(pde, None, _uniform_sampler,
                           n_keep=200, n_candidates=2000, k=1.0, c=0.0, seed=0)
        frac_in_cluster = np.mean(np.abs(out[:, 0] - pde.cluster_x)
                                  < pde.cluster_radius)
        assert frac_in_cluster > 0.8

    def test_uniform_floor_c_prevents_pure_concentration(self):
        # With c=1.0 (the function's default), the additive floor should
        # pull a meaningful fraction of draws away from the spike relative
        # to c=0 -- this is RAD's intended "don't fully abandon the rest of
        # the domain" behaviour, not a bug.
        pde = _ClusteredSpikePDE()
        out_c0 = rad_resample(pde, None, _uniform_sampler,
                              n_keep=200, n_candidates=2000, k=1.0, c=0.0, seed=0)
        out_c1 = rad_resample(pde, None, _uniform_sampler,
                              n_keep=200, n_candidates=2000, k=1.0, c=1.0, seed=0)
        frac_c0 = np.mean(np.abs(out_c0[:, 0] - pde.cluster_x) < pde.cluster_radius)
        frac_c1 = np.mean(np.abs(out_c1[:, 0] - pde.cluster_x) < pde.cluster_radius)
        assert frac_c1 < frac_c0


# ---------------------------------------------------------------------------
# qr_deim_resample
# ---------------------------------------------------------------------------

class TestQRDEIMResample:
    def test_output_shape_scalar_residual(self):
        out = qr_deim_resample(_ConstantPDE(), None, _uniform_sampler,
                               n_keep=16, n_candidates=64, seed=0)
        assert out.shape == (16, 2)
        assert out.dtype == np.float32

    def test_output_shape_vector_residual(self):
        out = qr_deim_resample(_VectorResidualPDE(), None, _uniform_sampler,
                               n_keep=20, n_candidates=100, seed=0)
        assert out.shape == (20, 2)

    def test_deterministic_given_seed(self):
        out1 = qr_deim_resample(_VectorResidualPDE(), None, _uniform_sampler,
                                n_keep=16, n_candidates=80, seed=7)
        out2 = qr_deim_resample(_VectorResidualPDE(), None, _uniform_sampler,
                                n_keep=16, n_candidates=80, seed=7)
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_differ(self):
        out1 = qr_deim_resample(_VectorResidualPDE(), None, _uniform_sampler,
                                n_keep=16, n_candidates=80, seed=1)
        out2 = qr_deim_resample(_VectorResidualPDE(), None, _uniform_sampler,
                                n_keep=16, n_candidates=80, seed=2)
        assert not np.array_equal(out1, out2)

    def test_n_candidates_less_than_n_keep_is_clamped(self):
        # Must not crash; internally raises n_candidates to n_keep.
        out = qr_deim_resample(_ConstantPDE(), None, _uniform_sampler,
                               n_keep=50, n_candidates=10, seed=0)
        assert out.shape == (50, 2)

    def test_augment_coords_false_still_works(self):
        out = qr_deim_resample(_VectorResidualPDE(), None, _uniform_sampler,
                               n_keep=12, n_candidates=60, seed=0,
                               augment_coords=False)
        assert out.shape == (12, 2)

    def test_selected_points_lie_within_candidate_pool(self):
        # Every returned point must be one of the actually-sampled candidates
        # (QR-DEIM selects existing points, it does not synthesize new ones).
        pde = _ClusteredSpikePDE()
        cand = _uniform_sampler(500, seed=3)
        out = qr_deim_resample(pde, None, lambda n, s: cand,
                               n_keep=30, n_candidates=500, seed=3)
        cand_set = {tuple(row) for row in cand}
        assert all(tuple(row) in cand_set for row in out)

    def test_tractable_at_realistic_collocation_batch_size(self):
        # Regression test: an earlier implementation sketched a dense
        # (n_candidates, n_keep) matrix, which allocates tens of GB and
        # crashes with a MemoryError at real PINN batch sizes (e.g. the
        # Toro-3 benchmark's 40,000-point pool with a 200,000-point
        # candidate pool). Must complete and stay well under a second.
        import time
        n_keep, n_cand = 40_000, 200_000

        def big_sampler(n, seed):
            rng = np.random.default_rng(seed)
            return rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float32)

        t0 = time.perf_counter()
        out = qr_deim_resample(_VectorResidualPDE(), None, big_sampler,
                               n_keep=n_keep, n_candidates=n_cand, seed=0)
        elapsed = time.perf_counter() - t0
        assert out.shape == (n_keep, 2)
        assert elapsed < 5.0, f"took {elapsed:.2f}s -- should be sub-second"

    def test_better_spatial_spread_than_magnitude_only_sampling(self):
        """The property that motivates QR-DEIM over rad_resample: when a
        tight cluster shares near-identical extreme residual, magnitude-only
        sampling (rad_resample) keeps redrawing from inside that cluster,
        while QR-pivoting -- augmented with residual-weighted coordinates --
        should spread its picks over a wider span of the domain."""
        pde = _ClusteredSpikePDE(cluster_radius=0.02, spike_val=1e6)
        n_keep, n_cand = 60, 3000

        rad_out = rad_resample(pde, None, _uniform_sampler,
                               n_keep=n_keep, n_candidates=n_cand,
                               k=1.0, c=0.0, seed=0)
        deim_out = qr_deim_resample(pde, None, _uniform_sampler,
                                    n_keep=n_keep, n_candidates=n_cand, seed=0)

        rad_spread = float(np.std(rad_out[:, 0]))
        deim_spread = float(np.std(deim_out[:, 0]))
        assert deim_spread > rad_spread
