"""underPINN's own JAX PINNs on the identical wave / heat / helmholtz / ode
setups used by ``multi/{wave1d,heat2d,helmholtz2d,ode_harmonic}.py``
(NVIDIA PhysicsNeMo Sym) -- same physics, same network depth/width, same
per-step minibatch sizes (matching each problem's real example config
exactly), scored against the identical exact/manufactured solution each
PhysicsNeMo case uses.

Each function below is a compact, self-contained reimplementation of the
corresponding real example (``examples/wave/wave.py``,
``examples/heat/forward.py``, ``examples/helmholtz/helmholtz.py``,
``examples/ode/ode_test.py``'s harmonic-oscillator case) -- same
architecture/PDE classes reused directly, but a minimal manual training
loop in place of each example's full Solver/callback/restart machinery, so
wall-clock is measured on the step loop alone and trained params are
available directly for evaluation on the same grid PhysicsNeMo's case uses
(the same approach ``compare_underpinn.py`` already takes for Burgers).

Run:
    python benchmarks/suite/physicsnemo/compare_underpinn_multi.py
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
import numpy as np                                            # noqa: E402
import optax                                                  # noqa: E402

from common import base_parser, jax_device_info, save_result, timed, warn_if_cpu  # noqa: E402

from underPINN.nn.mlp import FourierMLP, MLP                  # noqa: E402
from underPINN.pde.diffusion import DiffusionPDE               # noqa: E402
from underPINN.pde.heat import SteadyHeatPDE                  # noqa: E402
from underPINN.pde.heat2d_unsteady import UnsteadyHeat2DPDE    # noqa: E402
from underPINN.pde.helmholtz import HelmholtzPDE              # noqa: E402
from underPINN.pde.ode import HarmonicOscillatorODE           # noqa: E402
from underPINN.pde.wave import WavePDE                        # noqa: E402
from underPINN.utils.sampling import (safe_choice, qr_deim_resample,  # noqa: E402
                                      rad_resample)


def _timed_train(step, params0, state0, epochs, seed_key):
    _p, _s, _k, warm = step(params0, state0, seed_key)
    jax.block_until_ready(warm)
    def train():
        p, s, k = params0, state0, seed_key
        loss = None
        for _ in range(epochs):
            p, s, k, loss = step(p, s, k)
        jax.block_until_ready(loss)
        return p, float(loss)
    (final_params, final_loss), wall = timed(train)
    return final_params, final_loss, wall


_RESAMPLERS = {"qr_deim": qr_deim_resample, "rad": rad_resample}


def _timed_train_adaptive(step, params0, state0, epochs, seed_key, pool0,
                          pde, domain_sampler, n_keep, sampling,
                          resample_period, seed, candidate_mult: int = 5,
                          adaptive_frac: float = 1.0):
    """Like ``_timed_train``, but the interior collocation pool (``pool0``,
    one ``(N, d)`` or ``(N,)`` array) is periodically replaced with a fresh
    QR-DEIM-R or RAD draw based on the *live* PDE residual, every
    ``resample_period`` epochs -- the same "uniform vs. adaptive collocation"
    axis as ``ablations/ablate_qr_deim_ramp_ns.py``, applied here to the
    PhysicsNeMo comparison problems. ``sampling="uniform"`` is a no-op (the
    pool drawn once at problem setup is reused for the whole run, exactly as
    the plain comparison already did).

    ``adaptive_frac`` controls how much of the pool is ever touched:
    ``1.0`` (the original behaviour here) replaces the *entire* pool every
    cycle. ``ablations/ablate_qr_deim_ramp_ns.py`` instead keeps a majority
    of its pool permanently fixed (``xy_uniform + xy_bl``, for stability)
    and only resamples a smaller fraction (``xy_adapt``, for local
    refinement) -- ``adaptive_frac < 1.0`` reproduces that split here: the
    first ``n_keep * (1 - adaptive_frac)`` points of ``pool0`` are frozen
    for the whole run (the "uniform, for stability" portion), and only the
    remaining ``n_keep * adaptive_frac`` are ever re-drawn (the "adaptive,
    for improvement" portion) -- concatenated back together into the
    same-shape pool ``step`` sees, so no retracing happens either way.

    ``step``'s signature is ``step(p, s, key, pool) -> (p, s, key, loss)``,
    with any IC/BC arrays the problem also needs closed over unchanged (they
    are never resampled). Passing the interior pool as an explicit,
    fixed-shape argument -- rather than a Python closure baked into the
    jaxpr at trace time -- means ``step`` is compiled exactly once: a
    resample only ever changes the pool's *values*, never its shape/dtype,
    so no retracing happens on any of the ``epochs // resample_period``
    resample events.
    """
    _p, _s, _k, warm = step(params0, state0, seed_key, pool0)
    jax.block_until_ready(warm)
    resample_fn = _RESAMPLERS.get(sampling)

    if resample_fn is not None:
        n_adapt = max(1, round(adaptive_frac * n_keep))
        n_fixed = n_keep - n_adapt
    else:
        n_adapt, n_fixed = 0, n_keep
    fixed_pool = pool0[:n_fixed] if n_fixed > 0 else None

    def train():
        p, s, key, pool = params0, state0, seed_key, pool0
        loss = None
        for ep in range(epochs):
            if resample_fn is not None and ep > 0 and ep % resample_period == 0:
                kwargs = {"k": 1.0, "c": 1.0} if sampling == "rad" else {}
                new_pts = resample_fn(pde, p, domain_sampler, n_keep=n_adapt,
                                      n_candidates=candidate_mult * n_adapt,
                                      seed=seed + ep, **kwargs)
                adapt_pool = jnp.asarray(new_pts)
                pool = (jnp.concatenate([fixed_pool, adapt_pool], axis=0)
                       if fixed_pool is not None else adapt_pool)
            p, s, key, loss = step(p, s, key, pool)
        jax.block_until_ready(loss)
        return p, float(loss)
    (final_params, final_loss), wall = timed(train)
    return final_params, final_loss, wall


# ═══════════════════════════════════════════════════════════════════════════
# Wave: u_tt = c^2 u_xx, matches examples/wave/wave.py + config.yaml
# ═══════════════════════════════════════════════════════════════════════════

def run_wave(epochs: int, seed: int, sampling: str = "uniform",
            resample_period: int = 500,
                    adaptive_frac: float = 1.0) -> dict:
    C, T_MAX = 1.0, 2.0
    N_R, N_IC, N_BC = 10000, 300, 300
    BATCH_R, BATCH_I, BATCH_B = 2048, 256, 256
    IC_W, IC_DOT_W, BC_W = 100.0, 100.0, 10.0
    LAYERS = [2, 64, 64, 64, 1]
    LR = 1e-3

    def domain_sampler(n, sd):
        r = np.random.default_rng(sd)
        return np.stack([r.uniform(-1, 1, n), r.uniform(0, T_MAX, n)],
                        axis=1).astype(np.float32)

    rng = np.random.default_rng(seed)
    xy_r0 = jnp.array(np.stack([rng.uniform(-1, 1, N_R),
                                rng.uniform(0, T_MAX, N_R)],
                               axis=1).astype(np.float32))
    x_ic = jnp.array(np.linspace(-1, 1, N_IC, dtype=np.float32))
    u_ic = jnp.array(np.sin(np.pi * np.linspace(-1, 1, N_IC)).astype(np.float32))
    t_bc = rng.uniform(0, T_MAX, N_BC).astype(np.float32)
    x_bc = jnp.array(np.concatenate([np.full(N_BC, -1., np.float32),
                                     np.full(N_BC, 1., np.float32)]))
    t_bc = jnp.array(np.concatenate([t_bc, t_bc]))

    model = FourierMLP(layers=LAYERS, n_fourier=16, sigma=max(2.0, C * np.pi))
    pde = WavePDE(model, c=C)
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 2)))

    def loss_fn(p, xy_ri, xi, ui, xb, tb):
        res = pde.residual(p, xy_ri)
        pde_l = jnp.mean(res ** 2)
        u_pred_ic = pde.u(p, xi, jnp.zeros_like(xi))
        ic_l = jnp.mean((u_pred_ic - ui) ** 2)
        ic_dot_l = jnp.mean(pde.u_t(p, xi, jnp.zeros_like(xi)) ** 2)
        bc_l = jnp.mean(pde.u(p, xb, tb) ** 2)
        return pde_l + IC_W * ic_l + IC_DOT_W * ic_dot_l + BC_W * bc_l

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=0.01)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))

    @jax.jit
    def step(p, s, key, xy_r_pool):
        key, k1, k2 = jax.random.split(key, 3)
        ir = safe_choice(k1, xy_r_pool.shape[0], BATCH_R)
        ii = safe_choice(k2, N_IC, BATCH_I)
        loss, g = jax.value_and_grad(loss_fn)(
            p, xy_r_pool[ir], x_ic[ii], u_ic[ii], x_bc, t_bc)
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, key, loss

    final_params, final_loss, wall = _timed_train_adaptive(
        step, params0, opt.init(params0), epochs, jax.random.PRNGKey(seed + 1),
        xy_r0, pde, domain_sampler, N_R, sampling, resample_period, seed, adaptive_frac=adaptive_frac)

    Nx_eval, Nt_eval = 101, 41
    x_eval = np.linspace(-1.0, 1.0, Nx_eval)
    t_eval = np.linspace(0.0, T_MAX, Nt_eval)
    XX, TT = np.meshgrid(x_eval, t_eval, indexing="ij")
    u_exact = np.sin(np.pi * XX) * np.cos(C * np.pi * TT)
    u_pred = np.array(pde.u(final_params, jnp.array(XX.ravel(), "f4"),
                            jnp.array(TT.ravel(), "f4"))).reshape(Nx_eval, Nt_eval)
    rel_l2 = float(np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-10))
    return {"problem": "wave_1d", "epochs": epochs, "wall_s": wall,
           "ms_per_epoch": 1e3 * wall / epochs, "final_loss": final_loss,
           "rel_l2": rel_l2, "sampling": sampling, "adaptive_frac": adaptive_frac}


# ═══════════════════════════════════════════════════════════════════════════
# Heat / 2-D Poisson, matches examples/heat/forward.py + heat_forward.yaml
# ═══════════════════════════════════════════════════════════════════════════

def run_heat(epochs: int, seed: int, sampling: str = "uniform",
            resample_period: int = 500,
                    adaptive_frac: float = 1.0) -> dict:
    N_R, N_BC = 5000, 300     # per-edge
    BATCH_R, BATCH_B = 2048, 256
    BC_W = 100.0
    LAYERS = [2, 64, 64, 64, 1]
    LR = 1e-3

    def source(x, y):
        return 2.0 * jnp.pi ** 2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

    def domain_sampler(n, sd):
        r = np.random.default_rng(sd)
        return r.uniform(0.0, 1.0, (n, 2)).astype(np.float32)

    rng = np.random.default_rng(seed)
    xy_r0 = jnp.array(rng.uniform(0.0, 1.0, (N_R, 2)).astype(np.float32))
    t = np.linspace(0.0, 1.0, N_BC, dtype=np.float32)
    xy_b = jnp.array(np.concatenate([
        np.stack([t, np.zeros_like(t)], axis=1),
        np.stack([t, np.ones_like(t)], axis=1),
        np.stack([np.zeros_like(t), t], axis=1),
        np.stack([np.ones_like(t), t], axis=1),
    ]))
    N_b_total = xy_b.shape[0]

    model = MLP(layers=LAYERS)
    pde = SteadyHeatPDE(model, source_fn=source)
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 2)))

    def loss_fn(p, xy_ri, xy_bi):
        res = pde.residual(p, xy_ri)
        pde_l = jnp.mean(res ** 2)
        bc_l = jnp.mean(model.apply(p, xy_bi)[:, 0] ** 2)
        return pde_l + BC_W * bc_l

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=0.01)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))

    @jax.jit
    def step(p, s, key, xy_r_pool):
        key, k1, k2 = jax.random.split(key, 3)
        ir = safe_choice(k1, xy_r_pool.shape[0], BATCH_R)
        ib = safe_choice(k2, N_b_total, BATCH_B)
        loss, g = jax.value_and_grad(loss_fn)(p, xy_r_pool[ir], xy_b[ib])
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, key, loss

    final_params, final_loss, wall = _timed_train_adaptive(
        step, params0, opt.init(params0), epochs, jax.random.PRNGKey(seed + 1),
        xy_r0, pde, domain_sampler, N_R, sampling, resample_period, seed, adaptive_frac=adaptive_frac)

    N_eval = 101
    x_eval = np.linspace(0.0, 1.0, N_eval)
    y_eval = np.linspace(0.0, 1.0, N_eval)
    XX, YY = np.meshgrid(x_eval, y_eval, indexing="ij")
    u_exact = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    xy_eval = jnp.array(np.stack([XX.ravel(), YY.ravel()], axis=1), "f4")
    u_pred = np.array(model.apply(final_params, xy_eval)[:, 0]).reshape(N_eval, N_eval)
    rel_l2 = float(np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-10))
    return {"problem": "heat_2d", "epochs": epochs, "wall_s": wall,
           "ms_per_epoch": 1e3 * wall / epochs, "final_loss": final_loss,
           "sampling": sampling, "adaptive_frac": adaptive_frac,
           "rel_l2": rel_l2}


# ═══════════════════════════════════════════════════════════════════════════
# Helmholtz, matches examples/helmholtz/helmholtz.py + config.yaml
# ═══════════════════════════════════════════════════════════════════════════

def run_helmholtz(epochs: int, seed: int, sampling: str = "uniform",
                  resample_period: int = 500,
                    adaptive_frac: float = 1.0) -> dict:
    K = 4.0
    N_R, N_BC = 8000, 600     # per-edge
    BATCH_R, BATCH_B = 2048, 256
    BC_W = 100.0
    LAYERS = [2, 128, 128, 128, 1]
    LR = 1e-3

    def domain_sampler(n, sd):
        r = np.random.default_rng(sd)
        return r.uniform(0.0, 1.0, (n, 2)).astype(np.float32)

    rng = np.random.default_rng(seed)
    xy_r0 = jnp.array(rng.uniform(0.0, 1.0, (N_R, 2)).astype(np.float32))
    t = np.linspace(0.0, 1.0, N_BC, dtype=np.float32)
    xy_b = jnp.array(np.concatenate([
        np.stack([t, np.zeros_like(t)], axis=1),
        np.stack([t, np.ones_like(t)], axis=1),
        np.stack([np.zeros_like(t), t], axis=1),
        np.stack([np.ones_like(t), t], axis=1),
    ]))
    N_b_total = xy_b.shape[0]

    model = FourierMLP(layers=LAYERS, n_fourier=32, sigma=K)
    pde = HelmholtzPDE(model, k=K)
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 2)))

    def loss_fn(p, xy_ri, xy_bi):
        res = pde.residual(p, xy_ri)
        pde_l = jnp.mean(res ** 2)
        bc_l = jnp.mean(model.apply(p, xy_bi)[:, 0] ** 2)
        return pde_l + BC_W * bc_l

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=0.01)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))

    @jax.jit
    def step(p, s, key, xy_r_pool):
        key, k1, k2 = jax.random.split(key, 3)
        ir = safe_choice(k1, xy_r_pool.shape[0], BATCH_R)
        ib = safe_choice(k2, N_b_total, BATCH_B)
        loss, g = jax.value_and_grad(loss_fn)(p, xy_r_pool[ir], xy_b[ib])
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, key, loss

    final_params, final_loss, wall = _timed_train_adaptive(
        step, params0, opt.init(params0), epochs, jax.random.PRNGKey(seed + 1),
        xy_r0, pde, domain_sampler, N_R, sampling, resample_period, seed, adaptive_frac=adaptive_frac)

    N_eval = 101
    x_eval = np.linspace(0.0, 1.0, N_eval)
    y_eval = np.linspace(0.0, 1.0, N_eval)
    XX, YY = np.meshgrid(x_eval, y_eval, indexing="ij")
    u_exact = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    xy_eval = jnp.array(np.stack([XX.ravel(), YY.ravel()], axis=1), "f4")
    u_pred = np.array(model.apply(final_params, xy_eval)[:, 0]).reshape(N_eval, N_eval)
    rel_l2 = float(np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-10))
    return {"problem": "helmholtz_2d", "epochs": epochs, "wall_s": wall,
           "ms_per_epoch": 1e3 * wall / epochs, "final_loss": final_loss,
           "rel_l2": rel_l2, "sampling": sampling, "adaptive_frac": adaptive_frac}


# ═══════════════════════════════════════════════════════════════════════════
# ODE harmonic oscillator, matches examples/ode/ode_test.py + config.yaml
# ═══════════════════════════════════════════════════════════════════════════

def run_ode_harmonic(epochs: int, seed: int, sampling: str = "uniform",
                     resample_period: int = 500,
                    adaptive_frac: float = 1.0) -> dict:
    OMEGA, T_MAX, U0, V0 = 2.0, 5.0, 1.0, 0.0
    N_R = 10000
    BATCH_R = 4096   # TrainingConfig default, not overridden by ode/config.yaml
    IC_W, IC_DOT_W = 100.0, 100.0
    LAYERS = [1, 64, 64, 64, 1]
    LR = 1e-3

    def domain_sampler(n, sd):
        r = np.random.default_rng(sd)
        return r.uniform(0.0, T_MAX, n).astype(np.float32)

    rng = np.random.default_rng(seed)
    t_r0 = jnp.array(rng.uniform(0.0, T_MAX, N_R).astype(np.float32))
    t_ic = jnp.array([0.0])
    u_ic = jnp.array([U0])
    u_ic_dot = jnp.array([V0])

    model = MLP(layers=LAYERS)
    pde = HarmonicOscillatorODE(model, omega=OMEGA)
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 1)))

    def loss_fn(p, t_ri):
        r_pde = pde.residual(p, t_ri)
        pde_l = jnp.mean(r_pde ** 2)
        ic_l = jnp.mean((pde.u(p, t_ic) - u_ic) ** 2)
        ic_dot_l = jnp.mean((pde.ut(p, t_ic) - u_ic_dot) ** 2)
        return pde_l + IC_W * ic_l + IC_DOT_W * ic_dot_l

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=0.01)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))

    @jax.jit
    def step(p, s, key, t_r_pool):
        key, k1 = jax.random.split(key, 2)
        ir = safe_choice(k1, t_r_pool.shape[0], BATCH_R)
        loss, g = jax.value_and_grad(loss_fn)(p, t_r_pool[ir])
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, key, loss

    final_params, final_loss, wall = _timed_train_adaptive(
        step, params0, opt.init(params0), epochs, jax.random.PRNGKey(seed + 1),
        t_r0, pde, domain_sampler, N_R, sampling, resample_period, seed, adaptive_frac=adaptive_frac)

    t_eval = jnp.linspace(0.0, T_MAX, 2000)
    u_exact = U0 * np.cos(OMEGA * np.array(t_eval))
    u_pred = np.array(pde.u(final_params, t_eval))
    rel_l2 = float(np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-10))
    return {"problem": "ode_harmonic", "epochs": epochs, "wall_s": wall,
           "ms_per_epoch": 1e3 * wall / epochs, "final_loss": final_loss,
           "rel_l2": rel_l2, "sampling": sampling, "adaptive_frac": adaptive_frac}


# ═══════════════════════════════════════════════════════════════════════════
# Diffusion 1D: u_t = alpha*u_xx, matches underPINN.pde.diffusion.DiffusionPDE's
# own docstring's canonical test case (no pre-existing example script to
# match against, unlike wave/heat/helmholtz/ode above, so this config is
# built directly from that docstring rather than copied from elsewhere)
# ═══════════════════════════════════════════════════════════════════════════

def run_diffusion(epochs: int, seed: int, sampling: str = "uniform",
                  resample_period: int = 500,
                    adaptive_frac: float = 1.0) -> dict:
    ALPHA, T_MAX = 0.01, 1.0
    N_R, N_IC, N_BC = 5000, 300, 300
    BATCH_R, BATCH_I, BATCH_B = 2048, 256, 256
    IC_W, BC_W = 100.0, 100.0
    LAYERS = [2, 64, 64, 64, 1]
    LR = 1e-3

    def domain_sampler(n, sd):
        r = np.random.default_rng(sd)
        return np.stack([r.uniform(0, 1, n), r.uniform(0, T_MAX, n)],
                        axis=1).astype(np.float32)

    rng = np.random.default_rng(seed)
    xt_r0 = jnp.array(np.stack([rng.uniform(0, 1, N_R), rng.uniform(0, T_MAX, N_R)],
                               axis=1).astype(np.float32))
    x_ic = jnp.array(np.linspace(0, 1, N_IC, dtype=np.float32))
    u_ic = jnp.array(np.sin(np.pi * np.linspace(0, 1, N_IC)).astype(np.float32))
    t_bc = rng.uniform(0, T_MAX, N_BC).astype(np.float32)
    x_bc = jnp.array(np.concatenate([np.zeros(N_BC, np.float32), np.ones(N_BC, np.float32)]))
    t_bc = jnp.array(np.concatenate([t_bc, t_bc]))

    model = MLP(layers=LAYERS)
    pde = DiffusionPDE(model, alpha=ALPHA)
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 2)))

    def loss_fn(p, xt_ri, xi, ui, xb, tb):
        res = pde.residual(p, xt_ri)
        pde_l = jnp.mean(res ** 2)
        u_pred_ic = pde.u(p, xi, jnp.zeros_like(xi))
        ic_l = jnp.mean((u_pred_ic - ui) ** 2)
        bc_l = jnp.mean(pde.u(p, xb, tb) ** 2)
        return pde_l + IC_W * ic_l + BC_W * bc_l

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=0.01)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))

    @jax.jit
    def step(p, s, key, xt_r_pool):
        key, k1, k2 = jax.random.split(key, 3)
        ir = safe_choice(k1, xt_r_pool.shape[0], BATCH_R)
        ii = safe_choice(k2, N_IC, BATCH_I)
        loss, g = jax.value_and_grad(loss_fn)(
            p, xt_r_pool[ir], x_ic[ii], u_ic[ii], x_bc, t_bc)
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, key, loss

    final_params, final_loss, wall = _timed_train_adaptive(
        step, params0, opt.init(params0), epochs, jax.random.PRNGKey(seed + 1),
        xt_r0, pde, domain_sampler, N_R, sampling, resample_period, seed, adaptive_frac=adaptive_frac)

    Nx_eval, Nt_eval = 101, 41
    x_eval = np.linspace(0.0, 1.0, Nx_eval)
    t_eval = np.linspace(0.0, T_MAX, Nt_eval)
    XX, TT = np.meshgrid(x_eval, t_eval, indexing="ij")
    u_exact = np.sin(np.pi * XX) * np.exp(-ALPHA * np.pi ** 2 * TT)
    u_pred = np.array(pde.u(final_params, jnp.array(XX.ravel(), "f4"),
                            jnp.array(TT.ravel(), "f4"))).reshape(Nx_eval, Nt_eval)
    rel_l2 = float(np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-10))
    return {"problem": "diffusion_1d", "epochs": epochs, "wall_s": wall,
           "ms_per_epoch": 1e3 * wall / epochs, "final_loss": final_loss,
           "rel_l2": rel_l2, "sampling": sampling, "adaptive_frac": adaptive_frac}


# ═══════════════════════════════════════════════════════════════════════════
# Heat2D unsteady: u_t = alpha*(u_xx+u_yy), matches
# underPINN.pde.heat2d_unsteady.UnsteadyHeat2DPDE's own docstring's canonical
# test case; layers=[3,64,64,64,64,1] matches examples/transfer/
# heat2d_transfer.py's LAYERS constant (production-representative, though
# that script itself is a transfer-learning example, not a clean forward
# baseline, so the rest of this config is built fresh)
# ═══════════════════════════════════════════════════════════════════════════

def run_heat2d_unsteady(epochs: int, seed: int, sampling: str = "uniform",
                        resample_period: int = 500,
                    adaptive_frac: float = 1.0) -> dict:
    ALPHA, T_MAX = 0.01, 1.0
    N_R, N_IC, N_BC = 8000, 400, 300   # N_BC per edge -> 4*N_BC total
    BATCH_R, BATCH_I, BATCH_B = 2048, 256, 256
    IC_W, BC_W = 100.0, 100.0
    LAYERS = [3, 64, 64, 64, 64, 1]
    LR = 1e-3

    def domain_sampler(n, sd):
        r = np.random.default_rng(sd)
        return np.stack([r.uniform(0, 1, n), r.uniform(0, 1, n), r.uniform(0, T_MAX, n)],
                        axis=1).astype(np.float32)

    rng = np.random.default_rng(seed)
    xyt_r0 = jnp.array(np.stack([rng.uniform(0, 1, N_R), rng.uniform(0, 1, N_R),
                                 rng.uniform(0, T_MAX, N_R)], axis=1).astype(np.float32))
    x_ic = rng.uniform(0, 1, N_IC).astype(np.float32)
    y_ic = rng.uniform(0, 1, N_IC).astype(np.float32)
    xy_ic = jnp.array(np.stack([x_ic, y_ic], axis=1))
    u_ic = jnp.array((np.sin(np.pi * x_ic) * np.sin(np.pi * y_ic)).astype(np.float32))

    t_bc = rng.uniform(0, T_MAX, N_BC).astype(np.float32)
    s_bc = rng.uniform(0, 1, N_BC).astype(np.float32)
    xy_bc = jnp.array(np.concatenate([
        np.stack([s_bc, np.zeros_like(s_bc)], axis=1),
        np.stack([s_bc, np.ones_like(s_bc)], axis=1),
        np.stack([np.zeros_like(s_bc), s_bc], axis=1),
        np.stack([np.ones_like(s_bc), s_bc], axis=1),
    ]))
    t_bc4 = jnp.array(np.tile(t_bc, 4))
    N_b_total = xy_bc.shape[0]

    model = MLP(layers=LAYERS)
    pde = UnsteadyHeat2DPDE(model, alpha=ALPHA)
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 3)))

    def loss_fn(p, xyt_ri, xy_i, u_i, xy_b, t_b):
        res = pde.residual(p, xyt_ri)
        pde_l = jnp.mean(res ** 2)
        u_pred_ic = pde.u(p, xy_i, jnp.zeros(xy_i.shape[0]))
        ic_l = jnp.mean((u_pred_ic - u_i) ** 2)
        bc_l = jnp.mean(pde.u(p, xy_b, t_b) ** 2)
        return pde_l + IC_W * ic_l + BC_W * bc_l

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=0.01)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))

    @jax.jit
    def step(p, s, key, xyt_r_pool):
        key, k1, k2, k3 = jax.random.split(key, 4)
        ir = safe_choice(k1, xyt_r_pool.shape[0], BATCH_R)
        ii = safe_choice(k2, N_IC, BATCH_I)
        ib = safe_choice(k3, N_b_total, BATCH_B)
        loss, g = jax.value_and_grad(loss_fn)(
            p, xyt_r_pool[ir], xy_ic[ii], u_ic[ii], xy_bc[ib], t_bc4[ib])
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, key, loss

    final_params, final_loss, wall = _timed_train_adaptive(
        step, params0, opt.init(params0), epochs, jax.random.PRNGKey(seed + 1),
        xyt_r0, pde, domain_sampler, N_R, sampling, resample_period, seed, adaptive_frac=adaptive_frac)

    N_eval, Nt_eval = 41, 21
    x_eval = np.linspace(0.0, 1.0, N_eval)
    y_eval = np.linspace(0.0, 1.0, N_eval)
    t_eval = np.linspace(0.0, T_MAX, Nt_eval)
    XX, YY, TT = np.meshgrid(x_eval, y_eval, t_eval, indexing="ij")
    u_exact = (np.sin(np.pi * XX) * np.sin(np.pi * YY)
              * np.exp(-2.0 * ALPHA * np.pi ** 2 * TT))
    xy_eval = jnp.array(np.stack([XX.ravel(), YY.ravel()], axis=1), "f4")
    t_eval_flat = jnp.array(TT.ravel(), "f4")
    u_pred = np.array(pde.u(final_params, xy_eval, t_eval_flat)).reshape(N_eval, N_eval, Nt_eval)
    rel_l2 = float(np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-10))
    return {"problem": "heat2d_unsteady", "epochs": epochs, "wall_s": wall,
           "ms_per_epoch": 1e3 * wall / epochs, "final_loss": final_loss,
           "rel_l2": rel_l2, "sampling": sampling, "adaptive_frac": adaptive_frac}


# ═══════════════════════════════════════════════════════════════════════════

RUNNERS = {
    "wave1d": (run_wave, 5000, "result_wave1d.json"),
    "heat2d": (run_heat, 5000, "result_heat2d.json"),
    "helmholtz2d": (run_helmholtz, 10000, "result_helmholtz2d.json"),
    "ode_harmonic": (run_ode_harmonic, 3000, "result_ode_harmonic.json"),
    "diffusion1d": (run_diffusion, 5000, "result_diffusion1d.json"),
    "heat2d_unsteady": (run_heat2d_unsteady, 5000, "result_heat2d_unsteady.json"),
}


def main() -> int:
    ap = base_parser("underPINN JAX PINNs on the wave/heat/helmholtz/ode "
                     "setups compared against PhysicsNeMo")
    ap.add_argument("--problems", nargs="*", default=list(RUNNERS),
                    choices=list(RUNNERS))
    ap.add_argument("--sampling", default="uniform",
                    choices=["uniform", "qr_deim", "rad"],
                    help="'uniform': fixed pool drawn once (default, matches "
                         "the original comparison). 'qr_deim'/'rad': "
                         "periodically replace the interior collocation "
                         "pool with a QR-DEIM-R / RAD adaptive resample "
                         "based on the live PDE residual.")
    ap.add_argument("--resample-period", type=int, default=500,
                    help="epochs between adaptive resamples (ignored for "
                         "--sampling uniform)")
    ap.add_argument("--adaptive-frac", type=float, default=1.0,
                    help="fraction of the pool subject to resampling "
                         "(ignored for --sampling uniform). 1.0 (default) "
                         "replaces the whole pool every cycle. <1.0 keeps "
                         "1-adaptive_frac of the pool permanently fixed "
                         "(uniform, for stability) and only resamples the "
                         "rest (adaptive, for improvement) -- the same "
                         "majority-fixed/minority-adaptive split "
                         "ablate_qr_deim_ramp_ns.py uses.")
    args = ap.parse_args()

    info = jax_device_info(require_gpu=not args.allow_cpu)
    warn_if_cpu(info)
    print(f"JAX backend: {info['platform']} ({info['device_name']})")
    print(f"sampling: {args.sampling}   resample_period: "
         f"{args.resample_period}   adaptive_frac: {args.adaptive_frac}\n")

    rows = {}
    for name in args.problems:
        fn, epochs, _ = RUNNERS[name]
        print(f"--- {name} (epochs={epochs})")
        r = fn(epochs, args.seed, sampling=args.sampling,
              resample_period=args.resample_period,
              adaptive_frac=args.adaptive_frac)
        rows[name] = r
        print(f"    {r['ms_per_epoch']:7.3f} ms/ep  final_loss={r['final_loss']:.4e}"
             f"  rel_L2={r['rel_l2']:.4e}")

    frac_tag = "" if args.adaptive_frac >= 1.0 else f"_frac{args.adaptive_frac:g}"
    tag = ("physicsnemo_compare_underpinn_multi" if args.sampling == "uniform"
          else f"physicsnemo_compare_underpinn_multi_{args.sampling}{frac_tag}")
    save_result(tag, {"device": info, "sampling": args.sampling,
                      "resample_period": args.resample_period,
                      "adaptive_frac": args.adaptive_frac, "runs": rows})

    print("\n" + "=" * 90)
    print(f"{'problem':16s} {'underPINN ms/ep':>16s} {'PN ms/ep':>10s} "
         f"{'underPINN rel_L2':>18s} {'PN rel_L2':>12s}")
    print("-" * 90)
    for name in args.problems:
        r = rows[name]
        _, _, pn_file = RUNNERS[name]
        pn_path = os.path.join(_HERE, "multi", pn_file)
        if os.path.exists(pn_path):
            with open(pn_path) as fh:
                pn = json.load(fh)
            pn_ms = pn["ms_per_epoch"]
            pn_l2 = pn.get("rel_l2", pn.get("rel_l2_axial_velocity"))
            print(f"{name:16s} {r['ms_per_epoch']:16.3f} {pn_ms:10.3f} "
                 f"{r['rel_l2']:18.4e} {pn_l2:12.4e}")
        else:
            print(f"{name:16s} {r['ms_per_epoch']:16.3f} {'n/a':>10s} "
                 f"{r['rel_l2']:18.4e} {'n/a':>12s}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ═══════════════════════════════════════════════════════════════════════════
# 3-D pipe flow, matches examples/pipe_flow/pipe_flow.yaml (clean timing --
# no ConsoleLogger/RestartManager overhead, unlike the full example script,
# for a fair comparison against physicsnemo's equally clean measurement)
# ═══════════════════════════════════════════════════════════════════════════

def run_pipe_flow(epochs: int, seed: int, sampling: str = "uniform",
                  resample_period: int = 500,
                    adaptive_frac: float = 1.0) -> dict:
    from underPINN.nn.mlp import GatedMLP
    from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE
    from underPINN.geometry.pipe import Pipe

    RE, R, L, U_MAX, X_LO = 40.0, 0.5, 7.0, 2.0, -3.5
    N_INTERIOR, N_WALL, N_INLET, N_OUTLET = 100000, 15000, 2000, 2000
    BATCH_R, BATCH_BC = 2048, 1024
    W_PDE, W_WALL, W_INLET, W_OUTLET = 1.0, 100.0, 50.0, 20.0
    LAYERS = [3, 192, 192, 192, 192, 4]
    LR = 1e-3
    # A full-cost 5x-candidate resample at N_INTERIOR=100,000 (500,000 NS
    # residual evals, each needing 1st/2nd-order autodiff of a 4-output net)
    # every 500 epochs across 70,000 epochs is not the bottleneck this
    # comparison is about -- keep the resampled pool itself at N_INTERIOR
    # but use a leaner 2x candidate multiplier here specifically.
    RESAMPLE_CANDIDATE_MULT = 2

    pipe = Pipe(R=R, L=L, x_lo=X_LO)
    xyz_r0 = jnp.array(pipe.sample_interior(N_INTERIOR, seed=seed))
    xyz_w = jnp.array(pipe.sample_wall(N_WALL, seed=seed + 1))
    xyz_in = jnp.array(pipe.sample_inlet(N_INLET, seed=seed + 2))
    xyz_out = jnp.array(pipe.sample_outlet(N_OUTLET, seed=seed + 3))

    def inlet_velocity(xyz):
        r2 = xyz[:, 1] ** 2 + xyz[:, 2] ** 2
        return U_MAX * (1.0 - r2 / R ** 2)

    model = GatedMLP(layers=LAYERS)
    pde = SteadyNS3DPDE(model, Re=RE)
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 3)))

    def loss_fn(p, xr, xw, xin, xout):
        res = pde.residual(p, xr)
        pde_l = jnp.mean(jnp.sum(res ** 2, axis=-1))
        out_w = model.apply(p, xw)
        wall_l = jnp.mean(out_w[:, 0] ** 2 + out_w[:, 1] ** 2 + out_w[:, 2] ** 2)
        out_in = model.apply(p, xin)
        u_ex = inlet_velocity(xin)
        in_l = (jnp.mean((out_in[:, 0] - u_ex) ** 2)
               + jnp.mean(out_in[:, 1] ** 2) + jnp.mean(out_in[:, 2] ** 2))
        out_out = model.apply(p, xout)
        outlet_l = jnp.mean(out_out[:, 3] ** 2)
        return (W_PDE * pde_l + W_WALL * wall_l + W_INLET * in_l
               + W_OUTLET * outlet_l)

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=0.01)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))

    N_w = xyz_w.shape[0]
    N_in, N_out = xyz_in.shape[0], xyz_out.shape[0]

    def domain_sampler(n, sd):
        return pipe.sample_interior(n, seed=sd)

    @jax.jit
    def step(p, s, key, xyz_r_pool):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        ir = safe_choice(k1, xyz_r_pool.shape[0], BATCH_R)
        iw = safe_choice(k2, N_w, BATCH_BC)
        iin = safe_choice(k3, N_in, min(BATCH_BC, N_in))
        iout = safe_choice(k4, N_out, min(BATCH_BC, N_out))
        loss, g = jax.value_and_grad(loss_fn)(
            p, xyz_r_pool[ir], xyz_w[iw], xyz_in[iin], xyz_out[iout])
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, key, loss

    final_params, final_loss, wall = _timed_train_adaptive(
        step, params0, opt.init(params0), epochs, jax.random.PRNGKey(seed + 99),
        xyz_r0, pde, domain_sampler, N_INTERIOR, sampling, resample_period,
        seed, candidate_mult=RESAMPLE_CANDIDATE_MULT,
        adaptive_frac=adaptive_frac)

    # score exactly as multi/pipe_flow3d.py does (3000 random points, same
    # rng seed=99 convention, its own axis labeling: x axial here)
    rng = np.random.default_rng(99)
    n_val = 3000
    rr = R * np.sqrt(rng.uniform(0.0, 1.0, n_val))
    th = rng.uniform(0.0, 2 * np.pi, n_val)
    y_v = (rr * np.cos(th)).astype(np.float32)
    z_v = (rr * np.sin(th)).astype(np.float32)
    x_v = rng.uniform(X_LO, X_LO + L, n_val).astype(np.float32)
    xyz_val = jnp.array(np.stack([x_v, y_v, z_v], axis=1))
    u_p, v_p, w_p, p_p = pde.exact_poiseuille(xyz_val, R=R, U_max=U_MAX, L=L, x_lo=X_LO)
    out_val = model.apply(final_params, xyz_val)

    def rel_l2(pred, exact):
        return float(jnp.linalg.norm(pred - exact) / (jnp.linalg.norm(exact) + 1e-10))

    rel_l2_u = rel_l2(out_val[:, 0], u_p)
    rel_l2_p = rel_l2(out_val[:, 3], p_p)
    return {"problem": "pipe_flow_3d", "epochs": epochs, "wall_s": wall,
           "ms_per_epoch": 1e3 * wall / epochs, "final_loss": final_loss,
           "rel_l2_axial_velocity": rel_l2_u, "rel_l2_pressure": rel_l2_p,
           "sampling": sampling, "adaptive_frac": adaptive_frac}
