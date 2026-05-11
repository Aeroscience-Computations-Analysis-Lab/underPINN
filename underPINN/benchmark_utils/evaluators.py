"""Per-problem benchmark evaluators for underPINN.

Each evaluator is self-contained: it builds a model + data, trains it for a
given epoch budget, then evaluates accuracy against an exact or high-fidelity
reference solution.

Evaluators expose a uniform interface used by :class:`BenchmarkRunner`:

    ev = BurgersEvaluator()
    wall = ev.train(epochs=5000, seed=0)
    metrics = ev.evaluate()    # {'rel_l2': ..., 'max_ae': ...}
    print(ev.loss_hist[-1])
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax

# ── shared imports ────────────────────────────────────────────────────────────
from underPINN.nn.mlp import MLP, FourierMLP
from underPINN.core.config import TrainingConfig
from underPINN.utils.metrics import relative_l2_error, max_absolute_error


# =============================================================================
#  Abstract base
# =============================================================================

class BaseBenchmarkEvaluator(ABC):
    """Protocol shared by all evaluators."""

    #: Short machine-readable key used as dict key and file stem
    name: str
    #: Human-readable label for legend / report
    label: str
    #: Set False to skip in fast-mode runs (e.g. expensive 3-D evaluators)
    fast: bool = True

    @abstractmethod
    def train(self, epochs: int, seed: int = 0) -> float:
        """Train the model for *epochs* steps.

        Returns
        -------
        float
            Wall-clock training time in seconds.
        """

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Compute accuracy metrics after :meth:`train`.

        Returns
        -------
        dict with keys ``rel_l2`` (relative L2 error, NaN if unavailable),
        ``max_ae`` (maximum absolute error), and any problem-specific extras.
        """

    @property
    def loss_hist(self) -> list:
        return getattr(self, "_loss_hist", [])

    @property
    def pde_hist(self) -> list:
        return getattr(self, "_pde_hist", [])

    @property
    def params(self):
        return getattr(self, "_params", None)


# =============================================================================
#  1-D Burgers  u_t + u u_x = ν u_xx
# =============================================================================

def _burgers_reference(nu: float = 0.01, N_x: int = 256) -> tuple:
    """High-fidelity Burgers reference via scipy RK45 + upwind FD.

    Returns (x_int, t_grid, U) where U[i, j] ≈ u(x_int[i], t_grid[j]).
    x_int are the *interior* x-points (excluding ±1 BCs).
    """
    from scipy.integrate import solve_ivp

    x = np.linspace(-1.0, 1.0, N_x + 2)    # includes boundaries
    x_int = x[1:-1]
    dx = x[1] - x[0]
    T = 1.5

    def rhs(t, u):
        u_full = np.concatenate([[0.0], u, [0.0]])
        # First-order upwind convection
        conv = np.where(
            u_full[1:-1] >= 0,
            u_full[1:-1] * (u_full[1:-1] - u_full[:-2]) / dx,
            u_full[1:-1] * (u_full[2:] - u_full[1:-1]) / dx,
        )
        diff = nu * (u_full[2:] - 2.0 * u_full[1:-1] + u_full[:-2]) / dx ** 2
        return -conv + diff

    u0 = -np.sin(np.pi * x_int)
    t_eval = np.linspace(0.0, T, 201)
    sol = solve_ivp(
        rhs, [0.0, T], u0, method="RK45",
        t_eval=t_eval, rtol=1e-9, atol=1e-11,
    )
    return x_int, sol.t, sol.y  # (N_x,), (201,), (N_x, 201)


class BurgersEvaluator(BaseBenchmarkEvaluator):
    name  = "burgers"
    label = "1-D Burgers (ν=0.01)"

    def __init__(self, nu: float = 0.01):
        from underPINN.pde.burgers import BurgersPDE
        from underPINN.losses.loss import PINNLoss
        from underPINN.solver.fbpinn import FBPINNSolver

        self._nu = nu
        self._FBPINNSolver = FBPINNSolver
        self._BurgersPDE = BurgersPDE
        self._PINNLoss = PINNLoss

    def train(self, epochs: int, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        N_r, N_ic, N_bc = 6000, 200, 300
        T = 1.5

        x_r  = jnp.array(rng.uniform(-1.0, 1.0, N_r).astype("f4"))
        t_r  = jnp.array(rng.uniform( 0.0,  T,   N_r).astype("f4"))
        x_ic = jnp.array(np.linspace(-1, 1, N_ic, dtype="f4"))
        u_ic = jnp.array(-np.sin(np.pi * x_ic))
        t_bc = rng.uniform(0.0, T, N_bc).astype("f4")
        x_bc = jnp.array(np.tile([-1.0, 1.0], N_bc).astype("f4"))
        t_bc = jnp.array(np.tile(t_bc, 2))
        u_bc = jnp.zeros(2 * N_bc, dtype="f4")

        model  = MLP(layers=[2, 64, 64, 64, 1])
        pde    = self._BurgersPDE(model, nu=self._nu)
        loss   = self._PINNLoss(model, pde, ic_weight=100.0, bc_weight=10.0, rba=True)
        solver = self._FBPINNSolver(model, pde, loss=loss)
        solver.init(jax.random.PRNGKey(seed))

        cfg = TrainingConfig(
            epochs=epochs, lr=1e-3,
            lr_schedule=optax.cosine_decay_schedule(1e-3, epochs, alpha=1e-2),
            batch_r=2048, batch_i=200, batch_b=300,
            log_every=max(1, epochs // 5),
        )

        t0 = time.perf_counter()
        solver.train(x_r, t_r, x_ic, u_ic, x_bc, t_bc, u_bc, config=cfg)
        wall = time.perf_counter() - t0

        self._model  = model
        self._pde    = pde
        self._params = solver.params
        self._loss_hist = solver.loss_hist
        self._pde_hist  = solver.pde_hist
        return wall

    def evaluate(self) -> dict:
        # Build reference on a 100×100 (x,t) test grid
        t_eval = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
        x_ref, t_ref_grid, U_ref = _burgers_reference(self._nu)

        preds, refs = [], []
        for t_val in t_eval:
            idx = int(np.argmin(np.abs(t_ref_grid - t_val)))
            u_ref = U_ref[:, idx].astype("f4")
            x_pts = x_ref.astype("f4")

            pts = jnp.stack([jnp.array(x_pts),
                             jnp.full(len(x_pts), t_val, "f4")], axis=1)
            u_pred = self._model.apply(self._params, pts)[:, 0]
            preds.append(np.array(u_pred))
            refs.append(u_ref)

        u_pred_all = np.concatenate(preds)
        u_ref_all  = np.concatenate(refs)
        rel_l2 = float(relative_l2_error(
            jnp.array(u_pred_all), jnp.array(u_ref_all)))
        max_ae = float(max_absolute_error(
            jnp.array(u_pred_all), jnp.array(u_ref_all)))
        return {"rel_l2": rel_l2, "max_ae": max_ae}


# =============================================================================
#  1-D Wave  u_tt = c² u_xx
# =============================================================================

class WaveEvaluator(BaseBenchmarkEvaluator):
    name  = "wave"
    label = "1-D Wave (c=1)"

    def __init__(self, c: float = 1.0):
        from underPINN.pde.wave import WavePDE
        self._c = c
        self._WavePDE = WavePDE

    def train(self, epochs: int, seed: int = 0) -> float:
        c = self._c
        T = 2.0
        rng = np.random.default_rng(seed)
        N_r, N_ic, N_bc = 6000, 300, 300

        x_r  = jnp.array(rng.uniform(-1, 1, N_r).astype("f4"))
        t_r  = jnp.array(rng.uniform( 0, T, N_r).astype("f4"))
        x_ic = jnp.array(np.linspace(-1, 1, N_ic, dtype="f4"))
        u_ic = jnp.array(np.sin(np.pi * x_ic).astype("f4"))
        t_bc_half = rng.uniform(0, T, N_bc).astype("f4")
        x_bc = jnp.array(np.concatenate([np.full(N_bc, -1., "f4"),
                                          np.full(N_bc,  1., "f4")]))
        t_bc = jnp.array(np.concatenate([t_bc_half, t_bc_half]))

        sigma = max(2.0, float(c) * np.pi)
        model = FourierMLP(layers=[2, 64, 64, 64, 1], n_fourier=16, sigma=sigma)
        pde   = self._WavePDE(model, c=c)

        lr_sched  = optax.cosine_decay_schedule(1e-3, epochs, alpha=1e-2)
        optimizer = optax.chain(optax.scale_by_adam(),
                                optax.scale_by_schedule(lr_sched),
                                optax.scale(-1.0))
        key    = jax.random.PRNGKey(seed)
        params = model.init(key, jnp.ones((1, 2)))
        state  = optimizer.init(params)

        IC_W, IC_DOT_W, BC_W = 100.0, 100.0, 10.0
        N_R, N_IC, N_BC = N_r, N_ic, x_bc.shape[0]
        bR, bI, bB = 2048, 256, 256

        @jax.jit
        def step(params, state, xr, tr, xic, uic, xbc, tbc):
            def loss_fn(p):
                res  = pde.residual(p, xr, tr)
                pl   = jnp.mean(res ** 2)
                il   = jnp.mean((pde.u(p, xic, jnp.zeros_like(xic)) - uic) ** 2)
                dl   = jnp.mean(pde.u_t(p, xic, jnp.zeros_like(xic)) ** 2)
                bl   = jnp.mean(pde.u(p, xbc, tbc) ** 2)
                return pl + IC_W * il + IC_DOT_W * dl + BC_W * bl, (pl, il, dl, bl)
            (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, state = optimizer.update(grads, state)
            params = optax.apply_updates(params, updates)
            return params, state, total, aux

        loss_hist = []
        key = jax.random.PRNGKey(seed + 77)
        t0 = time.perf_counter()
        for ep in range(epochs):
            key, k1, k2, k3 = jax.random.split(key, 4)
            ir = jax.random.randint(k1, (bR,), 0, N_R)
            ii = jax.random.randint(k2, (bI,), 0, N_IC)
            ib = jax.random.randint(k3, (bB,), 0, N_BC)
            params, state, total, _ = step(
                params, state,
                x_r[ir], t_r[ir], x_ic[ii], u_ic[ii], x_bc[ib], t_bc[ib])
            loss_hist.append(float(total))
        wall = time.perf_counter() - t0

        self._model  = model
        self._pde    = pde
        self._params = params
        self._loss_hist = loss_hist
        self._pde_hist  = [float("nan")] * len(loss_hist)
        return wall

    def evaluate(self) -> dict:
        Nx, Nt = 100, 100
        x_plt = jnp.linspace(-1, 1, Nx)
        t_plt = jnp.linspace(0, 2.0, Nt)
        XX, TT = jnp.meshgrid(x_plt, t_plt, indexing="ij")
        pts = jnp.stack([XX.ravel(), TT.ravel()], axis=1)

        u_pred  = self._model.apply(self._params, pts)[:, 0]
        u_exact = self._pde.exact(XX.ravel(), TT.ravel())

        return {
            "rel_l2": float(relative_l2_error(u_pred, u_exact)),
            "max_ae": float(max_absolute_error(u_pred, u_exact)),
        }


# =============================================================================
#  2-D Helmholtz  Δu + k²u = f
# =============================================================================

class HelmholtzEvaluator(BaseBenchmarkEvaluator):
    name  = "helmholtz"
    label = "2-D Helmholtz (k=1)"

    def __init__(self, k: float = 1.0):
        from underPINN.pde.helmholtz import HelmholtzPDE
        from underPINN.losses.steady_loss import SteadyLoss
        from underPINN.solver.steady_solver import SteadySolver
        self._k = k
        self._HelmholtzPDE = HelmholtzPDE
        self._SteadyLoss   = SteadyLoss
        self._SteadySolver = SteadySolver

    def train(self, epochs: int, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        N_r, N_b = 4000, 400

        # Interior: unit square (0,1)²
        xy_r = jnp.array(rng.uniform(0, 1, (N_r, 2)).astype("f4"))
        # Boundary: sample on all four edges
        t    = rng.uniform(0, 1, N_b).astype("f4")
        xy_b = jnp.array(np.vstack([
            np.column_stack([np.zeros(N_b), t]),
            np.column_stack([np.ones(N_b), t]),
            np.column_stack([t, np.zeros(N_b)]),
            np.column_stack([t, np.ones(N_b)]),
        ]).astype("f4"))
        u_b = jnp.zeros(4 * N_b, dtype="f4")

        sigma = max(3.0, float(self._k) * np.pi * 1.5)
        model  = FourierMLP(layers=[2, 64, 64, 64, 1], n_fourier=16, sigma=sigma)
        pde    = self._HelmholtzPDE(model, k=self._k)
        loss   = self._SteadyLoss(model, pde, bc_weight=20.0)
        solver = self._SteadySolver(model, pde, loss=loss)
        solver.init(jax.random.PRNGKey(seed))

        cfg = TrainingConfig(
            epochs=epochs, lr=1e-3,
            lr_schedule=optax.cosine_decay_schedule(1e-3, epochs, alpha=1e-2),
            batch_r=2048, batch_b=256,
            log_every=max(1, epochs // 5),
        )
        t0 = time.perf_counter()
        solver.train(xy_r, xy_b, u_b, config=cfg)
        wall = time.perf_counter() - t0

        self._model  = model
        self._pde    = pde
        self._params = solver.params
        self._loss_hist = solver.loss_hist
        self._pde_hist  = solver.pde_hist
        return wall

    def evaluate(self) -> dict:
        N = 50
        x = jnp.linspace(0, 1, N)
        xy = jnp.array(np.array(
            np.meshgrid(x, x, indexing="ij")).reshape(2, -1).T.astype("f4"))
        u_pred  = self._pde.u(self._params, xy)
        u_exact = self._pde.exact(xy)
        return {
            "rel_l2": float(relative_l2_error(u_pred, u_exact)),
            "max_ae": float(max_absolute_error(u_pred, u_exact)),
        }


# =============================================================================
#  2-D Steady Heat / Poisson  Δu = -f
# =============================================================================

class SteadyHeatEvaluator(BaseBenchmarkEvaluator):
    name  = "heat_steady"
    label = "2-D Steady Heat"

    def train(self, epochs: int, seed: int = 0) -> float:
        from underPINN.pde.heat import SteadyHeatPDE
        from underPINN.losses.steady_loss import SteadyLoss
        from underPINN.solver.steady_solver import SteadySolver

        rng = np.random.default_rng(seed)
        N_r, N_b = 4000, 400

        xy_r = jnp.array(rng.uniform(0, 1, (N_r, 2)).astype("f4"))
        t    = rng.uniform(0, 1, N_b).astype("f4")
        xy_b = jnp.array(np.vstack([
            np.column_stack([np.zeros(N_b), t]),
            np.column_stack([np.ones(N_b), t]),
            np.column_stack([t, np.zeros(N_b)]),
            np.column_stack([t, np.ones(N_b)]),
        ]).astype("f4"))
        u_b = jnp.zeros(4 * N_b, dtype="f4")

        model  = MLP(layers=[2, 64, 64, 64, 1])

        def source(x, y):
            return 2.0 * jnp.pi ** 2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

        pde    = SteadyHeatPDE(model, source_fn=source)
        loss   = SteadyLoss(model, pde, bc_weight=20.0)
        solver = SteadySolver(model, pde, loss=loss)
        solver.init(jax.random.PRNGKey(seed))

        cfg = TrainingConfig(
            epochs=epochs, lr=1e-3,
            lr_schedule=optax.cosine_decay_schedule(1e-3, epochs, alpha=1e-2),
            batch_r=2048, batch_b=256,
            log_every=max(1, epochs // 5),
        )
        t0 = time.perf_counter()
        solver.train(xy_r, xy_b, u_b, config=cfg)
        wall = time.perf_counter() - t0

        self._model  = model
        self._pde    = pde
        self._params = solver.params
        self._loss_hist = solver.loss_hist
        self._pde_hist  = solver.pde_hist
        return wall

    def evaluate(self) -> dict:
        N = 50
        x = jnp.linspace(0, 1, N)
        xy = jnp.array(np.array(
            np.meshgrid(x, x, indexing="ij")).reshape(2, -1).T.astype("f4"))
        u_pred  = self._pde.u(self._params, xy)
        u_exact = self._pde.exact(xy)
        return {
            "rel_l2": float(relative_l2_error(u_pred, u_exact)),
            "max_ae": float(max_absolute_error(u_pred, u_exact)),
        }


# =============================================================================
#  ODE — Exponential decay  u' = -λu,  u(0) = 1
# =============================================================================

class ODEExpEvaluator(BaseBenchmarkEvaluator):
    name  = "ode_exp"
    label = "ODE Exp Decay (λ=2)"

    def __init__(self, lam: float = 2.0, T: float = 3.0):
        from underPINN.pde.ode import ExponentialDecayODE
        from underPINN.losses.ode_loss import ODELoss
        from underPINN.solver.ode_solver import ODESolver
        self._lam, self._T = lam, T
        self._ODE, self._Loss, self._Solver = (
            ExponentialDecayODE, ODELoss, ODESolver)

    def train(self, epochs: int, seed: int = 0) -> float:
        T = self._T
        t_r  = jnp.linspace(0, T, 500).reshape(-1, 1).astype("f4")
        t_ic = jnp.array([[0.0]], dtype="f4")
        u_ic = jnp.array([[1.0]], dtype="f4")

        model  = MLP(layers=[1, 64, 64, 1])
        pde    = self._ODE(model, lam=self._lam)
        loss   = self._Loss(model, pde, ic_weight=50.0)
        solver = self._Solver(model, pde, loss=loss)
        solver.init(jax.random.PRNGKey(seed))

        cfg = TrainingConfig(
            epochs=epochs, lr=1e-3,
            lr_schedule=optax.cosine_decay_schedule(1e-3, epochs, alpha=1e-2),
            log_every=max(1, epochs // 5),
        )
        t0 = time.perf_counter()
        solver.train(t_r, t_ic, u_ic, config=cfg)
        wall = time.perf_counter() - t0

        self._model  = model
        self._pde    = pde
        self._params = solver.params
        self._loss_hist = solver.loss_hist
        self._pde_hist  = solver.pde_hist
        return wall

    def evaluate(self) -> dict:
        t_test = jnp.linspace(0, self._T, 1000).reshape(-1, 1).astype("f4")
        u_pred  = self._pde.u(self._params, t_test)
        u_exact = self._pde.exact(t_test)
        return {
            "rel_l2": float(relative_l2_error(u_pred, u_exact)),
            "max_ae": float(max_absolute_error(u_pred, u_exact)),
        }


# =============================================================================
#  ODE — Harmonic oscillator  u'' + ω²u = 0,  u(0)=1, u'(0)=0
# =============================================================================

class ODEHarmonicEvaluator(BaseBenchmarkEvaluator):
    name  = "ode_harmonic"
    label = "ODE Harmonic (ω=2)"

    def __init__(self, omega: float = 2.0, T: float = 3.0):
        from underPINN.pde.ode import HarmonicOscillatorODE
        from underPINN.losses.ode_loss import ODELoss
        from underPINN.solver.ode_solver import ODESolver
        self._omega, self._T = omega, T
        self._ODE, self._Loss, self._Solver = (
            HarmonicOscillatorODE, ODELoss, ODESolver)

    def train(self, epochs: int, seed: int = 0) -> float:
        T = self._T
        t_r  = jnp.linspace(0, T, 500).reshape(-1, 1).astype("f4")
        t_ic = jnp.array([[0.0]], dtype="f4")
        u_ic = jnp.array([[1.0]], dtype="f4")

        model  = FourierMLP(layers=[1, 64, 64, 1], n_fourier=16,
                            sigma=float(self._omega))
        pde    = self._ODE(model, omega=self._omega)
        loss   = self._Loss(model, pde, ic_weight=50.0, ic_derivative_weight=50.0)
        solver = self._Solver(model, pde, loss=loss)
        solver.init(jax.random.PRNGKey(seed))

        cfg = TrainingConfig(
            epochs=epochs, lr=1e-3,
            lr_schedule=optax.cosine_decay_schedule(1e-3, epochs, alpha=1e-2),
            log_every=max(1, epochs // 5),
        )
        u_ic_dot = jnp.array([[0.0]], dtype="f4")
        t0 = time.perf_counter()
        solver.train(t_r, t_ic, u_ic, u_ic_dot, config=cfg)
        wall = time.perf_counter() - t0

        self._model  = model
        self._pde    = pde
        self._params = solver.params
        self._loss_hist = solver.loss_hist
        self._pde_hist  = solver.pde_hist
        return wall

    def evaluate(self) -> dict:
        t_test = jnp.linspace(0, self._T, 1000).reshape(-1, 1).astype("f4")
        u_pred  = self._pde.u(self._params, t_test)
        u_exact = self._pde.exact(t_test)
        return {
            "rel_l2": float(relative_l2_error(u_pred, u_exact)),
            "max_ae": float(max_absolute_error(u_pred, u_exact)),
        }


# =============================================================================
#  3-D Steady Pipe Flow (Hagen-Poiseuille)
# =============================================================================

class PipeFlowEvaluator(BaseBenchmarkEvaluator):
    name  = "pipe_flow"
    label = "3-D Pipe Flow (Re=10)"
    fast  = False  # double-jacfwd Hessians are expensive

    def __init__(self, Re: float = 10.0):
        from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE
        from underPINN.geometry.pipe import Pipe
        self._Re = Re
        self._SteadyNS3DPDE = SteadyNS3DPDE
        self._Pipe = Pipe

    def train(self, epochs: int, seed: int = 0) -> float:
        R, L, U_max = 0.5, 2.0, 1.0
        pipe = self._Pipe(R=R, L=L)

        xyz_int  = jnp.array(np.array(pipe.sample_interior(2000), dtype="f4"))
        xyz_wall = jnp.array(np.array(pipe.sample_wall(600),      dtype="f4"))
        xyz_in   = jnp.array(np.array(pipe.sample_inlet(200),     dtype="f4"))
        xyz_out  = jnp.array(np.array(pipe.sample_outlet(200),    dtype="f4"))

        W_PDE, W_WALL, W_IN, W_OUT = 1.0, 100.0, 50.0, 20.0

        model = MLP(layers=[3, 64, 64, 64, 64, 4])
        pde   = self._SteadyNS3DPDE(model, Re=self._Re)

        key    = jax.random.PRNGKey(seed)
        params = model.init(key, jnp.ones((1, 3)))

        lr_sched  = optax.cosine_decay_schedule(1e-3, epochs, alpha=1e-2)
        optimizer = optax.chain(optax.scale_by_adam(),
                                optax.scale_by_schedule(lr_sched),
                                optax.scale(-1.0))
        state = optimizer.init(params)

        @jax.jit
        def step(params, state, xint, xwall, xin, xout):
            def loss_fn(p):
                cont, mx, my, mz = pde.residual(p, xint)
                pde_l = jnp.mean(cont**2 + mx**2 + my**2 + mz**2)
                u_w, v_w, w_w, _ = pde.uvwp(p, xwall)
                wall_l = jnp.mean(u_w**2 + v_w**2 + w_w**2)
                r_in  = jnp.sqrt(xin[:, 1]**2 + xin[:, 2]**2)
                u_ex  = U_max * (1 - r_in**2 / R**2)
                u_in, v_in, w_in, _ = pde.uvwp(p, xin)
                in_l  = jnp.mean((u_in - u_ex)**2 + v_in**2 + w_in**2)
                u_out, v_out, w_out, _ = pde.uvwp(p, xout)
                out_l = jnp.mean(v_out**2 + w_out**2)
                return (W_PDE*pde_l + W_WALL*wall_l + W_IN*in_l + W_OUT*out_l,
                        (pde_l, wall_l, in_l, out_l))
            (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, state = optimizer.update(grads, state)
            params = optax.apply_updates(params, updates)
            return params, state, total, aux

        loss_hist, pde_hist = [], []
        key = jax.random.PRNGKey(seed + 11)
        B = min(256, xyz_int.shape[0])
        Bwall = min(128, xyz_wall.shape[0])

        t0 = time.perf_counter()
        for ep in range(epochs):
            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            ir  = jax.random.randint(k1, (B,),     0, xyz_int.shape[0])
            iw  = jax.random.randint(k2, (Bwall,), 0, xyz_wall.shape[0])
            ii  = jax.random.randint(k3, (min(64, xyz_in.shape[0]),), 0, xyz_in.shape[0])
            io_ = jax.random.randint(k4, (min(64, xyz_out.shape[0]),), 0, xyz_out.shape[0])
            params, state, total, (pl, *_) = step(
                params, state,
                xyz_int[ir], xyz_wall[iw], xyz_in[ii], xyz_out[io_])
            loss_hist.append(float(total))
            pde_hist.append(float(pl))
        wall = time.perf_counter() - t0

        self._model  = model
        self._pde    = pde
        self._params = params
        self._loss_hist = loss_hist
        self._pde_hist  = pde_hist
        self._R, self._U_max, self._L = R, U_max, L
        return wall

    def evaluate(self) -> dict:
        xyz_test = jnp.array(np.array(
            self._Pipe(R=self._R, L=self._L).sample_interior(2000), dtype="f4"))
        u_pred, v_pred, w_pred, _ = self._pde.uvwp(self._params, xyz_test)
        _, _, _, u_ex = self._pde.exact_poiseuille(
            xyz_test, R=self._R, U_max=self._U_max, L=self._L)
        # exact_poiseuille returns (u_exact, v_exact, w_exact, p_exact)
        u_ex_axial = u_ex  # re-check exact_poiseuille signature
        # Actually exact_poiseuille(xyz, R, U_max, L) → u, v, w, p
        u_e, v_e, w_e, p_e = self._pde.exact_poiseuille(
            xyz_test, R=self._R, U_max=self._U_max, L=self._L)
        speed_pred  = jnp.sqrt(u_pred**2 + v_pred**2 + w_pred**2)
        speed_exact = jnp.sqrt(u_e**2 + v_e**2 + w_e**2)
        return {
            "rel_l2": float(relative_l2_error(speed_pred, speed_exact)),
            "max_ae": float(max_absolute_error(speed_pred, speed_exact)),
        }


# =============================================================================
#  Registry
# =============================================================================

EVALUATOR_REGISTRY: dict[str, type] = {
    "burgers":       BurgersEvaluator,
    "wave":          WaveEvaluator,
    "helmholtz":     HelmholtzEvaluator,
    "heat_steady":   SteadyHeatEvaluator,
    "ode_exp":       ODEExpEvaluator,
    "ode_harmonic":  ODEHarmonicEvaluator,
    "pipe_flow":     PipeFlowEvaluator,
}

#: Problems excluded from ``--fast`` mode
SLOW_PROBLEMS = {k for k, v in EVALUATOR_REGISTRY.items()
                 if not v.fast}
