"""Per-problem benchmark evaluators for underPINN.

Each evaluator is self-contained: builds a model + data, trains it, evaluates
accuracy against an exact/reference solution, and produces a solution plot.

Interface used by :class:`BenchmarkRunner`::

    ev = BurgersEvaluator()
    wall   = ev.train(epochs=5000, seed=0)
    metrics = ev.evaluate()          # {'rel_l2': ..., 'max_ae': ...}
    path   = ev.plot("outputs/bench") # saves {name}_solution.png
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Dict

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

from underPINN.nn.mlp import MLP, FourierMLP
from underPINN.core.config import TrainingConfig
from underPINN.utils.metrics import relative_l2_error, max_absolute_error


# =============================================================================
#  Shared plot helpers
# =============================================================================

_CMAP_SOLN = "RdBu_r"
_CMAP_ERR  = "Reds"


def _spacetime_panel(fig, axes, x_grid, t_grid, u_pred, u_exact, title):
    """Fill three axes with PINN | Exact | |Error| heatmaps for (x,t) data."""
    vmin = min(float(u_pred.min()), float(u_exact.min()))
    vmax = max(float(u_pred.max()), float(u_exact.max()))
    err  = np.abs(u_pred - u_exact)

    for ax, data, lbl in zip(
        axes[:3],
        [u_pred, u_exact, err],
        ["PINN", "Exact", "|Error|"],
    ):
        vlo, vhi = (vmin, vmax) if lbl != "|Error|" else (0, float(err.max()) + 1e-12)
        cm  = _CMAP_SOLN if lbl != "|Error|" else _CMAP_ERR
        pcm = ax.pcolormesh(x_grid, t_grid, data.T,
                            cmap=cm, vmin=vlo, vmax=vhi, shading="auto")
        fig.colorbar(pcm, ax=ax, shrink=0.85)
        ax.set_title(lbl)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
    axes[0].set_title(f"PINN  ({title})")


def _spatial2d_panel(fig, axes, x_grid, y_grid, u_pred, u_exact, title):
    """Fill three axes with PINN | Exact | |Error| for 2-D (x,y) data."""
    vmin = min(float(u_pred.min()), float(u_exact.min()))
    vmax = max(float(u_pred.max()), float(u_exact.max()))
    err  = np.abs(u_pred - u_exact)

    for ax, data, lbl in zip(
        axes[:3],
        [u_pred, u_exact, err],
        ["PINN", "Exact", "|Error|"],
    ):
        vlo, vhi = (vmin, vmax) if lbl != "|Error|" else (0, float(err.max()) + 1e-12)
        cm  = _CMAP_SOLN if lbl != "|Error|" else _CMAP_ERR
        pcm = ax.pcolormesh(x_grid, y_grid, data.T,
                            cmap=cm, vmin=vlo, vmax=vhi, shading="auto")
        fig.colorbar(pcm, ax=ax, shrink=0.85)
        ax.set_title(lbl)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    axes[0].set_title(f"PINN  ({title})")


def _loss_ax(ax, loss_hist, pde_hist=None):
    """Semilogy loss panel."""
    xs = np.arange(1, len(loss_hist) + 1)
    ax.semilogy(xs, loss_hist, lw=1.5, label="Total")
    if pde_hist and not all(np.isnan(pde_hist)):
        ax.semilogy(xs, pde_hist, lw=1.2, ls="--", label="PDE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)


# =============================================================================
#  Abstract base
# =============================================================================

class BaseBenchmarkEvaluator(ABC):
    """Protocol shared by all evaluators."""

    name:  str        #: machine-readable key, used as file-stem
    label: str        #: human label for legends
    fast:  bool = True

    @abstractmethod
    def train(self, epochs: int, seed: int = 0) -> float:
        """Train for *epochs* steps; return wall time in seconds."""

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Return {'rel_l2', 'max_ae', ...} after train()."""

    @abstractmethod
    def plot(self, out_dir: str, suffix: str = "") -> str:
        """Save solution figure to *out_dir*; return the file path."""

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
    """High-fidelity Burgers reference via scipy RK45 + upwind FD."""
    from scipy.integrate import solve_ivp
    x   = np.linspace(-1.0, 1.0, N_x + 2)
    x_int = x[1:-1]
    dx  = x[1] - x[0]
    T   = 1.5

    def rhs(t, u):
        u_full = np.concatenate([[0.0], u, [0.0]])
        conv = np.where(
            u_full[1:-1] >= 0,
            u_full[1:-1] * (u_full[1:-1] - u_full[:-2]) / dx,
            u_full[1:-1] * (u_full[2:] - u_full[1:-1]) / dx,
        )
        diff = nu * (u_full[2:] - 2.0 * u_full[1:-1] + u_full[:-2]) / dx**2
        return -conv + diff

    u0    = -np.sin(np.pi * x_int)
    t_eval = np.linspace(0.0, T, 201)
    sol   = solve_ivp(rhs, [0.0, T], u0, method="RK45",
                      t_eval=t_eval, rtol=1e-9, atol=1e-11)
    return x_int, sol.t, sol.y   # (N_x,), (201,), (N_x, 201)


class BurgersEvaluator(BaseBenchmarkEvaluator):
    name  = "burgers"
    label = "1-D Burgers (ν=0.01)"

    def __init__(self, nu: float = 0.01):
        from underPINN.pde.burgers import BurgersPDE
        from underPINN.losses.loss import PINNLoss
        from underPINN.solver.fbpinn import FBPINNSolver
        self._nu = nu
        self._FBPINNSolver = FBPINNSolver
        self._BurgersPDE   = BurgersPDE
        self._PINNLoss     = PINNLoss

    def train(self, epochs: int, seed: int = 0) -> float:
        rng  = np.random.default_rng(seed)
        N_r, N_ic, N_bc = 6000, 200, 300
        T    = 1.5
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
        self._model      = model
        self._pde        = pde
        self._params     = solver.params
        self._loss_hist  = solver.loss_hist
        self._pde_hist   = solver.pde_hist
        self._T          = T
        return wall

    def evaluate(self) -> dict:
        t_eval = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
        x_ref, t_ref_grid, U_ref = _burgers_reference(self._nu)
        preds, refs = [], []
        for t_val in t_eval:
            idx    = int(np.argmin(np.abs(t_ref_grid - t_val)))
            u_ref  = U_ref[:, idx].astype("f4")
            x_pts  = x_ref.astype("f4")
            pts    = jnp.stack([jnp.array(x_pts),
                                jnp.full(len(x_pts), t_val, "f4")], axis=1)
            u_pred = self._model.apply(self._params, pts)[:, 0]
            preds.append(np.array(u_pred))
            refs.append(u_ref)
        u_pred_all = np.concatenate(preds)
        u_ref_all  = np.concatenate(refs)
        return {
            "rel_l2": float(relative_l2_error(jnp.array(u_pred_all),
                                               jnp.array(u_ref_all))),
            "max_ae": float(max_absolute_error(jnp.array(u_pred_all),
                                               jnp.array(u_ref_all))),
        }

    def plot(self, out_dir: str, suffix: str = "") -> str:
        Nx, Nt = 200, 100
        T = self._T
        x_plt = np.linspace(-1, 1, Nx, dtype="f4")
        t_plt = np.linspace(0,  T, Nt, dtype="f4")
        XX, TT = np.meshgrid(x_plt, t_plt, indexing="ij")
        pts    = jnp.stack([jnp.array(XX.ravel()), jnp.array(TT.ravel())], axis=1)
        u_pred = np.array(self._model.apply(self._params, pts)[:, 0]).reshape(Nx, Nt)

        # Reference on same grid
        x_ref, t_ref, U_ref = _burgers_reference(self._nu)
        from scipy.interpolate import RegularGridInterpolator
        interp  = RegularGridInterpolator((x_ref, t_ref), U_ref,
                                          method="linear", bounds_error=False,
                                          fill_value=0.0)
        u_exact = interp(np.column_stack([XX.ravel(), TT.ravel()])).reshape(Nx, Nt)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        _spacetime_panel(fig, axes[:3], x_plt, t_plt, u_pred, u_exact,
                         f"Burgers ν={self._nu}")
        _loss_ax(axes[3], self._loss_hist, self._pde_hist)
        fig.suptitle(f"1-D Burgers (ν={self._nu})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"burgers{suffix}_solution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {path}")
        return path


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
        c    = self._c
        T    = 2.0
        rng  = np.random.default_rng(seed)
        N_r, N_ic, N_bc = 6000, 300, 300
        x_r  = jnp.array(rng.uniform(-1, 1, N_r).astype("f4"))
        t_r  = jnp.array(rng.uniform( 0, T, N_r).astype("f4"))
        x_ic = jnp.array(np.linspace(-1, 1, N_ic, dtype="f4"))
        u_ic = jnp.array(np.sin(np.pi * x_ic).astype("f4"))
        t_bc_half = rng.uniform(0, T, N_bc).astype("f4")
        x_bc = jnp.array(np.concatenate([np.full(N_bc, -1., "f4"),
                                          np.full(N_bc,  1., "f4")]))
        t_bc = jnp.array(np.concatenate([t_bc_half, t_bc_half]))

        sigma    = max(2.0, float(c) * np.pi)
        model    = FourierMLP(layers=[2, 64, 64, 64, 1], n_fourier=16, sigma=sigma)
        pde      = self._WavePDE(model, c=c)
        lr_sched = optax.cosine_decay_schedule(1e-3, epochs, alpha=1e-2)
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
                return pl + IC_W*il + IC_DOT_W*dl + BC_W*bl, (pl, il, dl, bl)
            (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, state = optimizer.update(grads, state)
            params = optax.apply_updates(params, updates)
            return params, state, total, aux

        loss_hist = []
        key = jax.random.PRNGKey(seed + 77)
        t0  = time.perf_counter()
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
        self._model     = model
        self._pde       = pde
        self._params    = params
        self._loss_hist = loss_hist
        self._pde_hist  = [float("nan")] * len(loss_hist)
        self._T         = T
        return wall

    def evaluate(self) -> dict:
        Nx, Nt = 100, 100
        x_plt  = jnp.linspace(-1, 1, Nx)
        t_plt  = jnp.linspace(0, self._T, Nt)
        XX, TT = jnp.meshgrid(x_plt, t_plt, indexing="ij")
        pts    = jnp.stack([XX.ravel(), TT.ravel()], axis=1)
        u_pred  = self._model.apply(self._params, pts)[:, 0]
        u_exact = self._pde.exact(XX.ravel(), TT.ravel())
        return {
            "rel_l2": float(relative_l2_error(u_pred, u_exact)),
            "max_ae": float(max_absolute_error(u_pred, u_exact)),
        }

    def plot(self, out_dir: str, suffix: str = "") -> str:
        Nx, Nt = 200, 100
        T      = self._T
        x_plt  = np.linspace(-1, 1, Nx, dtype="f4")
        t_plt  = np.linspace( 0, T, Nt, dtype="f4")
        XX, TT = np.meshgrid(x_plt, t_plt, indexing="ij")
        pts    = jnp.stack([jnp.array(XX.ravel()), jnp.array(TT.ravel())], axis=1)
        u_pred  = np.array(self._model.apply(self._params, pts)[:, 0]).reshape(Nx, Nt)
        u_exact = np.array(self._pde.exact(
            jnp.array(XX.ravel()), jnp.array(TT.ravel()))).reshape(Nx, Nt)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        _spacetime_panel(fig, axes[:3], x_plt, t_plt, u_pred, u_exact,
                         f"Wave c={self._c}")
        _loss_ax(axes[3], self._loss_hist)
        fig.suptitle(f"1-D Wave (c={self._c})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"wave{suffix}_solution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {path}")
        return path


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
        self._k            = k
        self._HelmholtzPDE = HelmholtzPDE
        self._SteadyLoss   = SteadyLoss
        self._SteadySolver = SteadySolver

    def train(self, epochs: int, seed: int = 0) -> float:
        rng  = np.random.default_rng(seed)
        N_r, N_b = 4000, 400
        xy_r = jnp.array(rng.uniform(0, 1, (N_r, 2)).astype("f4"))
        t    = rng.uniform(0, 1, N_b).astype("f4")
        xy_b = jnp.array(np.vstack([
            np.column_stack([np.zeros(N_b), t]),
            np.column_stack([np.ones(N_b),  t]),
            np.column_stack([t, np.zeros(N_b)]),
            np.column_stack([t, np.ones(N_b)]),
        ]).astype("f4"))
        u_b  = jnp.zeros(4 * N_b, dtype="f4")

        sigma  = max(3.0, float(self._k) * np.pi * 1.5)
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
        self._model     = model
        self._pde       = pde
        self._params    = solver.params
        self._loss_hist = solver.loss_hist
        self._pde_hist  = solver.pde_hist
        return wall

    def evaluate(self) -> dict:
        N   = 50
        x   = jnp.linspace(0, 1, N)
        xy  = jnp.array(np.array(
            np.meshgrid(x, x, indexing="ij")).reshape(2, -1).T.astype("f4"))
        u_pred  = self._pde.u(self._params, xy)
        u_exact = self._pde.exact(xy)
        return {
            "rel_l2": float(relative_l2_error(u_pred, u_exact)),
            "max_ae": float(max_absolute_error(u_pred, u_exact)),
        }

    def plot(self, out_dir: str, suffix: str = "") -> str:
        N  = 100
        x  = np.linspace(0, 1, N, dtype="f4")
        XY = np.array(np.meshgrid(x, x, indexing="ij")).reshape(2, -1).T
        xy = jnp.array(XY)
        u_pred  = np.array(self._pde.u(self._params, xy)).reshape(N, N)
        u_exact = np.array(self._pde.exact(xy)).reshape(N, N)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        _spatial2d_panel(fig, axes[:3], x, x, u_pred, u_exact,
                         f"Helmholtz k={self._k}")
        _loss_ax(axes[3], self._loss_hist, self._pde_hist)
        fig.suptitle(f"2-D Helmholtz (k={self._k})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"helmholtz{suffix}_solution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {path}")
        return path


# =============================================================================
#  2-D Steady Heat / Poisson
# =============================================================================

class SteadyHeatEvaluator(BaseBenchmarkEvaluator):
    name  = "heat_steady"
    label = "2-D Steady Heat"

    def train(self, epochs: int, seed: int = 0) -> float:
        from underPINN.pde.heat import SteadyHeatPDE
        from underPINN.losses.steady_loss import SteadyLoss
        from underPINN.solver.steady_solver import SteadySolver

        rng  = np.random.default_rng(seed)
        N_r, N_b = 4000, 400
        xy_r = jnp.array(rng.uniform(0, 1, (N_r, 2)).astype("f4"))
        t    = rng.uniform(0, 1, N_b).astype("f4")
        xy_b = jnp.array(np.vstack([
            np.column_stack([np.zeros(N_b), t]),
            np.column_stack([np.ones(N_b),  t]),
            np.column_stack([t, np.zeros(N_b)]),
            np.column_stack([t, np.ones(N_b)]),
        ]).astype("f4"))
        u_b  = jnp.zeros(4 * N_b, dtype="f4")

        def source(x, y):
            return 2.0 * jnp.pi**2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

        model  = MLP(layers=[2, 64, 64, 64, 1])
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
        self._model     = model
        self._pde       = pde
        self._params    = solver.params
        self._loss_hist = solver.loss_hist
        self._pde_hist  = solver.pde_hist
        return wall

    def evaluate(self) -> dict:
        N   = 50
        x   = jnp.linspace(0, 1, N)
        xy  = jnp.array(np.array(
            np.meshgrid(x, x, indexing="ij")).reshape(2, -1).T.astype("f4"))
        u_pred  = self._pde.u(self._params, xy)
        u_exact = self._pde.exact(xy)
        return {
            "rel_l2": float(relative_l2_error(u_pred, u_exact)),
            "max_ae": float(max_absolute_error(u_pred, u_exact)),
        }

    def plot(self, out_dir: str, suffix: str = "") -> str:
        N  = 100
        x  = np.linspace(0, 1, N, dtype="f4")
        XY = np.array(np.meshgrid(x, x, indexing="ij")).reshape(2, -1).T
        xy = jnp.array(XY)
        u_pred  = np.array(self._pde.u(self._params, xy)).reshape(N, N)
        u_exact = np.array(self._pde.exact(xy)).reshape(N, N)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        _spatial2d_panel(fig, axes[:3], x, x, u_pred, u_exact, "Steady Heat")
        _loss_ax(axes[3], self._loss_hist, self._pde_hist)
        fig.suptitle("2-D Steady Heat  (∇²u = -f)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"heat_steady{suffix}_solution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {path}")
        return path


# =============================================================================
#  ODE — Exponential decay
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
        T    = self._T
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
        self._model     = model
        self._pde       = pde
        self._params    = solver.params
        self._loss_hist = solver.loss_hist
        self._pde_hist  = solver.pde_hist
        return wall

    def evaluate(self) -> dict:
        t_test  = jnp.linspace(0, self._T, 1000).reshape(-1, 1).astype("f4")
        u_pred  = self._pde.u(self._params, t_test)
        u_exact = self._pde.exact(t_test)
        return {
            "rel_l2": float(relative_l2_error(u_pred, u_exact)),
            "max_ae": float(max_absolute_error(u_pred, u_exact)),
        }

    def plot(self, out_dir: str, suffix: str = "") -> str:
        t_test  = jnp.linspace(0, self._T, 500).reshape(-1, 1).astype("f4")
        u_pred  = np.array(self._pde.u(self._params, t_test)).ravel()
        u_exact = np.array(self._pde.exact(t_test)).ravel()
        t_np    = np.array(t_test).ravel()
        err     = np.abs(u_pred - u_exact)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Panel 1: solution
        axes[0].plot(t_np, u_exact, "k-",  lw=2,   label="Exact")
        axes[0].plot(t_np, u_pred,  "r--", lw=1.8, label="PINN")
        axes[0].set_xlabel("t"); axes[0].set_ylabel("u(t)")
        axes[0].set_title(f"Exp Decay  u'=−{self._lam}u")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        # Panel 2: point-wise error
        axes[1].plot(t_np, err, "b-", lw=1.5)
        axes[1].set_xlabel("t"); axes[1].set_ylabel("|PINN − Exact|")
        axes[1].set_title("Absolute Error")
        axes[1].grid(True, alpha=0.3)

        # Panel 3: loss
        _loss_ax(axes[2], self._loss_hist, self._pde_hist)

        fig.suptitle(f"ODE Exp Decay  λ={self._lam}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"ode_exp{suffix}_solution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {path}")
        return path


# =============================================================================
#  ODE — Harmonic oscillator
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
        T    = self._T
        t_r  = jnp.linspace(0, T, 500).reshape(-1, 1).astype("f4")
        t_ic = jnp.array([[0.0]], dtype="f4")
        u_ic = jnp.array([[1.0]], dtype="f4")
        model  = FourierMLP(layers=[1, 64, 64, 1], n_fourier=16,
                            sigma=float(self._omega))
        pde    = self._ODE(model, omega=self._omega)
        loss   = self._Loss(model, pde, ic_weight=50.0, ic_derivative_weight=50.0)
        solver = self._Solver(model, pde, loss=loss)
        solver.init(jax.random.PRNGKey(0))
        cfg = TrainingConfig(
            epochs=epochs, lr=1e-3,
            lr_schedule=optax.cosine_decay_schedule(1e-3, epochs, alpha=1e-2),
            log_every=max(1, epochs // 5),
        )
        u_ic_dot = jnp.array([[0.0]], dtype="f4")
        t0 = time.perf_counter()
        solver.train(t_r, t_ic, u_ic, u_ic_dot, config=cfg)
        wall = time.perf_counter() - t0
        self._model     = model
        self._pde       = pde
        self._params    = solver.params
        self._loss_hist = solver.loss_hist
        self._pde_hist  = solver.pde_hist
        return wall

    def evaluate(self) -> dict:
        t_test  = jnp.linspace(0, self._T, 1000).reshape(-1, 1).astype("f4")
        u_pred  = self._pde.u(self._params, t_test)
        u_exact = self._pde.exact(t_test)
        return {
            "rel_l2": float(relative_l2_error(u_pred, u_exact)),
            "max_ae": float(max_absolute_error(u_pred, u_exact)),
        }

    def plot(self, out_dir: str, suffix: str = "") -> str:
        t_test  = jnp.linspace(0, self._T, 500).reshape(-1, 1).astype("f4")
        u_pred  = np.array(self._pde.u(self._params, t_test)).ravel()
        u_exact = np.array(self._pde.exact(t_test)).ravel()
        t_np    = np.array(t_test).ravel()
        err     = np.abs(u_pred - u_exact)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].plot(t_np, u_exact, "k-",  lw=2,   label="Exact")
        axes[0].plot(t_np, u_pred,  "r--", lw=1.8, label="PINN")
        axes[0].set_xlabel("t"); axes[0].set_ylabel("u(t)")
        axes[0].set_title(f"Harmonic  u''+{self._omega}²u=0")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_np, err, "b-", lw=1.5)
        axes[1].set_xlabel("t"); axes[1].set_ylabel("|PINN − Exact|")
        axes[1].set_title("Absolute Error")
        axes[1].grid(True, alpha=0.3)

        _loss_ax(axes[2], self._loss_hist, self._pde_hist)

        fig.suptitle(f"ODE Harmonic Oscillator  ω={self._omega}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"ode_harmonic{suffix}_solution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {path}")
        return path


# =============================================================================
#  3-D Steady Pipe Flow (Hagen-Poiseuille)
# =============================================================================

class PipeFlowEvaluator(BaseBenchmarkEvaluator):
    name  = "pipe_flow"
    label = "3-D Pipe Flow (Re=10)"
    fast  = False

    def __init__(self, Re: float = 10.0):
        from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE
        from underPINN.geometry.pipe import Pipe
        self._Re           = Re
        self._SteadyNS3DPDE = SteadyNS3DPDE
        self._Pipe          = Pipe

    def train(self, epochs: int, seed: int = 0) -> float:
        R, L, U_max = 0.5, 2.0, 1.0
        pipe     = self._Pipe(R=R, L=L)
        xyz_int  = jnp.array(np.array(pipe.sample_interior(2000), dtype="f4"))
        xyz_wall = jnp.array(np.array(pipe.sample_wall(600),      dtype="f4"))
        xyz_in   = jnp.array(np.array(pipe.sample_inlet(200),     dtype="f4"))
        xyz_out  = jnp.array(np.array(pipe.sample_outlet(200),    dtype="f4"))
        W_PDE, W_WALL, W_IN, W_OUT = 1.0, 100.0, 50.0, 20.0

        model    = MLP(layers=[3, 64, 64, 64, 64, 4])
        pde      = self._SteadyNS3DPDE(model, Re=self._Re)
        key      = jax.random.PRNGKey(seed)
        params   = model.init(key, jnp.ones((1, 3)))
        lr_sched = optax.cosine_decay_schedule(1e-3, epochs, alpha=1e-2)
        optimizer = optax.chain(optax.scale_by_adam(),
                                optax.scale_by_schedule(lr_sched),
                                optax.scale(-1.0))
        state = optimizer.init(params)

        @jax.jit
        def step(params, state, xint, xwall, xin, xout):
            def loss_fn(p):
                cont, mx, my, mz = pde.residual(p, xint)
                pde_l  = jnp.mean(cont**2 + mx**2 + my**2 + mz**2)
                u_w, v_w, w_w, _ = pde.uvwp(p, xwall)
                wall_l = jnp.mean(u_w**2 + v_w**2 + w_w**2)
                r_in   = jnp.sqrt(xin[:, 1]**2 + xin[:, 2]**2)
                u_ex   = U_max * (1 - r_in**2 / R**2)
                u_in, v_in, w_in, _ = pde.uvwp(p, xin)
                in_l   = jnp.mean((u_in - u_ex)**2 + v_in**2 + w_in**2)
                u_out, v_out, w_out, _ = pde.uvwp(p, xout)
                out_l  = jnp.mean(v_out**2 + w_out**2)
                return (W_PDE*pde_l + W_WALL*wall_l + W_IN*in_l + W_OUT*out_l,
                        (pde_l, wall_l, in_l, out_l))
            (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, state = optimizer.update(grads, state)
            params = optax.apply_updates(params, updates)
            return params, state, total, aux

        loss_hist, pde_hist = [], []
        key = jax.random.PRNGKey(seed + 11)
        B, Bw = min(256, xyz_int.shape[0]), min(128, xyz_wall.shape[0])
        Bi, Bo = min(64, xyz_in.shape[0]), min(64, xyz_out.shape[0])
        t0 = time.perf_counter()
        for ep in range(epochs):
            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            ir = jax.random.randint(k1, (B,),  0, xyz_int.shape[0])
            iw = jax.random.randint(k2, (Bw,), 0, xyz_wall.shape[0])
            ii = jax.random.randint(k3, (Bi,), 0, xyz_in.shape[0])
            io = jax.random.randint(k4, (Bo,), 0, xyz_out.shape[0])
            params, state, total, (pl, *_) = step(
                params, state, xyz_int[ir], xyz_wall[iw], xyz_in[ii], xyz_out[io])
            loss_hist.append(float(total))
            pde_hist.append(float(pl))
        wall = time.perf_counter() - t0
        self._model     = model
        self._pde       = pde
        self._params    = params
        self._loss_hist = loss_hist
        self._pde_hist  = pde_hist
        self._R, self._U_max, self._L = R, U_max, L
        return wall

    def evaluate(self) -> dict:
        pipe     = self._Pipe(R=self._R, L=self._L)
        xyz_test = jnp.array(np.array(pipe.sample_interior(2000), dtype="f4"))
        u_e, v_e, w_e, _ = self._pde.exact_poiseuille(
            xyz_test, R=self._R, U_max=self._U_max, L=self._L)
        u_p, v_p, w_p, _ = self._pde.uvwp(self._params, xyz_test)
        speed_pred  = jnp.sqrt(u_p**2 + v_p**2 + w_p**2)
        speed_exact = jnp.sqrt(u_e**2 + v_e**2 + w_e**2)
        return {
            "rel_l2": float(relative_l2_error(speed_pred, speed_exact)),
            "max_ae": float(max_absolute_error(speed_pred, speed_exact)),
        }

    def plot(self, out_dir: str, suffix: str = "") -> str:
        R, U_max = self._R, self._U_max
        # Cross-section at x = L/2
        N = 80
        y_plt = np.linspace(-R, R, N, dtype="f4")
        z_plt = np.linspace(-R, R, N, dtype="f4")
        YY, ZZ = np.meshgrid(y_plt, z_plt, indexing="ij")
        mask   = (YY**2 + ZZ**2) <= R**2
        x_mid  = np.full(N * N, self._L / 2, dtype="f4")
        pts    = jnp.stack([jnp.array(x_mid),
                            jnp.array(YY.ravel()),
                            jnp.array(ZZ.ravel())], axis=1)
        u_pred_flat, _, _, _ = self._pde.uvwp(self._params, pts)
        u_pred  = np.array(u_pred_flat).reshape(N, N)
        r2      = YY**2 + ZZ**2
        u_exact = U_max * (1 - r2 / R**2)
        u_pred[~mask] = np.nan
        u_exact[~mask] = np.nan

        # Radial profile
        r_line = np.linspace(0, R * 0.98, 100, dtype="f4")
        pts_r  = jnp.stack([jnp.full(100, self._L / 2, "f4"),
                            jnp.array(r_line),
                            jnp.zeros(100, "f4")], axis=1)
        u_r_pred, _, _, _ = self._pde.uvwp(self._params, pts_r)
        u_r_exact = U_max * (1 - r_line**2 / R**2)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        # Cross-section: PINN
        im0 = axes[0].pcolormesh(y_plt, z_plt, u_pred.T,
                                  cmap=_CMAP_SOLN, shading="auto")
        fig.colorbar(im0, ax=axes[0]); axes[0].set_aspect("equal")
        axes[0].set_title("PINN  u(y,z) @ x=L/2")
        axes[0].set_xlabel("y"); axes[0].set_ylabel("z")

        # Cross-section: exact
        im1 = axes[1].pcolormesh(y_plt, z_plt, u_exact.T,
                                  cmap=_CMAP_SOLN, shading="auto",
                                  vmin=float(np.nanmin(u_exact)),
                                  vmax=float(np.nanmax(u_exact)))
        fig.colorbar(im1, ax=axes[1]); axes[1].set_aspect("equal")
        axes[1].set_title("Exact  (Hagen-Poiseuille)")
        axes[1].set_xlabel("y"); axes[1].set_ylabel("z")

        # Radial profile
        axes[2].plot(r_line, u_r_exact,  "k-",  lw=2,   label="Exact")
        axes[2].plot(r_line, np.array(u_r_pred), "r--", lw=1.8, label="PINN")
        axes[2].set_xlabel("r"); axes[2].set_ylabel("u")
        axes[2].set_title("Radial Profile u(r)")
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        # Loss
        _loss_ax(axes[3], self._loss_hist, self._pde_hist)

        fig.suptitle(f"3-D Pipe Flow (Re={self._Re})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"pipe_flow{suffix}_solution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {path}")
        return path


# =============================================================================
#  Registry
# =============================================================================

EVALUATOR_REGISTRY: dict[str, type] = {
    "burgers":      BurgersEvaluator,
    "wave":         WaveEvaluator,
    "helmholtz":    HelmholtzEvaluator,
    "heat_steady":  SteadyHeatEvaluator,
    "ode_exp":      ODEExpEvaluator,
    "ode_harmonic": ODEHarmonicEvaluator,
    "pipe_flow":    PipeFlowEvaluator,
}

SLOW_PROBLEMS = {k for k, v in EVALUATOR_REGISTRY.items() if not v.fast}
