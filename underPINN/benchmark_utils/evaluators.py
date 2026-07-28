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

import math
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
    #: Harder-to-converge physics (shocks, viscous SBLI, 3-D N-S) that need a
    #: bigger epoch budget than smooth PDEs (Burgers/wave/heat/ODE) to reach
    #: comparable accuracy — independent of ``fast`` (whether it's cheap
    #: enough per-epoch to run by default at all).
    complex: bool = False

    @abstractmethod
    def train(self, epochs: int, seed: int = 0) -> float:
        """Train for *epochs* steps; return wall time in seconds."""

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Return {'rel_l2', 'max_ae', ...} after train()."""

    @abstractmethod
    def plot(self, out_dir: str, suffix: str = "") -> str:
        """Save solution figure to *out_dir*; return the file path."""

    def plot_pgf(self, out_dir: str, suffix: str = ""):
        """Export the same solution data as PGFPlots ``.dat`` files (see
        :mod:`underPINN.utils.pgf_export`) — not abstract, since a caller
        (:meth:`BenchmarkRunner.run`) already tolerates it being unimplemented
        for any evaluator that hasn't added it yet."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement plot_pgf()")

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
        N_r, N_ic, N_bc = 20000, 200, 300
        T    = 1.5
        x_r  = jnp.array(rng.uniform(-1.0, 1.0, N_r).astype("f4"))
        t_r  = jnp.array(rng.uniform( 0.0,  T,   N_r).astype("f4"))
        x_ic = jnp.array(np.linspace(-1, 1, N_ic, dtype="f4"))
        u_ic = jnp.array(-np.sin(np.pi * x_ic))
        t_bc = rng.uniform(0.0, T, N_bc).astype("f4")
        x_bc = jnp.array(np.tile([-1.0, 1.0], N_bc).astype("f4"))
        t_bc = jnp.array(np.tile(t_bc, 2))
        u_bc = jnp.zeros(2 * N_bc, dtype="f4")

        model  = MLP(layers=[2, 64, 64, 64, 64, 64, 1])
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

    def plot_pgf(self, out_dir: str, suffix: str = "") -> str:
        from underPINN.utils.pgf_export import save_heatmap_png, save_lines_dat
        from underPINN.utils.pgf_tex import field_comparison_tex, loss_tex
        Nx, Nt = 200, 100
        T = self._T
        x_plt = np.linspace(-1, 1, Nx, dtype="f4")
        t_plt = np.linspace(0,  T, Nt, dtype="f4")
        XX, TT = np.meshgrid(x_plt, t_plt, indexing="ij")
        pts    = jnp.stack([jnp.array(XX.ravel()), jnp.array(TT.ravel())], axis=1)
        u_pred = np.array(self._model.apply(self._params, pts)[:, 0]).reshape(Nx, Nt)

        x_ref, t_ref, U_ref = _burgers_reference(self._nu)
        from scipy.interpolate import RegularGridInterpolator
        interp  = RegularGridInterpolator((x_ref, t_ref), U_ref,
                                          method="linear", bounds_error=False,
                                          fill_value=0.0)
        u_exact = interp(np.column_stack([XX.ravel(), TT.ravel()])).reshape(Nx, Nt)
        err = np.abs(u_pred - u_exact)
        vlim = float(max(abs(u_pred.min()), abs(u_pred.max()),
                        abs(u_exact.min()), abs(u_exact.max())))

        pinn_png = os.path.join(out_dir, f"burgers{suffix}_pinn.png")
        exact_png = os.path.join(out_dir, f"burgers{suffix}_exact.png")
        err_png = os.path.join(out_dir, f"burgers{suffix}_err.png")
        ext = save_heatmap_png(pinn_png, x_plt, t_plt, u_pred, "RdBu_r", -vlim, vlim)
        save_heatmap_png(exact_png, x_plt, t_plt, u_exact, "RdBu_r", -vlim, vlim)
        save_heatmap_png(err_png, x_plt, t_plt, err, "Reds", 0.0, float(err.max()))

        fig_path = field_comparison_tex(
            os.path.join(out_dir, f"burgers{suffix}_fields.tex"),
            [
                dict(png_rel=os.path.basename(pinn_png), extent=ext, vmin=-vlim,
                    vmax=vlim, cmap="divRdBu", title="PINN", cbar_label="$u$",
                    xlabel="$x$", ylabel="$t$"),
                dict(png_rel=os.path.basename(exact_png), extent=ext, vmin=-vlim,
                    vmax=vlim, cmap="divRdBu", title="Exact", cbar_label="$u$",
                    xlabel="$x$", ylabel="$t$"),
                dict(png_rel=os.path.basename(err_png), extent=ext, vmin=0.0,
                    vmax=float(err.max()), cmap="seqReds", title="$|$Error$|$",
                    cbar_label="", xlabel="$x$", ylabel="$t$"),
            ])

        loss_path = os.path.join(out_dir, f"burgers{suffix}_loss.dat")
        save_lines_dat(loss_path, epoch=np.arange(1, len(self._loss_hist) + 1),
                       total=self._loss_hist, pde=self._pde_hist)
        loss_tex(os.path.join(out_dir, f"burgers{suffix}_loss.tex"),
                os.path.basename(loss_path), "1-D Burgers training loss")
        return fig_path


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
        N_r, N_ic, N_bc = 20000, 300, 300
        x_r  = jnp.array(rng.uniform(-1, 1, N_r).astype("f4"))
        t_r  = jnp.array(rng.uniform( 0, T, N_r).astype("f4"))
        x_ic = jnp.array(np.linspace(-1, 1, N_ic, dtype="f4"))
        u_ic = jnp.array(np.sin(np.pi * x_ic).astype("f4"))
        t_bc_half = rng.uniform(0, T, N_bc).astype("f4")
        x_bc = jnp.array(np.concatenate([np.full(N_bc, -1., "f4"),
                                          np.full(N_bc,  1., "f4")]))
        t_bc = jnp.array(np.concatenate([t_bc_half, t_bc_half]))

        sigma    = max(2.0, float(c) * np.pi)
        model    = FourierMLP(layers=[2, 64, 64, 64, 64, 64, 1], n_fourier=16, sigma=sigma)
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
                # WavePDE.residual takes a packed (N, 2) array [x, t], per the
                # BasePDE convention — NOT separate x, t positional args.
                res  = pde.residual(p, jnp.stack([xr, tr], axis=1))
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

    def plot_pgf(self, out_dir: str, suffix: str = "") -> str:
        from underPINN.utils.pgf_export import save_heatmap_png, save_lines_dat
        from underPINN.utils.pgf_tex import field_comparison_tex, loss_tex
        Nx, Nt = 200, 100
        T      = self._T
        x_plt  = np.linspace(-1, 1, Nx, dtype="f4")
        t_plt  = np.linspace( 0, T, Nt, dtype="f4")
        XX, TT = np.meshgrid(x_plt, t_plt, indexing="ij")
        pts    = jnp.stack([jnp.array(XX.ravel()), jnp.array(TT.ravel())], axis=1)
        u_pred  = np.array(self._model.apply(self._params, pts)[:, 0]).reshape(Nx, Nt)
        u_exact = np.array(self._pde.exact(
            jnp.array(XX.ravel()), jnp.array(TT.ravel()))).reshape(Nx, Nt)
        err = np.abs(u_pred - u_exact)
        vlim = float(max(abs(u_pred.min()), abs(u_pred.max()),
                        abs(u_exact.min()), abs(u_exact.max())))

        pinn_png = os.path.join(out_dir, f"wave{suffix}_pinn.png")
        exact_png = os.path.join(out_dir, f"wave{suffix}_exact.png")
        err_png = os.path.join(out_dir, f"wave{suffix}_err.png")
        ext = save_heatmap_png(pinn_png, x_plt, t_plt, u_pred, "RdBu_r", -vlim, vlim)
        save_heatmap_png(exact_png, x_plt, t_plt, u_exact, "RdBu_r", -vlim, vlim)
        save_heatmap_png(err_png, x_plt, t_plt, err, "Reds", 0.0, float(err.max()))

        fig_path = field_comparison_tex(
            os.path.join(out_dir, f"wave{suffix}_fields.tex"),
            [
                dict(png_rel=os.path.basename(pinn_png), extent=ext, vmin=-vlim,
                    vmax=vlim, cmap="divRdBu", title="PINN", cbar_label="$u$",
                    xlabel="$x$", ylabel="$t$"),
                dict(png_rel=os.path.basename(exact_png), extent=ext, vmin=-vlim,
                    vmax=vlim, cmap="divRdBu", title="Exact", cbar_label="$u$",
                    xlabel="$x$", ylabel="$t$"),
                dict(png_rel=os.path.basename(err_png), extent=ext, vmin=0.0,
                    vmax=float(err.max()), cmap="seqReds", title="$|$Error$|$",
                    cbar_label="", xlabel="$x$", ylabel="$t$"),
            ])

        loss_path = os.path.join(out_dir, f"wave{suffix}_loss.dat")
        save_lines_dat(loss_path, epoch=np.arange(1, len(self._loss_hist) + 1),
                       total=self._loss_hist)
        loss_tex(os.path.join(out_dir, f"wave{suffix}_loss.tex"),
                os.path.basename(loss_path), "1-D Wave training loss",
                has_pde=False)
        return fig_path


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
        N_r, N_b = 20000, 400
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
        model  = FourierMLP(layers=[2, 64, 64, 64, 64, 64,1], n_fourier=16, sigma=sigma)
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

    def plot_pgf(self, out_dir: str, suffix: str = "") -> str:
        from underPINN.utils.pgf_export import save_heatmap_png, save_lines_dat
        from underPINN.utils.pgf_tex import field_comparison_tex, loss_tex
        N  = 100
        x  = np.linspace(0, 1, N, dtype="f4")
        XY = np.array(np.meshgrid(x, x, indexing="ij")).reshape(2, -1).T
        xy = jnp.array(XY)
        u_pred  = np.array(self._pde.u(self._params, xy)).reshape(N, N)
        u_exact = np.array(self._pde.exact(xy)).reshape(N, N)
        err = np.abs(u_pred - u_exact)
        vlim = float(max(abs(u_pred.min()), abs(u_pred.max()),
                        abs(u_exact.min()), abs(u_exact.max())))

        pinn_png = os.path.join(out_dir, f"helmholtz{suffix}_pinn.png")
        exact_png = os.path.join(out_dir, f"helmholtz{suffix}_exact.png")
        err_png = os.path.join(out_dir, f"helmholtz{suffix}_err.png")
        ext = save_heatmap_png(pinn_png, x, x, u_pred, "RdBu_r", -vlim, vlim)
        save_heatmap_png(exact_png, x, x, u_exact, "RdBu_r", -vlim, vlim)
        save_heatmap_png(err_png, x, x, err, "Reds", 0.0, float(err.max()))

        fig_path = field_comparison_tex(
            os.path.join(out_dir, f"helmholtz{suffix}_fields.tex"),
            [
                dict(png_rel=os.path.basename(pinn_png), extent=ext, vmin=-vlim,
                    vmax=vlim, cmap="divRdBu", title="PINN", cbar_label="$u$"),
                dict(png_rel=os.path.basename(exact_png), extent=ext, vmin=-vlim,
                    vmax=vlim, cmap="divRdBu", title="Exact", cbar_label="$u$"),
                dict(png_rel=os.path.basename(err_png), extent=ext, vmin=0.0,
                    vmax=float(err.max()), cmap="seqReds", title="$|$Error$|$",
                    cbar_label=""),
            ], aspect_equal=True)

        loss_path = os.path.join(out_dir, f"helmholtz{suffix}_loss.dat")
        save_lines_dat(loss_path, epoch=np.arange(1, len(self._loss_hist) + 1),
                       total=self._loss_hist, pde=self._pde_hist)
        loss_tex(os.path.join(out_dir, f"helmholtz{suffix}_loss.tex"),
                os.path.basename(loss_path), "2-D Helmholtz training loss")
        return fig_path


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
        N_r, N_b = 20000, 400
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

        model  = MLP(layers=[2, 64, 64, 64, 64, 64, 1])
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

    def plot_pgf(self, out_dir: str, suffix: str = "") -> tuple:
        from underPINN.utils.pgf_export import save_lines_dat, save_surf_dat
        N  = 100
        x  = np.linspace(0, 1, N, dtype="f4")
        XY = np.array(np.meshgrid(x, x, indexing="ij")).reshape(2, -1).T
        xy = jnp.array(XY)
        u_pred  = np.array(self._pde.u(self._params, xy)).reshape(N, N)
        u_exact = np.array(self._pde.exact(xy)).reshape(N, N)

        rows, cols = save_surf_dat(
            os.path.join(out_dir, f"heat_steady{suffix}_pinn.dat"), x, x, u_pred)
        save_surf_dat(
            os.path.join(out_dir, f"heat_steady{suffix}_exact.dat"), x, x, u_exact)
        save_surf_dat(
            os.path.join(out_dir, f"heat_steady{suffix}_err.dat"), x, x,
            np.abs(u_pred - u_exact))
        save_lines_dat(
            os.path.join(out_dir, f"heat_steady{suffix}_loss.dat"),
            epoch=np.arange(1, len(self._loss_hist) + 1),
            total=self._loss_hist, pde=self._pde_hist)
        print(f"  plot_pgf → {out_dir}/heat_steady{suffix}_*.dat")
        return rows, cols


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
        model  = FourierMLP(layers=[1, 64, 64, 64, 64, 64, 1], n_fourier=16,
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
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("u(t)")
        axes[0].set_title(f"Harmonic  u''+{self._omega}²u=0")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_np, err, "b-", lw=1.5)
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("|PINN − Exact|")
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

    def plot_pgf(self, out_dir: str, suffix: str = "") -> list:
        from underPINN.utils.pgf_export import save_lines_dat
        t_test  = jnp.linspace(0, self._T, 500).reshape(-1, 1).astype("f4")
        u_pred  = np.array(self._pde.u(self._params, t_test)).ravel()
        u_exact = np.array(self._pde.exact(t_test)).ravel()
        t_np    = np.array(t_test).ravel()
        err     = np.abs(u_pred - u_exact)

        names = save_lines_dat(
            os.path.join(out_dir, f"ode_harmonic{suffix}_solution.dat"),
            t=t_np, pinn=u_pred, exact=u_exact, err=err)
        save_lines_dat(
            os.path.join(out_dir, f"ode_harmonic{suffix}_loss.dat"),
            epoch=np.arange(1, len(self._loss_hist) + 1),
            total=self._loss_hist, pde=self._pde_hist)
        print(f"  plot_pgf → {out_dir}/ode_harmonic{suffix}_*.dat")
        return names


# =============================================================================
#  3-D Steady Pipe Flow (Hagen-Poiseuille)
# =============================================================================

class PipeFlowEvaluator(BaseBenchmarkEvaluator):
    name    = "pipe_flow"
    label   = "3-D Pipe Flow (Re=10)"
    fast    = False
    complex = True

    def __init__(self, Re: float = 10.0):
        from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE
        from underPINN.geometry.pipe import Pipe
        self._Re           = Re
        self._SteadyNS3DPDE = SteadyNS3DPDE
        self._Pipe          = Pipe

    def train(self, epochs: int, seed: int = 0) -> float:
        R, L, U_max = 0.5, 2.0, 1.0
        pipe     = self._Pipe(R=R, L=L)
        xyz_int  = jnp.array(np.array(pipe.sample_interior(40000), dtype="f4"))
        xyz_wall = jnp.array(np.array(pipe.sample_wall(600),      dtype="f4"))
        xyz_in   = jnp.array(np.array(pipe.sample_inlet(200),     dtype="f4"))
        xyz_out  = jnp.array(np.array(pipe.sample_outlet(200),    dtype="f4"))
        W_PDE, W_WALL, W_IN, W_OUT = 1.0, 100.0, 50.0, 20.0

        model    = MLP(layers=[3, 128, 128, 128, 128, 128, 4])
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
                # SteadyNS3DPDE.residual returns a packed (N, 4) array
                # [cont, mom_x, mom_y, mom_z], per the BasePDE convention —
                # NOT four separate unpackable values.
                res    = pde.residual(p, xint)
                pde_l  = jnp.mean(jnp.sum(res ** 2, axis=-1))
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
        fig.colorbar(im0, ax=axes[0])
        axes[0].set_aspect("equal")
        axes[0].set_title("PINN  u(y,z) @ x=L/2")
        axes[0].set_xlabel("y")
        axes[0].set_ylabel("z")

        # Cross-section: exact
        im1 = axes[1].pcolormesh(y_plt, z_plt, u_exact.T,
                                  cmap=_CMAP_SOLN, shading="auto",
                                  vmin=float(np.nanmin(u_exact)),
                                  vmax=float(np.nanmax(u_exact)))
        fig.colorbar(im1, ax=axes[1])
        axes[1].set_aspect("equal")
        axes[1].set_title("Exact  (Hagen-Poiseuille)")
        axes[1].set_xlabel("y")
        axes[1].set_ylabel("z")

        # Radial profile
        axes[2].plot(r_line, u_r_exact,  "k-",  lw=2,   label="Exact")
        axes[2].plot(r_line, np.array(u_r_pred), "r--", lw=1.8, label="PINN")
        axes[2].set_xlabel("r")
        axes[2].set_ylabel("u")
        axes[2].set_title("Radial Profile u(r)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Loss
        _loss_ax(axes[3], self._loss_hist, self._pde_hist)

        fig.suptitle(f"3-D Pipe Flow (Re={self._Re})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"pipe_flow{suffix}_solution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {path}")
        return path

    def plot_pgf(self, out_dir: str, suffix: str = "") -> tuple:
        from underPINN.utils.pgf_export import save_lines_dat, save_surf_dat
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

        rows, cols = save_surf_dat(
            os.path.join(out_dir, f"pipe_flow{suffix}_pinn.dat"), y_plt, z_plt, u_pred)
        save_surf_dat(
            os.path.join(out_dir, f"pipe_flow{suffix}_exact.dat"), y_plt, z_plt, u_exact)
        save_surf_dat(
            os.path.join(out_dir, f"pipe_flow{suffix}_err.dat"), y_plt, z_plt,
            np.abs(u_pred - u_exact))

        # Radial profile
        r_line = np.linspace(0, R * 0.98, 100, dtype="f4")
        pts_r  = jnp.stack([jnp.full(100, self._L / 2, "f4"),
                            jnp.array(r_line),
                            jnp.zeros(100, "f4")], axis=1)
        u_r_pred, _, _, _ = self._pde.uvwp(self._params, pts_r)
        u_r_exact = U_max * (1 - r_line**2 / R**2)
        save_lines_dat(
            os.path.join(out_dir, f"pipe_flow{suffix}_radial.dat"),
            r=r_line, pinn=np.array(u_r_pred), exact=u_r_exact)

        save_lines_dat(
            os.path.join(out_dir, f"pipe_flow{suffix}_loss.dat"),
            epoch=np.arange(1, len(self._loss_hist) + 1),
            total=self._loss_hist, pde=self._pde_hist)
        print(f"  plot_pgf → {out_dir}/pipe_flow{suffix}_*.dat")
        return rows, cols


# =============================================================================
#  2-D Compressible Euler — Mach-3 oblique-shock ramp
# =============================================================================

class RampEvaluator(BaseBenchmarkEvaluator):
    """Steady compressible Euler flow over a wedge; exact oblique-shock state.

    Mirrors ``examples/ramp/ramp.py`` (M∞=3, θ=10°, γ=1.4) but with a smaller
    network and collocation pool for benchmark-scale epoch budgets.  The
    reference field is the analytic piecewise-constant oblique-shock solution:
    freestream above the shock line, the exact post-shock state below it.
    """

    name    = "ramp"
    label   = "2-D Ramp — Oblique Shock (M=3, θ=10°)"
    complex = True

    def __init__(self, M_inf: float = 3.0, theta_deg: float = 10.0,
                 gamma: float = 1.4):
        from underPINN.pde.compressible_euler import CompressibleEulerPDE
        from underPINN.geometry.ramp import RampGeometry
        self._M_inf, self._theta_deg, self._gamma = M_inf, theta_deg, gamma
        self._CompressibleEulerPDE = CompressibleEulerPDE
        self._RampGeometry         = RampGeometry

    def train(self, epochs: int, seed: int = 0) -> float:
        M_inf, theta_deg, gamma = self._M_inf, self._theta_deg, self._gamma
        L, H = 1.0, 0.8
        geom = self._RampGeometry(theta_deg, L=L, H=H)

        xy_r  = jnp.array(np.array(geom.sample_interior(30000, seed=seed), "f4"))
        xy_in = jnp.array(np.array(geom.sample_inlet(300),     "f4"))
        xy_w  = jnp.array(np.array(geom.sample_ramp_wall(400), "f4"))
        xy_up = jnp.array(np.array(geom.sample_upper(200),     "f4"))
        nx, ny = geom.ramp_normal()

        model  = MLP(layers=[2, 128, 128, 128, 128, 128, 4])
        pde    = self._CompressibleEulerPDE(model, gamma=gamma, art_visc=1e-3)
        rho_inf, u_inf, v_inf, p_inf = pde.freestream(M_inf)

        key    = jax.random.PRNGKey(seed)
        params = model.init(key, jnp.ones((1, 2)))
        lr_sched  = optax.cosine_decay_schedule(1e-3, epochs, alpha=1e-2)
        optimizer = optax.chain(optax.scale_by_adam(),
                                optax.scale_by_schedule(lr_sched),
                                optax.scale(-1.0))
        state = optimizer.init(params)
        # Loss weights match examples/ramp/config.yaml exactly.
        W_PDE, W_INLET, W_WALL, W_UPPER = 1.0, 200.0, 80.0, 30.0
        N_r, N_in, N_w, N_up = (xy_r.shape[0], xy_in.shape[0],
                                xy_w.shape[0], xy_up.shape[0])
        bR, bI, bW, bU = (min(2048, N_r), min(256, N_in),
                         min(256, N_w), min(200, N_up))

        @jax.jit
        def step(params, state, r_b, in_b, w_b, up_b):
            def loss_fn(p):
                res   = pde.residual(p, r_b)
                pde_l = jnp.mean(jnp.sum(res ** 2, axis=-1))

                pv_in = pde.apply(p, in_b)
                in_l  = (jnp.mean((pv_in[:, 0] - rho_inf) ** 2)
                         + jnp.mean((pv_in[:, 1] - u_inf) ** 2)
                         + jnp.mean((pv_in[:, 2] - v_inf) ** 2)
                         + jnp.mean((pv_in[:, 3] - p_inf) ** 2))

                pv_w  = pde.apply(p, w_b)               # slip:  u·n = 0
                wall_l = jnp.mean((pv_w[:, 1] * nx + pv_w[:, 2] * ny) ** 2)

                pv_up = pde.apply(p, up_b)               # freestream farfield
                up_l  = (jnp.mean((pv_up[:, 0] - rho_inf) ** 2)
                         + jnp.mean((pv_up[:, 1] - u_inf) ** 2)
                         + jnp.mean((pv_up[:, 2] - v_inf) ** 2)
                         + jnp.mean((pv_up[:, 3] - p_inf) ** 2))

                total = W_PDE*pde_l + W_INLET*in_l + W_WALL*wall_l + W_UPPER*up_l
                return total, (pde_l, in_l, wall_l, up_l)
            (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, state = optimizer.update(grads, state)
            params = optax.apply_updates(params, updates)
            return params, state, total, aux

        # RAR-D resampling of the interior pool, matching examples/ramp's
        # rar_period=2000 cadence (scaled to "~5 resamples over the run" so
        # it also does something useful at the smaller benchmark epoch tiers
        # instead of only firing at the full 2000-epoch cadence).
        from underPINN.utils.sampling import rad_resample
        rar_period = max(1, epochs // 5)
        loss_hist, pde_hist = [], []
        key = jax.random.PRNGKey(seed + 11)
        t0 = time.perf_counter()
        for ep in range(epochs):
            if ep > 0 and ep % rar_period == 0:
                xy_r = jnp.array(rad_resample(
                    pde, params, lambda n, s: geom.sample_interior(n, seed=s),
                    n_keep=N_r, n_candidates=5 * N_r,
                    k=1.0, c=1.0, seed=seed + ep))

            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            ir = jax.random.randint(k1, (bR,), 0, N_r)
            ii = jax.random.randint(k2, (bI,), 0, N_in)
            iw = jax.random.randint(k3, (bW,), 0, N_w)
            iu = jax.random.randint(k4, (bU,), 0, N_up)
            params, state, total, (pl, *_) = step(
                params, state, xy_r[ir], xy_in[ii], xy_w[iw], xy_up[iu])
            loss_hist.append(float(total))
            pde_hist.append(float(pl))
        wall = time.perf_counter() - t0

        self._model, self._pde, self._params = model, pde, params
        self._loss_hist, self._pde_hist = loss_hist, pde_hist
        self._geom, self._L, self._H = geom, L, H
        self._shock = pde.oblique_shock(M_inf, theta_deg)
        return wall

    def _exact_field(self, XX, YY):
        """Piecewise-exact Mach number: freestream above the shock line,
        the analytic post-shock state below it."""
        beta = math.radians(self._shock["beta_deg"])
        below_shock = YY <= XX * math.tan(beta)
        M1 = self._M_inf
        M2 = self._shock["M2"]
        return np.where(below_shock, M2, M1)

    def _mach_field(self, params, XX, YY):
        pts  = jnp.array(np.stack([XX.ravel(), YY.ravel()], axis=1), "f4")
        pv   = np.array(self._pde.apply(params, pts))
        a    = np.sqrt(self._gamma * pv[:, 3] / np.maximum(pv[:, 0], 1e-9))
        mach = np.sqrt(pv[:, 1] ** 2 + pv[:, 2] ** 2) / np.maximum(a, 1e-9)
        return mach.reshape(XX.shape)

    def evaluate(self) -> dict:
        XX, YY, mask = self._geom.make_grid(Nx=120, Ny=100)
        mach_pred  = self._mach_field(self._params, XX, YY)
        mach_exact = self._exact_field(XX, YY)
        p_ = jnp.array(mach_pred[mask])
        e_ = jnp.array(mach_exact[mask])
        return {
            "rel_l2": float(relative_l2_error(p_, e_)),
            "max_ae": float(max_absolute_error(p_, e_)),
        }

    def plot(self, out_dir: str, suffix: str = "") -> str:
        XX, YY, mask = self._geom.make_grid(Nx=200, Ny=160)
        mach_pred  = self._mach_field(self._params, XX, YY)
        mach_exact = self._exact_field(XX, YY)
        mach_pred  = mach_pred.copy()
        mach_exact = mach_exact.copy()
        mach_pred[~mask]  = np.nan
        mach_exact[~mask] = np.nan
        x_np, y_np = XX[0, :], YY[:, 0]

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        vmin, vmax = 0.0, self._M_inf + 0.2
        for ax, data, lbl in zip(axes[:2], [mach_pred, mach_exact],
                                 ["PINN", "Exact (piecewise)"]):
            cf = ax.contourf(x_np, y_np, data, levels=50, cmap="jet",
                             vmin=vmin, vmax=vmax)
            fig.colorbar(cf, ax=ax)
            ax.set_title(f"Mach — {lbl}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(0, self._L)
            ax.set_ylim(0, self._H)
        err = np.abs(mach_pred - mach_exact)
        cf2 = axes[2].contourf(x_np, y_np, err, levels=50, cmap=_CMAP_ERR)
        fig.colorbar(cf2, ax=axes[2])
        beta = math.radians(self._shock["beta_deg"])
        x_shock = np.array([0.0, min(self._L, self._H / max(math.tan(beta), 1e-9))])
        y_shock = x_shock * math.tan(beta)
        axes[2].plot(x_shock, y_shock, "k--", lw=1.5, label="shock")
        axes[2].set_title("|Mach error|")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].legend(fontsize=8)

        _loss_ax(axes[3], self._loss_hist, self._pde_hist)
        fig.suptitle(f"Ramp Euler — M∞={self._M_inf}, θ={self._theta_deg}° "
                    f"(β={self._shock['beta_deg']:.1f}°)",
                    fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"ramp{suffix}_solution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {path}")
        return path

    def plot_pgf(self, out_dir: str, suffix: str = "") -> tuple:
        from underPINN.utils.pgf_export import save_lines_dat, save_surf_dat
        XX, YY, mask = self._geom.make_grid(Nx=200, Ny=160)
        mach_pred  = self._mach_field(self._params, XX, YY)
        mach_exact = self._exact_field(XX, YY)
        mach_pred  = mach_pred.copy()
        mach_exact = mach_exact.copy()
        mach_pred[~mask]  = np.nan
        mach_exact[~mask] = np.nan
        err = np.abs(mach_pred - mach_exact)
        # make_grid returns (Ny, Nx); save_surf_dat wants Z as (len(x), len(y))
        x_np, y_np = XX[0, :], YY[:, 0]

        rows, cols = save_surf_dat(
            os.path.join(out_dir, f"ramp{suffix}_pinn.dat"), x_np, y_np, mach_pred.T)
        save_surf_dat(
            os.path.join(out_dir, f"ramp{suffix}_exact.dat"), x_np, y_np, mach_exact.T)
        save_surf_dat(
            os.path.join(out_dir, f"ramp{suffix}_err.dat"), x_np, y_np, err.T)

        beta = math.radians(self._shock["beta_deg"])
        x_shock = np.array([0.0, min(self._L, self._H / max(math.tan(beta), 1e-9))],
                          dtype="f4")
        y_shock = x_shock * math.tan(beta)
        save_lines_dat(
            os.path.join(out_dir, f"ramp{suffix}_shock_line.dat"),
            x=x_shock, y=y_shock)

        save_lines_dat(
            os.path.join(out_dir, f"ramp{suffix}_loss.dat"),
            epoch=np.arange(1, len(self._loss_hist) + 1),
            total=self._loss_hist, pde=self._pde_hist)
        print(f"  plot_pgf → {out_dir}/ramp{suffix}_*.dat")
        return rows, cols


# =============================================================================
#  1-D Unsteady Compressible Euler — Toro test 3 (severe blast wave)
# =============================================================================

class Toro3Evaluator(BaseBenchmarkEvaluator):
    """1-D Riemann problem — Toro test 3 (Woodward–Colella blast wave).

    Left (1, 0, 1000) / right (1, 0, 0.01), γ=1.4 — a five-decade pressure
    jump.  Mirrors ``examples/toro3/toro3.py``: reference-state
    non-dimensionalisation + log-space (exp) positivity for the extreme
    dynamic range, evaluated against the exact Riemann solution.
    """

    name    = "toro3"
    label   = "1-D Toro Test 3 (blast wave)"
    complex = True

    def __init__(self, gamma: float = 1.4):
        from underPINN.pde.euler_1d_unsteady import Euler1DUnsteadyPDE
        from underPINN.utils.riemann import exact_riemann_1d
        self._gamma = gamma
        self._Euler1DUnsteadyPDE = Euler1DUnsteadyPDE
        self._exact_riemann_1d   = exact_riemann_1d

    def train(self, epochs: int, seed: int = 0) -> float:
        gamma = self._gamma
        x0, t_final = 0.5, 0.012
        left, right = (1.0, 0.0, 1000.0), (1.0, 0.0, 0.01)

        rho_ref = max(left[0], right[0])
        p_ref   = max(left[2], right[2])
        u_ref   = float(np.sqrt(p_ref / rho_ref))
        t_ref   = 1.0 / u_ref                     # L = 1
        left_nd  = (left[0] / rho_ref,  left[1] / u_ref,  left[2] / p_ref)
        right_nd = (right[0] / rho_ref, right[1] / u_ref, right[2] / p_ref)
        tf_nd = t_final / t_ref
        self._nd = (rho_ref, u_ref, p_ref, t_ref)
        self._x0, self._t_final = x0, t_final
        self._left, self._right = left, right

        rng   = np.random.default_rng(seed)
        n_int, n_ic, n_bc = 40000, 5000, 3000
        xt_r = np.stack([rng.uniform(0.0, 1.0,  n_int),
                         rng.uniform(0.0, tf_nd, n_int)], axis=1).astype("f4")
        x_ic = rng.uniform(0.0, 1.0, n_ic).astype("f4")
        le   = x_ic < x0
        rho_ic = np.where(le, left_nd[0], right_nd[0]).astype("f4")
        u_ic   = np.where(le, left_nd[1], right_nd[1]).astype("f4")
        p_ic   = np.where(le, left_nd[2], right_nd[2]).astype("f4")
        xt_ic  = np.stack([x_ic, np.zeros(n_ic, "f4")], axis=1)
        t_bc   = rng.uniform(0.0, tf_nd, n_bc).astype("f4")
        xt_bcL = np.stack([np.zeros(n_bc, "f4"), t_bc], axis=1)
        xt_bcR = np.stack([np.ones(n_bc, "f4"),  t_bc], axis=1)

        xt_r_j, xt_ic_j = jnp.array(xt_r), jnp.array(xt_ic)
        ic_tgt = jnp.array(np.stack([rho_ic, u_ic, p_ic], axis=1))
        xt_bcL_j, xt_bcR_j = jnp.array(xt_bcL), jnp.array(xt_bcR)
        bcL_tgt = jnp.array(np.array(left_nd,  "f4"))
        bcR_tgt = jnp.array(np.array(right_nd, "f4"))

        # Fixed (non-trainable) artificial viscosity at the example's vetted
        # value -- the example config runs with trainable_visc: false and
        # art_visc: 0.001; passing plain net params (no "log_av" key) makes
        # Euler1DUnsteadyPDE use self.art_visc as a fixed constant instead of
        # treating it as trainable (see Euler1DUnsteadyPDE._is_combined).
        model = MLP(layers=[2, 128, 128, 128, 128, 128, 3])
        pde   = self._Euler1DUnsteadyPDE(model, gamma=gamma, art_visc=0.001,
                                         transform="exp")
        key    = jax.random.PRNGKey(seed)
        params = model.init(key, jnp.ones((1, 2)))

        lr_sched  = optax.cosine_decay_schedule(1e-3, epochs, alpha=1e-2)
        optimizer = optax.chain(optax.scale_by_adam(),
                                optax.scale_by_schedule(lr_sched),
                                optax.scale(-1.0))
        state = optimizer.init(params)
        W_PDE, W_IC, W_BC = 1.0, 100.0, 10.0  # w_ic matches the example (100.0)
        N_r, N_ic_, N_bc_ = xt_r_j.shape[0], xt_ic_j.shape[0], xt_bcL_j.shape[0]
        bR, bI, bB = min(2048, N_r), min(400, N_ic_), min(300, N_bc_)

        @jax.jit
        def step(params, state, r_b, ic_b, ic_t, bcL_b, bcR_b):
            def loss_fn(p):
                res   = pde.residual(p, r_b)
                pde_l = jnp.mean(jnp.sum(res ** 2, axis=-1))
                out_ic = pde.apply(p, ic_b)
                ic_l   = jnp.mean(jnp.sum((out_ic - ic_t) ** 2, axis=-1))
                out_l  = pde.apply(p, bcL_b)
                out_r  = pde.apply(p, bcR_b)
                bc_l   = (jnp.mean(jnp.sum((out_l - bcL_tgt) ** 2, axis=-1))
                          + jnp.mean(jnp.sum((out_r - bcR_tgt) ** 2, axis=-1)))
                total = W_PDE*pde_l + W_IC*ic_l + W_BC*bc_l
                return total, (pde_l, ic_l, bc_l)
            (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, state = optimizer.update(grads, state)
            params = optax.apply_updates(params, updates)
            return params, state, total, aux

        loss_hist, pde_hist = [], []
        key = jax.random.PRNGKey(seed + 7)
        t0 = time.perf_counter()
        for ep in range(epochs):
            key, k1, k2, k3 = jax.random.split(key, 4)
            ir = jax.random.randint(k1, (bR,), 0, N_r)
            ii = jax.random.randint(k2, (bI,), 0, N_ic_)
            ib = jax.random.randint(k3, (bB,), 0, N_bc_)
            params, state, total, (pl, *_) = step(
                params, state, xt_r_j[ir], xt_ic_j[ii], ic_tgt[ii],
                xt_bcL_j[ib], xt_bcR_j[ib])
            loss_hist.append(float(total))
            pde_hist.append(float(pl))
        wall = time.perf_counter() - t0

        self._model, self._pde, self._params = model, pde, params
        self._loss_hist, self._pde_hist = loss_hist, pde_hist
        self._tf_nd = tf_nd
        return wall

    def _fields(self, Nx=400):
        rho_ref, u_ref, p_ref, t_ref = self._nd
        xg = np.linspace(0.0, 1.0, Nx, dtype="f4")
        pts = jnp.array(np.stack([xg, np.full(Nx, self._tf_nd, "f4")], axis=1))
        pv_nd = np.array(self._pde.apply(self._params, pts))
        re_nd, ue_nd, pe_nd = self._exact_riemann_1d(
            xg, self._tf_nd, self._x0, self._gamma,
            (self._left[0] / rho_ref,  self._left[1] / u_ref,  self._left[2] / p_ref),
            (self._right[0] / rho_ref, self._right[1] / u_ref, self._right[2] / p_ref))
        return xg, pv_nd, np.stack([re_nd, ue_nd, pe_nd], axis=1)

    def evaluate(self) -> dict:
        _, pv_nd, exact_nd = self._fields()
        return {
            "rel_l2": float(relative_l2_error(jnp.array(pv_nd),
                                              jnp.array(exact_nd))),
            "max_ae": float(max_absolute_error(jnp.array(pv_nd),
                                               jnp.array(exact_nd))),
        }

    def plot(self, out_dir: str, suffix: str = "") -> str:
        rho_ref, u_ref, p_ref, t_ref = self._nd
        xg, pv_nd, exact_nd = self._fields()
        rho_p, u_p, p_p = pv_nd[:, 0]*rho_ref, pv_nd[:, 1]*u_ref, pv_nd[:, 2]*p_ref
        rho_e, u_e, p_e = (exact_nd[:, 0]*rho_ref, exact_nd[:, 1]*u_ref,
                          exact_nd[:, 2]*p_ref)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        for ax, yp, ye, name in zip(axes[:3], [rho_p, u_p, p_p],
                                    [rho_e, u_e, p_e], ["ρ", "u", "p"]):
            ax.plot(xg, ye, "k-",  lw=2.0, label="Exact")
            ax.plot(xg, yp, "r--", lw=1.6, label="PINN")
            ax.set_xlabel("x")
            ax.set_ylabel(name)
            ax.set_title(f"{name}  (t={self._t_final})")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        _loss_ax(axes[3], self._loss_hist, self._pde_hist)
        fig.suptitle("Toro Test 3 — blast wave (5-decade pressure jump)",
                    fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"toro3{suffix}_solution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {path}")
        return path

    def plot_pgf(self, out_dir: str, suffix: str = "") -> list:
        from underPINN.utils.pgf_export import save_lines_dat
        rho_ref, u_ref, p_ref, t_ref = self._nd
        xg, pv_nd, exact_nd = self._fields()
        rho_p, u_p, p_p = pv_nd[:, 0]*rho_ref, pv_nd[:, 1]*u_ref, pv_nd[:, 2]*p_ref
        rho_e, u_e, p_e = (exact_nd[:, 0]*rho_ref, exact_nd[:, 1]*u_ref,
                          exact_nd[:, 2]*p_ref)

        names = save_lines_dat(
            os.path.join(out_dir, f"toro3{suffix}_solution.dat"),
            x=xg, rho_pinn=rho_p, rho_exact=rho_e,
            u_pinn=u_p, u_exact=u_e, p_pinn=p_p, p_exact=p_e)
        save_lines_dat(
            os.path.join(out_dir, f"toro3{suffix}_loss.dat"),
            epoch=np.arange(1, len(self._loss_hist) + 1),
            total=self._loss_hist, pde=self._pde_hist)
        print(f"  plot_pgf → {out_dir}/toro3{suffix}_*.dat")
        return names


# =============================================================================
#  2-D Compressible Navier–Stokes — viscous Mach-3 compression ramp (SBLI)
# =============================================================================

class RampNSEvaluator(BaseBenchmarkEvaluator):
    """Viscous compression-ramp SBLI; mirrors ``examples/ramp_ns/ramp_ns.py``.

    Supersonic M∞=3 flow over a flat wall then a θ=15° compression ramp, with
    a short slip run (no-penetration only) before a no-slip + isothermal
    (T=T₀) wall — the same slip/no-slip split and hybrid uniform + wall-
    clustered collocation used in the full example, at benchmark scale.

    Reference: the analytic inviscid oblique-shock Mach state (freestream
    upstream of the corner shock, the exact post-shock state downstream of
    it), evaluated only in the **outer** flow — a thin near-wall band is
    excluded from the metric since the boundary layer there is real viscous
    physics an inviscid reference cannot capture.

    Uses second-order AD (viscous stresses require gradients of gradients),
    comparable in cost to the 3-D pipe flow case, so it is marked ``fast=False``
    (run with ``--all``).
    """

    name    = "ramp_ns"
    label   = "2-D Ramp NS — Viscous SBLI (M=3, Re=1e4)"
    fast    = False
    complex = True

    def __init__(self, M_inf: float = 3.0, theta_deg: float = 15.0,
                 gamma: float = 1.4, Re: float = 1.0e4, Pr: float = 0.72):
        from underPINN.pde.compressible_ns_2d import CompressibleNS2DPDE
        from underPINN.pde.compressible_euler import CompressibleEulerPDE
        from underPINN.geometry.ramp import RampGeometry
        self._M_inf, self._theta_deg = M_inf, theta_deg
        self._gamma, self._Re, self._Pr = gamma, Re, Pr
        self._CompressibleNS2DPDE = CompressibleNS2DPDE
        self._CompressibleEulerPDE = CompressibleEulerPDE
        self._RampGeometry = RampGeometry

    def train(self, epochs: int, seed: int = 0) -> float:
        M_inf, theta_deg, gamma, Re, Pr = (
            self._M_inf, self._theta_deg, self._gamma, self._Re, self._Pr)
        L, H, ramp_start, slip_end = 2.0, 1.0, 0.8, 0.15
        geom = self._RampGeometry(theta_deg, L=L, H=H,
                                  ramp_start=ramp_start, slip_end=slip_end)

        # Hybrid interior pool, matching the full example: fixed uniform base
        # + wall-clustered (BL) points + a residual-ADAPTIVE pool that gets
        # RAR-D-migrated toward the shock every rar_period epochs. x_min
        # excludes the inlet/wall corner (a geometric BC singularity with a
        # far larger raw residual than the shock -- see sample_interior's
        # docstring) so RAR's budget lands on the shock, not the corner.
        from underPINN.utils.sampling import rad_resample
        n_adapt   = 6000
        rar_x_min = 0.25
        xy_uniform = geom.sample_interior(40000, seed=seed)
        xy_bl      = geom.sample_boundary_layer(5000, beta=4.0, seed=seed + 7)
        xy_adapt   = geom.sample_interior(n_adapt, seed=seed + 101, x_min=rar_x_min)
        xy_adapt_init = np.array(xy_adapt)
        xy_r  = jnp.array(np.concatenate([xy_uniform, xy_bl, xy_adapt], axis=0))
        xy_in   = jnp.array(np.array(geom.sample_inlet(200),        "f4"))
        xy_w    = jnp.array(np.array(geom.sample_noslip_wall(200),  "f4"))
        xy_slip = jnp.array(np.array(geom.sample_slip_wall(100),    "f4"))
        xy_up   = jnp.array(np.array(geom.sample_upper(150),        "f4"))

        model = MLP(layers=[2, 128, 128, 128, 128, 128, 4])
        pde   = self._CompressibleNS2DPDE(model, gamma=gamma, M_inf=M_inf,
                                          Re=Re, Pr=Pr, art_visc=2e-3)
        T0 = pde.total_temperature()
        rho_inf, u_inf, v_inf, T_inf = pde.freestream()

        key    = jax.random.PRNGKey(seed)
        params = model.init(key, jnp.ones((1, 2)))
        lr_sched  = optax.cosine_decay_schedule(1e-3, epochs, alpha=1e-2)
        optimizer = optax.chain(optax.scale_by_adam(),
                                optax.scale_by_schedule(lr_sched),
                                optax.scale(-1.0))
        state = optimizer.init(params)
        W_PDE, W_INLET, W_WALL, W_SLIP, W_UPPER = 1.0, 100.0, 100.0, 80.0, 20.0
        N_r, N_in, N_w = xy_r.shape[0], xy_in.shape[0], xy_w.shape[0]
        N_slip, N_up   = xy_slip.shape[0], xy_up.shape[0]
        bR  = min(1536, N_r)
        bI  = min(250, N_in)
        bW  = min(250, N_w)
        bS  = min(180, N_slip)
        bU  = min(180, N_up)

        @jax.jit
        def step(params, state, r_b, in_b, w_b, slip_b, up_b):
            def loss_fn(p):
                res   = pde.residual(p, r_b)
                pde_l = jnp.mean(jnp.sum(res ** 2, axis=-1))

                pv_in = pde.apply(p, in_b)
                in_l  = (jnp.mean((pv_in[:, 0] - rho_inf) ** 2)
                         + jnp.mean((pv_in[:, 1] - u_inf) ** 2)
                         + jnp.mean((pv_in[:, 2] - v_inf) ** 2)
                         + jnp.mean((pv_in[:, 3] - T_inf) ** 2))

                pv_w  = pde.apply(p, w_b)               # no-slip + isothermal
                wall_l = (jnp.mean(pv_w[:, 1] ** 2)
                          + jnp.mean(pv_w[:, 2] ** 2)
                          + jnp.mean((pv_w[:, 3] - T0) ** 2))

                pv_s  = pde.apply(p, slip_b)            # slip: v = 0 only
                slip_l = jnp.mean(pv_s[:, 2] ** 2)

                pv_up = pde.apply(p, up_b)               # freestream farfield
                up_l  = (jnp.mean((pv_up[:, 0] - rho_inf) ** 2)
                         + jnp.mean((pv_up[:, 1] - u_inf) ** 2)
                         + jnp.mean((pv_up[:, 2] - v_inf) ** 2)
                         + jnp.mean((pv_up[:, 3] - T_inf) ** 2))

                total = (W_PDE*pde_l + W_INLET*in_l + W_WALL*wall_l
                         + W_SLIP*slip_l + W_UPPER*up_l)
                return total, (pde_l, in_l, wall_l, slip_l, up_l)
            (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, state = optimizer.update(grads, state)
            params = optax.apply_updates(params, updates)
            return params, state, total, aux

        rar_period = max(1, epochs // 5)  # 5 resamples over the whole run
        loss_hist, pde_hist = [], []
        key = jax.random.PRNGKey(seed + 11)
        t0 = time.perf_counter()
        for ep in range(epochs):
            if ep > 0 and ep % rar_period == 0:
                xy_adapt = rad_resample(
                    pde, params,
                    lambda n, s: geom.sample_interior(n, seed=s, x_min=rar_x_min),
                    n_keep=n_adapt, n_candidates=5 * n_adapt,
                    k=1.0, c=1.0, seed=seed + ep)
                xy_r = jnp.array(np.concatenate(
                    [xy_uniform, xy_bl, xy_adapt], axis=0))

            key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
            ir  = jax.random.randint(k1, (bR,), 0, N_r)
            ii  = jax.random.randint(k2, (bI,), 0, N_in)
            iw  = jax.random.randint(k3, (bW,), 0, N_w)
            isl = jax.random.randint(k4, (bS,), 0, N_slip)
            iu  = jax.random.randint(k5, (bU,), 0, N_up)
            params, state, total, (pl, *_) = step(
                params, state, xy_r[ir], xy_in[ii], xy_w[iw],
                xy_slip[isl], xy_up[iu])
            loss_hist.append(float(total))
            pde_hist.append(float(pl))
        wall = time.perf_counter() - t0

        self._model, self._pde, self._params = model, pde, params
        self._loss_hist, self._pde_hist = loss_hist, pde_hist
        self._geom, self._L, self._H = geom, L, H
        self._ramp_start = ramp_start
        self._xy_adapt_init  = xy_adapt_init
        self._xy_adapt_final = np.array(xy_adapt)
        # Analytic inviscid oblique-shock state, via a model-free helper PDE
        # (its oblique_shock()/freestream() are pure closed-form calculations).
        euler = self._CompressibleEulerPDE(None, gamma=gamma)
        self._shock = euler.oblique_shock(M_inf, theta_deg)
        return wall

    def _exact_field(self, XX, YY):
        """Piecewise-exact outer Mach: freestream upstream of the corner shock,
        the analytic post-shock state below the shock line downstream of it."""
        beta = math.radians(self._shock["beta_deg"])
        dx = np.maximum(XX - self._ramp_start, 0.0)
        below_shock = (YY <= dx * math.tan(beta)) & (XX >= self._ramp_start)
        return np.where(below_shock, self._shock["M2"], self._M_inf)

    def _mach_field(self, params, XX, YY):
        pts  = jnp.array(np.stack([XX.ravel(), YY.ravel()], axis=1), "f4")
        mach = np.array(self._pde.mach(params, pts))
        return mach.reshape(XX.shape)

    def _outer_mask(self, XX, YY, mask, band_frac=0.12):
        """Domain-interior mask, excluding a near-wall band (boundary layer)."""
        band = band_frac * self._H
        return mask & (YY > self._geom.y_wall(XX) + band)

    def evaluate(self) -> dict:
        XX, YY, mask = self._geom.make_grid(Nx=140, Ny=110)
        outer = self._outer_mask(XX, YY, mask)
        mach_pred  = self._mach_field(self._params, XX, YY)
        mach_exact = self._exact_field(XX, YY)
        p_ = jnp.array(mach_pred[outer])
        e_ = jnp.array(mach_exact[outer])
        return {
            "rel_l2": float(relative_l2_error(p_, e_)),
            "max_ae": float(max_absolute_error(p_, e_)),
        }

    def plot(self, out_dir: str, suffix: str = "") -> str:
        XX, YY, mask = self._geom.make_grid(Nx=200, Ny=160)
        outer = self._outer_mask(XX, YY, mask)
        mach_pred  = self._mach_field(self._params, XX, YY)
        mach_exact = self._exact_field(XX, YY)

        mach_full = mach_pred.copy()
        mach_full[~mask] = np.nan
        mach_outer_exact = mach_exact.copy()
        mach_outer_exact[~outer] = np.nan
        err = np.abs(mach_pred - mach_exact)
        err[~outer] = np.nan
        x_np, y_np = XX[0, :], YY[:, 0]

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        vmin, vmax = 0.0, self._M_inf + 0.2
        cf0 = axes[0].contourf(x_np, y_np, mach_full, levels=50, cmap="jet",
                               vmin=vmin, vmax=vmax)
        fig.colorbar(cf0, ax=axes[0])
        axes[0].set_title("Mach — PINN (full field)")

        cf1 = axes[1].contourf(x_np, y_np, mach_outer_exact, levels=50,
                               cmap="jet", vmin=vmin, vmax=vmax)
        fig.colorbar(cf1, ax=axes[1])
        axes[1].set_title("Mach — exact outer (inviscid)")

        cf2 = axes[2].contourf(x_np, y_np, err, levels=50, cmap=_CMAP_ERR)
        fig.colorbar(cf2, ax=axes[2])
        beta = math.radians(self._shock["beta_deg"])
        x_shock = np.array([self._ramp_start,
                            min(self._L, self._ramp_start
                                + self._H / max(math.tan(beta), 1e-9))])
        y_shock = (x_shock - self._ramp_start) * math.tan(beta)
        axes[2].plot(x_shock, y_shock, "k--", lw=1.5, label="shock")
        axes[2].set_title("|Mach error|  (outer flow only)")
        axes[2].legend(fontsize=8)

        for ax in axes[:3]:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(0, self._L)
            ax.set_ylim(0, self._H)

        _loss_ax(axes[3], self._loss_hist, self._pde_hist)
        fig.suptitle(f"Ramp NS (SBLI) — M∞={self._M_inf}, θ={self._theta_deg}°, "
                    f"Re={self._Re:g}  (β={self._shock['beta_deg']:.1f}°)",
                    fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"ramp_ns{suffix}_solution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {path}")
        return path

    def plot_pgf(self, out_dir: str, suffix: str = "") -> tuple:
        from underPINN.utils.pgf_export import save_lines_dat, save_surf_dat
        XX, YY, mask = self._geom.make_grid(Nx=200, Ny=160)
        outer = self._outer_mask(XX, YY, mask)
        mach_pred  = self._mach_field(self._params, XX, YY)
        mach_exact = self._exact_field(XX, YY)

        mach_full = mach_pred.copy()
        mach_full[~mask] = np.nan
        mach_outer_exact = mach_exact.copy()
        mach_outer_exact[~outer] = np.nan
        err = np.abs(mach_pred - mach_exact)
        err[~outer] = np.nan
        # make_grid returns (Ny, Nx); save_surf_dat wants Z as (len(x), len(y))
        x_np, y_np = XX[0, :], YY[:, 0]

        rows, cols = save_surf_dat(
            os.path.join(out_dir, f"ramp_ns{suffix}_pinn.dat"), x_np, y_np, mach_full.T)
        save_surf_dat(
            os.path.join(out_dir, f"ramp_ns{suffix}_exact.dat"), x_np, y_np,
            mach_outer_exact.T)
        save_surf_dat(
            os.path.join(out_dir, f"ramp_ns{suffix}_err.dat"), x_np, y_np, err.T)

        beta = math.radians(self._shock["beta_deg"])
        x_shock = np.array([self._ramp_start,
                           min(self._L, self._ramp_start
                               + self._H / max(math.tan(beta), 1e-9))], dtype="f4")
        y_shock = (x_shock - self._ramp_start) * math.tan(beta)
        save_lines_dat(
            os.path.join(out_dir, f"ramp_ns{suffix}_shock_line.dat"),
            x=x_shock, y=y_shock)

        save_lines_dat(
            os.path.join(out_dir, f"ramp_ns{suffix}_loss.dat"),
            epoch=np.arange(1, len(self._loss_hist) + 1),
            total=self._loss_hist, pde=self._pde_hist)

        save_lines_dat(
            os.path.join(out_dir, f"ramp_ns{suffix}_rar_init.dat"),
            x=self._xy_adapt_init[:, 0], y=self._xy_adapt_init[:, 1])
        save_lines_dat(
            os.path.join(out_dir, f"ramp_ns{suffix}_rar_final.dat"),
            x=self._xy_adapt_final[:, 0], y=self._xy_adapt_final[:, 1])
        print(f"  plot_pgf → {out_dir}/ramp_ns{suffix}_*.dat")
        return rows, cols


# =============================================================================
#  Registry
# =============================================================================

EVALUATOR_REGISTRY: dict[str, type] = {
    "burgers":      BurgersEvaluator,
    "wave":         WaveEvaluator,
    "helmholtz":    HelmholtzEvaluator,
    "heat_steady":  SteadyHeatEvaluator,
    "ode_harmonic": ODEHarmonicEvaluator,
    "ramp":         RampEvaluator,
    "toro3":        Toro3Evaluator,
    "pipe_flow":    PipeFlowEvaluator,
    "ramp_ns":      RampNSEvaluator,
}

SLOW_PROBLEMS = {k for k, v in EVALUATOR_REGISTRY.items() if not v.fast}
