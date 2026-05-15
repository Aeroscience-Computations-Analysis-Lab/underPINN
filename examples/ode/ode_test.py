"""ODE test cases — Exponential Decay + Harmonic Oscillator PINNs.

Run directly or via the CLI:

    python examples/ode/ode_test.py                  # uses config.yaml
    python examples/ode/ode_test.py myconfig.yaml    # custom config
    python -m underPINN run examples/ode/config.yaml

Two ODE problems are solved in sequence:
  1. Exponential Decay:    du/dt + λu = 0,  u(0) = u0
  2. Harmonic Oscillator:  d²u/dt² + ω²u = 0,  u(0) = u0,  u'(0) = v0
"""
from __future__ import annotations

import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.config.loader import cfg_get, save_config
from underPINN.nn.mlp import MLP
from underPINN.pde.ode import ExponentialDecayODE, HarmonicOscillatorODE
from underPINN.losses.ode_loss import ODELoss
from underPINN.solver.ode_solver import ODESolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions


def run_ode(cfg) -> dict:
    """Train PINNs on both ODE test cases defined in *cfg*.

    Runs exponential-decay first, then harmonic-oscillator. Both sets of
    outputs (loss plots, solution plots, predictions .npz) are written to
    ``output.dir``.
    """
    # ── Unpack shared config ──────────────────────────────────────────────────
    tr      = cfg.training
    seed    = cfg_get(tr, "seed",    default=0)
    out     = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/ode") if out else "outputs/ode"
    os.makedirs(out_dir, exist_ok=True)

    epochs    = tr.epochs
    lr        = tr.lr
    lr_alpha  = cfg_get(tr, "lr_alpha",  default=0.01)
    log_every = cfg_get(tr, "log_every", default=500)
    patience  = cfg_get(tr, "early_stopping_patience", default=200)
    n_col     = cfg_get(cfg.data, "n_collocation", default=3000)
    layers    = cfg.network.layers

    results: dict = {}

    # ── Helper to build TrainingConfig ────────────────────────────────────────
    def _make_config(ep: int) -> TrainingConfig:
        return TrainingConfig(
            epochs=ep,
            lr=lr,
            lr_schedule=optax.cosine_decay_schedule(lr, decay_steps=ep, alpha=lr_alpha),
            seed=seed,
            log_every=log_every,
            callbacks=[
                ConsoleLogger(log_every=log_every),
                EarlyStopping(patience=patience),
            ],
        )

    # ── 1. Exponential Decay  du/dt + λu = 0 ─────────────────────────────────
    exp_cfg = cfg_get(cfg, "exponential_decay", default=None)
    if exp_cfg is not None:
        print("\n" + "=" * 60)
        print("Exponential Decay  du/dt + λu = 0")
        print("=" * 60)

        lam = float(cfg_get(exp_cfg, "lambda", default=1.0))
        T   = float(cfg_get(exp_cfg, "T",      default=5.0))
        u0  = float(cfg_get(exp_cfg, "u0",     default=1.0))

        t_r  = jnp.linspace(0.0, T, n_col)
        t_ic = jnp.array([0.0])
        u_ic = jnp.array([u0])

        model  = MLP(layers=layers)
        pde    = ExponentialDecayODE(model, lam=lam)
        loss   = ODELoss(model, pde, ic_weight=100.0)
        solver = ODESolver(model, pde, loss)
        solver.init(jax.random.PRNGKey(seed))
        solver.train(t_r, t_ic, u_ic, config=_make_config(epochs))

        t_test  = jnp.linspace(0.0, T, 1000)
        u_pred  = pde.u(solver.params, t_test)
        u_exact = pde.exact(t_test)
        rel_l2  = float(jnp.linalg.norm(u_pred - u_exact) / (jnp.linalg.norm(u_exact) + 1e-10))
        print(f"  Exponential Decay — Rel-L2: {rel_l2:.4e}")
        results["exp_decay_rel_l2"] = rel_l2

        # Save checkpoint
        solver.save_checkpoint(out_dir, stem="params_exp_decay", metadata={
            "problem": "ode", "ode": "exponential_decay",
            "network": {"type": "mlp", "layers": list(layers)},
            "physics": {"lambda": lam, "u0": u0, "T": T},
            "rel_l2": rel_l2,
        })

        # Save predictions
        save_predictions(
            out_dir,
            coords  = {"t": np.array(t_r)},
            outputs = {"u_pred": np.array(pde.u(solver.params, t_r))},
            exact   = {"u_exact": np.array(pde.exact(t_r))},
            filename="predictions_exp_decay.npz",
        )

        # Loss + solution plot
        np.save(os.path.join(out_dir, "loss_exp_decay.npy"),
                np.array(solver.loss_hist))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(np.array(t_test), np.array(u_exact), label="Exact",  lw=1.5)
        axes[0].plot(np.array(t_test), np.array(u_pred),  label="PINN", lw=1.5, ls="--")
        axes[0].set_xlabel("t"); axes[0].legend()
        axes[0].set_title(f"Exp Decay  λ={lam}  (Rel-L2={rel_l2:.2e})")
        axes[1].semilogy(solver.loss_hist,  label="Total",  lw=1.2)
        axes[1].semilogy(solver.pde_hist,   label="PDE",    alpha=0.7)
        axes[1].semilogy(solver.ic_hist,    label="IC",     alpha=0.7)
        axes[1].set_xlabel("Epoch"); axes[1].legend()
        axes[1].set_title("Training loss")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "exp_decay.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 2. Harmonic Oscillator  d²u/dt² + ω²u = 0 ───────────────────────────
    ho_cfg = cfg_get(cfg, "harmonic_oscillator", default=None)
    if ho_cfg is not None:
        print("\n" + "=" * 60)
        print("Harmonic Oscillator  d²u/dt² + ω²u = 0")
        print("=" * 60)

        omega = float(cfg_get(ho_cfg, "omega", default=2.0))
        T     = float(cfg_get(ho_cfg, "T",     default=10.0))
        u0    = float(cfg_get(ho_cfg, "u0",    default=1.0))
        v0    = float(cfg_get(ho_cfg, "v0",    default=0.0))

        t_r      = jnp.linspace(0.0, T, n_col)
        t_ic     = jnp.array([0.0])
        u_ic     = jnp.array([u0])
        u_ic_dot = jnp.array([v0])

        # Use more epochs for the harder harmonic problem
        ho_epochs = cfg_get(tr, "harmonic_epochs", default=epochs * 4)

        model  = MLP(layers=layers)
        pde    = HarmonicOscillatorODE(model, omega=omega)
        loss   = ODELoss(model, pde, ic_weight=100.0, ic_derivative_weight=100.0)
        solver = ODESolver(model, pde, loss)
        solver.init(jax.random.PRNGKey(seed + 1))
        solver.train(t_r, t_ic, u_ic, u_ic_dot=u_ic_dot, config=_make_config(ho_epochs))

        t_test  = jnp.linspace(0.0, T, 2000)
        u_pred  = pde.u(solver.params, t_test)
        u_exact = pde.exact(t_test)
        rel_l2  = float(jnp.linalg.norm(u_pred - u_exact) / (jnp.linalg.norm(u_exact) + 1e-10))
        print(f"  Harmonic Oscillator — Rel-L2: {rel_l2:.4e}")
        results["harmonic_rel_l2"] = rel_l2

        # Save checkpoint
        solver.save_checkpoint(out_dir, stem="params_harmonic", metadata={
            "problem": "ode", "ode": "harmonic_oscillator",
            "network": {"type": "mlp", "layers": list(layers)},
            "physics": {"omega": omega, "T": T},
            "rel_l2": rel_l2,
        })

        # Save predictions
        save_predictions(
            out_dir,
            coords  = {"t": np.array(t_r)},
            outputs = {"u_pred": np.array(pde.u(solver.params, t_r))},
            exact   = {"u_exact": np.array(pde.exact(t_r))},
            filename="predictions_harmonic.npz",
        )

        # Loss + solution plot
        np.save(os.path.join(out_dir, "loss_harmonic.npy"),
                np.array(solver.loss_hist))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(np.array(t_test), np.array(u_exact), label="Exact",  lw=1.5)
        axes[0].plot(np.array(t_test), np.array(u_pred),  label="PINN", lw=1.5, ls="--")
        axes[0].set_xlabel("t"); axes[0].legend()
        axes[0].set_title(f"Harmonic  ω={omega}  (Rel-L2={rel_l2:.2e})")
        axes[1].semilogy(solver.loss_hist,  label="Total",  lw=1.2)
        axes[1].semilogy(solver.pde_hist,   label="PDE",    alpha=0.7)
        axes[1].semilogy(solver.ic_hist,    label="IC",     alpha=0.7)
        axes[1].set_xlabel("Epoch"); axes[1].legend()
        axes[1].set_title("Training loss")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "harmonic.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    print(f"\nOutputs saved to: {out_dir}/")
    return results


if __name__ == "__main__":
    import sys, pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_ode(load_config(cfg_path))
