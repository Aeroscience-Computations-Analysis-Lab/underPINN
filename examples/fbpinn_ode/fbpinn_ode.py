"""FBPINN on a multiscale 1-D ODE — domain decomposition over a long interval.

Reproduces the headline multiscale Finite Basis PINN benchmark from Moseley,
Markham & Nissen-Meyer (2021), "Finite Basis Physics-Informed Neural Networks":

    du/dx = Σ_i ω_i cos(ω_i x),   u(x0) = 0   →   exact  u(x) = Σ_i sin(ω_i x)

With well-separated frequencies (default ω = {1, 15}) over a *long* domain
(default [−2π, 2π], ~30 oscillations of the fast mode) a single global PINN
fails badly: spectral bias prevents it from fitting the high-frequency content,
and the long domain compounds the problem.  An FBPINN instead splits the line
into many small, *overlapping* subdomains, puts a tiny network on each, and
blends them with a smooth partition-of-unity window — so each subnet only has
to learn a locally-simple (low-frequency) slice of the solution.

Architecture (``underPINN.nn.fbpinn.FBPINN``)::

    u_hat(x) = u0 + g(x) · Σ_j  w_j(x) · NN_j(x − c_j)

* ``c_j``          subdomain centres (evenly spaced over [x_start, x_end])
* ``w_j(x)``       overlapping sigmoid windows (``window_1d``) — the partition
                   of unity that localises each subnet
* ``NN_j``         a small per-subdomain network
* ``g(x) = x − x0``  a hard *constraining operator* so the initial condition
                   u(x0) = u0 holds exactly (no IC loss term, no weight to tune).
                   ``x0`` is the domain start; for integer frequencies on a
                   ±2π-style domain the exact solution satisfies u(x0) = 0 too.

Only the PDE residual is minimised:  L = mean( (du_hat/dx − Σ ω_i cos ω_i x)² ).

Run directly or via the CLI::

    python examples/fbpinn_ode/fbpinn_ode.py
    python examples/fbpinn_ode/fbpinn_ode.py myconfig.yaml
    python -m underPINN run examples/fbpinn_ode/config.yaml
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.config.loader import cfg_get, save_config
from underPINN.nn.fbpinn import FBPINN, window_1d
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.io import save_predictions
from underPINN.utils.restart import RestartManager


# ---------------------------------------------------------------------------
# Subdomain layout — overlapping 1-D windows
# ---------------------------------------------------------------------------

def _build_subdomains(x_start, x_end, n_sub, overlap):
    """Evenly spaced, overlapping 1-D subdomains for the FBPINN window arrays.

    Returns ``(shifts, xs_min, xs_max, smins, smaxs)`` — each of shape
    ``(n_sub, 1)`` — ready to hand to :class:`FBPINN`.

    Each subdomain is centred at ``c_j`` and spans ``c_j ± half`` where
    ``half = 0.5 · W · (1 + overlap)`` and ``W = (x_end − x_start)/n_sub`` is the
    base spacing — so neighbouring windows overlap by a fraction ``overlap`` of
    a cell.  The window softness ``s`` is set to the overlap width so the
    sigmoid windows blend smoothly into a partition of unity.
    """
    W = (x_end - x_start) / n_sub
    centres = x_start + (np.arange(n_sub) + 0.5) * W
    half = 0.5 * W * (1.0 + overlap)
    s = max(overlap * W, 1e-3 * W)

    shifts = centres[:, None]
    xs_min = (centres - half)[:, None]
    xs_max = (centres + half)[:, None]
    smins  = np.full((n_sub, 1), s)
    smaxs  = np.full((n_sub, 1), s)
    return (jnp.asarray(shifts), jnp.asarray(xs_min), jnp.asarray(xs_max),
            jnp.asarray(smins), jnp.asarray(smaxs))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_fbpinn_ode(cfg) -> dict:
    """Train an FBPINN on  du/dx = cos(ω x),  u(0) = u0."""
    ph  = cfg.physics
    tr  = cfg.training
    out = cfg_get(cfg, "output", default=None)
    out_dir = (cfg_get(out, "dir", default="outputs/fbpinn_ode")
               if out else "outputs/fbpinn_ode")
    os.makedirs(out_dir, exist_ok=True)

    # Frequencies: prefer a list `omegas` (multiscale); fall back to scalar `omega`.
    omegas_cfg = cfg_get(ph, "omegas", default=None)
    if omegas_cfg is None:
        omegas_cfg = [float(cfg_get(ph, "omega", default=15.0))]
    omegas  = np.asarray([float(w) for w in omegas_cfg], dtype=np.float64)
    omega_j = jnp.asarray(omegas)
    x_start = float(cfg_get(ph, "x_start", default=-2.0 * np.pi))
    x_end   = float(cfg_get(ph, "x_end",   default= 2.0 * np.pi))
    u0      = float(cfg_get(ph, "u0",      default=0.0))

    sd       = cfg_get(cfg, "subdomains", default=None)
    n_sub    = int(cfg_get(sd, "n", default=15) if sd else 15)
    overlap  = float(cfg_get(sd, "overlap", default=0.3) if sd else 0.3)

    layers   = list(cfg.network.layers)
    n_col    = int(cfg_get(cfg.data, "n_collocation", default=2000))

    epochs    = int(tr.epochs)
    lr        = float(tr.lr)
    lr_alpha  = float(cfg_get(tr, "lr_alpha",  default=0.01))
    log_every = int(cfg_get(tr, "log_every",   default=500))
    seed      = int(cfg_get(tr, "seed",        default=0))

    # ── Subdomain decomposition + FBPINN model ────────────────────────────────
    shifts, xs_min, xs_max, smins, smaxs = _build_subdomains(
        x_start, x_end, n_sub, overlap)
    model = FBPINN(layers=layers, shifts=shifts, xs_min=xs_min, xs_max=xs_max,
                   smins=smins, smaxs=smaxs)

    omega_max = float(omegas.max())
    n_osc = omega_max * (x_end - x_start) / (2 * np.pi)
    print(f"FBPINN ODE:  du/dx = Σ ω·cos(ω·x),  ω={omegas.tolist()},  "
          f"u({x_start:.3f})={u0}")
    print(f"  Domain [{x_start:.3f}, {x_end:.3f}] (length {x_end-x_start:.2f}) "
          f"split into {n_sub} subdomains (overlap={overlap}),  subnet={layers}")
    print(f"  Exact solution: Σ sin(ω x)  →  ~{n_osc:.0f} oscillations of the "
          f"fast mode (ω={omega_max:g}) across the domain")

    key    = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, 1)))

    # ── Hard IC constraint + PDE residual ─────────────────────────────────────
    # u_hat(x) = u0 + (x − x_start) · FBPINN(x)  ⇒  u_hat(x_start) = u0 exactly,
    # so no initial-condition loss term is needed.
    def _u_vec(p, x):                       # x : (N, 1) → (N, 1)
        return u0 + (x - x_start) * model.apply(p, x)

    def _dudx_vec(p, x):
        # Each output u_i depends only on its own input x_i, so a single
        # forward-mode JVP with an all-ones tangent returns the whole vector
        # of per-point derivatives du_i/dx_i in one batched pass — far cheaper
        # than vmapping a scalar grad over every collocation point.
        u_val, du_val = jax.jvp(lambda xx: _u_vec(p, xx),
                                (x,), (jnp.ones_like(x),))
        return u_val, du_val

    def _forcing(x):                        # x : (N, 1) → (N, 1)
        # f(x) = Σ_i ω_i cos(ω_i x)
        return jnp.sum(omega_j[None, :] * jnp.cos(omega_j[None, :] * x),
                       axis=1, keepdims=True)

    x_col = jnp.linspace(x_start, x_end, n_col).reshape(-1, 1)

    lr_sched  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    optimizer = optax.adam(learning_rate=lr_sched)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, xc):
        def loss_fn(p):
            _, du = _dudx_vec(p, xc)
            res = du - _forcing(xc)
            return jnp.mean(res ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # ── Restart-aware training loop ───────────────────────────────────────────
    save_restart = int(cfg_get(tr, "save_restart_every", default=500))
    restart = RestartManager(out_dir, save_every=save_restart, cfg=cfg)
    start_ep, params, opt_state, hists = restart.maybe_restore(params, opt_state)
    loss_hist = hists.get("loss_hist", [])

    logger = ConsoleLogger(log_every=log_every)
    for ep in range(start_ep, epochs):
        params, opt_state, loss = step(params, opt_state, x_col)
        loss_hist.append(float(loss))
        logger.on_epoch_end(ep, {"loss": float(loss)})
        restart.maybe_save(ep, params, opt_state, {"loss_hist": loss_hist})
    restart.done()
    logger.on_train_end({"loss": loss_hist[-1] if loss_hist else float("nan")})

    # ── Evaluate against the exact solution ───────────────────────────────────
    x_test  = jnp.linspace(x_start, x_end, 4000)
    u_pred  = _u_vec(params, x_test.reshape(-1, 1))[:, 0]
    # exact  u(x) = u0 + Σ_i sin(ω_i x)
    u_exact = u0 + jnp.sum(jnp.sin(omega_j[None, :] * x_test[:, None]), axis=1)
    rel_l2  = float(jnp.linalg.norm(u_pred - u_exact)
                    / (jnp.linalg.norm(u_exact) + 1e-12))
    print(f"  Rel-L2 vs exact Σ sin(ω x) : {rel_l2:.4e}")

    # ── Save predictions, checkpoint, config ─────────────────────────────────
    save_predictions(
        out_dir,
        coords  = {"x": np.array(x_test)},
        outputs = {"u_pred": np.array(u_pred)},
        exact   = {"u_exact": np.array(u_exact)},
        filename="predictions.npz",
    )
    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(loss_hist))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    save_checkpoint(params, out_dir, metadata={
        "problem": "fbpinn_ode",
        "network": {"type": "fbpinn", "layers": layers, "n_subdomains": n_sub,
                    "overlap": overlap},
        "physics": {"omegas": omegas.tolist(), "x_start": x_start,
                    "x_end": x_end, "u0": u0},
        "results": {"n_epochs": len(loss_hist), "rel_l2": rel_l2},
    })

    # ── Figure: solution, partition-of-unity windows, training loss ──────────
    xg = np.array(x_test)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(xg, np.array(u_exact), label="Exact", lw=2.0)
    axes[0].plot(xg, np.array(u_pred), "--", label="FBPINN", lw=1.5)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x)")
    axes[0].set_title(f"Solution  ω={omegas.tolist()}  (Rel-L2={rel_l2:.2e})")
    axes[0].legend()

    for j in range(n_sub):
        wj = np.array(window_1d(x_test, xs_min[j, 0], xs_max[j, 0],
                                smins[j, 0], smaxs[j, 0]))
        axes[1].plot(xg, wj, lw=1.0)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("window  w_j(x)")
    axes[1].set_title(f"{n_sub} overlapping subdomains")

    axes[2].semilogy(loss_hist, lw=1.0)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("PDE residual loss")
    axes[2].set_title("Training loss")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fbpinn_ode.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nOutputs saved to: {out_dir}/")
    return {"params": params, "loss_hist": loss_hist,
            "rel_l2": rel_l2, "n_epochs": len(loss_hist)}


if __name__ == "__main__":
    import sys
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
                   else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_fbpinn_ode(load_config(cfg_path))
