"""Toro test 3 — 1-D unsteady compressible Euler PINN (severe blast-wave).

The left half of the Woodward–Colella blast-wave problem (Toro, *Riemann
Solvers and Numerical Methods*, Test 3):

    x ∈ [0, 1],  diaphragm at x0 = 0.5,  γ = 1.4,  evolve to t = 0.012
    Left  state (x < x0):  (ρ, u, p) = (1.0, 0,  1000.0)
    Right state (x > x0):  (ρ, u, p) = (1.0, 0,     0.01)

Pressure spans five decades and the post-shock density (~6) overshoots the
initial data, so the naive Sod recipe is badly conditioned.  Two ingredients
fix that:

* **exp / log positivity** —  ρ = exp(f_ρ),  p = exp(f_p)  (set in the PDE via
  ``transform="exp"``).  The relative gradient d log p / d f_p = 1 is constant
  across decades, unlike softplus whose gradient ∝ p collapses at low pressure.
* **reference-state non-dimensionalisation** —  solve for  ρ̃=ρ/ρ_ref,
  ũ=u/u_ref,  p̃=p/p_ref  with  ρ_ref=max ρ,  p_ref=max p,  u_ref=√(p_ref/ρ_ref),
  x̃=(x−x_min)/L,  t̃=t·u_ref/L.  The Euler equations are invariant under this
  scaling, so the same conservative residual is used — but every primitive and
  every flux is now O(1), which balances the loss across the four equations.

A learnable artificial viscosity (ε = softplus(log_av)) and residual-adaptive
resampling (RAR) capture the strong shock.

Run directly or via the CLI:

    python examples/toro3/toro3.py
    python examples/toro3/toro3.py myconfig.yaml
    python -m underPINN run examples/toro3/config.yaml

Network: (x̃, t̃) → (ρ̃, ũ, p̃)
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.config.loader import cfg_get, save_config
from underPINN.nn.factory import build_model, network_config
from underPINN.pde.euler_1d_unsteady import Euler1DUnsteadyPDE
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.restart import RestartManager
from underPINN.utils.sampling import rad_resample, safe_choice
from underPINN.utils.riemann import exact_riemann_1d


def _ic_state(x, x0, left, right):
    """Riemann initial data (ρ, u, p) at t = 0 for positions x."""
    x = np.asarray(x)
    le = x < x0
    rho = np.where(le, left[0], right[0]).astype(np.float32)
    u   = np.where(le, left[1], right[1]).astype(np.float32)
    p   = np.where(le, left[2], right[2]).astype(np.float32)
    return rho, u, p


def run_toro3(cfg) -> dict:
    """Train a 1-D Euler PINN on Toro test 3 (non-dimensional, exp positivity)."""
    ph  = cfg.physics
    tr  = cfg.training
    lw  = cfg.loss
    out = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/toro3") if out else "outputs/toro3"
    os.makedirs(out_dir, exist_ok=True)

    gamma    = float(cfg_get(ph, "gamma",   default=1.4))
    x_min    = float(cfg_get(ph, "x_min",   default=0.0))
    x_max    = float(cfg_get(ph, "x_max",   default=1.0))
    x0       = float(cfg_get(ph, "x0",      default=0.5))
    t_final  = float(cfg_get(ph, "t_final", default=0.012))
    left     = tuple(cfg_get(ph, "left",  default=[1.0, 0.0, 1000.0]))
    right    = tuple(cfg_get(ph, "right", default=[1.0, 0.0, 0.01]))
    art_visc = float(cfg_get(ph, "art_visc", default=0.02))
    train_av = bool(cfg_get(ph, "trainable_visc", default=True))
    transform = str(cfg_get(ph, "transform", default="exp"))

    # ── Reference-state non-dimensionalisation ────────────────────────────────
    L      = x_max - x_min
    rho_ref = float(cfg_get(ph, "rho_ref", default=max(left[0], right[0])))
    p_ref   = float(cfg_get(ph, "p_ref",   default=max(left[2], right[2])))
    u_ref   = float(np.sqrt(p_ref / rho_ref))
    t_ref   = L / u_ref

    def _nd(state):                       # (ρ, u, p) → non-dimensional
        return (state[0] / rho_ref, state[1] / u_ref, state[2] / p_ref)

    left_nd  = _nd(left)
    right_nd = _nd(right)
    x0_nd    = (x0 - x_min) / L
    tf_nd    = t_final / t_ref            # non-dimensional final time

    epochs    = int(tr.epochs)
    lr        = float(tr.lr)
    lr_alpha  = float(cfg_get(tr, "lr_alpha",  default=0.01))
    log_every = int(cfg_get(tr, "log_every",   default=500))
    seed      = int(cfg_get(tr, "seed",        default=0))
    batch_r   = int(cfg_get(tr, "batch_r",     default=4096))
    batch_bc  = int(cfg_get(tr, "batch_bc",    default=512))

    # RAR (residual-based adaptive resampling) of the interior pool
    rar_period = int(cfg_get(tr, "rar_period",     default=0))
    rar_cand   = int(cfg_get(tr, "rar_candidates", default=5))
    rar_k      = float(cfg_get(tr, "rar_k",        default=1.0))
    rar_c      = float(cfg_get(tr, "rar_c",        default=1.0))

    d     = cfg.data
    n_int = int(cfg_get(d, "n_interior", default=30000))
    n_ic  = int(cfg_get(d, "n_ic",       default=3000))
    n_bc  = int(cfg_get(d, "n_bc",       default=1500))

    W_PDE = float(cfg_get(lw, "w_pde", default=1.0))
    W_IC  = float(cfg_get(lw, "w_ic",  default=20.0))
    W_BC  = float(cfg_get(lw, "w_bc",  default=10.0))

    print(f"Toro test 3:  x∈[{x_min},{x_max}],  x0={x0},  t_final={t_final},  γ={gamma}")
    print(f"  Left  (ρ,u,p) = {left}")
    print(f"  Right (ρ,u,p) = {right}")
    print(f"  Non-dim refs:  ρ_ref={rho_ref:g}  p_ref={p_ref:g}  "
          f"u_ref={u_ref:.4g}  t_ref={t_ref:.4g}  →  t̃_final={tf_nd:.4g}")
    print(f"  Positivity transform: {transform}")
    print(f"  Artificial viscosity: {'TRAINABLE init ' if train_av else 'fixed '}ε≈{art_visc}")

    # ── Collocation data (all in non-dimensional x̃∈[0,1], t̃∈[0,t̃_final]) ──────
    def _interior_sampler(n, s):
        r = np.random.default_rng(s)
        return np.stack([r.uniform(0.0, 1.0,   n),
                         r.uniform(0.0, tf_nd, n)], axis=1).astype(np.float32)

    rng   = np.random.default_rng(seed)
    xt_r  = _interior_sampler(n_int, seed)
    x_ic  = rng.uniform(0.0, 1.0, n_ic).astype(np.float32)
    xt_ic = np.stack([x_ic, np.zeros(n_ic, np.float32)], axis=1)
    rho_ic, u_ic, p_ic = _ic_state(x_ic, x0_nd, left_nd, right_nd)

    t_bc   = rng.uniform(0.0, tf_nd, n_bc).astype(np.float32)
    xt_bcL = np.stack([np.zeros(n_bc, np.float32),         t_bc], axis=1)
    xt_bcR = np.stack([np.ones(n_bc, np.float32),          t_bc], axis=1)

    xt_r_j   = jnp.array(xt_r)
    xt_ic_j  = jnp.array(xt_ic)
    ic_tgt   = jnp.array(np.stack([rho_ic, u_ic, p_ic], axis=1))
    xt_bcL_j = jnp.array(xt_bcL)
    xt_bcR_j = jnp.array(xt_bcR)
    bcL_tgt  = jnp.array(np.array(left_nd,  np.float32))
    bcR_tgt  = jnp.array(np.array(right_nd, np.float32))

    # ── Model + PDE (exp positivity, non-dimensional) ─────────────────────────
    net_cfg  = network_config(cfg)
    net_type = net_cfg["type"]
    layers   = net_cfg["layers"]
    model    = build_model(net_cfg)
    pde      = Euler1DUnsteadyPDE(model, gamma=gamma, art_visc=art_visc,
                                  transform=transform)
    print(f"  Network: {type(model).__name__} ({net_type})  layers={layers}")

    key    = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, 2)))
    if train_av:
        av_init = art_visc if art_visc > 0.0 else 1.0e-2
        raw0    = Euler1DUnsteadyPDE.inverse_softplus(av_init)
        params  = {"net": params, "log_av": jnp.asarray(raw0, jnp.float32)}
        print(f"  log_av init = {raw0:.3f}  →  ε ≈ {av_init}")

    lr_sched  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    optimizer = optax.chain(optax.scale_by_adam(),
                            optax.scale_by_schedule(lr_sched),
                            optax.scale(-1.0))
    opt_state = optimizer.init(params)

    # ── JIT step ──────────────────────────────────────────────────────────────
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

            total = W_PDE * pde_l + W_IC * ic_l + W_BC * bc_l
            return total, (pde_l, ic_l, bc_l)

        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return params, state, total, aux

    # ── Restart ───────────────────────────────────────────────────────────────
    save_restart = int(cfg_get(tr, "save_restart_every", default=500))
    restart = RestartManager(out_dir, save_every=save_restart, cfg=cfg)
    start_ep, params, opt_state, hists = restart.maybe_restore(params, opt_state)
    loss_hist = hists.get("loss_hist", [])
    eps_hist  = hists.get("eps_hist",  [])

    logger = ConsoleLogger(log_every=log_every)
    N_r, N_ic, N_bc = xt_r_j.shape[0], xt_ic_j.shape[0], xt_bcL_j.shape[0]
    key = jax.random.PRNGKey(seed + 7)

    # Restore the RNG key and the (possibly RAR-resampled) interior pool so a
    # resumed run continues bit-exactly instead of resetting them.
    saved_state = restart.restore_arrays()
    if "key" in saved_state:
        key = jnp.asarray(saved_state["key"], dtype=jnp.uint32)
    if "xt_r" in saved_state:
        xt_r_j = jnp.asarray(saved_state["xt_r"])
        N_r = xt_r_j.shape[0]

    try:
        for ep in range(start_ep, epochs):
            # RAR: refresh the interior pool toward the strong shock
            if rar_period > 0 and ep > 0 and ep % rar_period == 0:
                xt_r_j = jnp.array(rad_resample(
                    pde, params, _interior_sampler,
                    n_keep=n_int, n_candidates=rar_cand * n_int,
                    k=rar_k, c=rar_c, seed=seed + ep))
                print(f"  [ep {ep:5d}] RAR resampled interior pool ({n_int} pts)")

            key, k1, k2, k3 = jax.random.split(key, 4)
            ir  = safe_choice(k1, N_r,  batch_r)
            ii  = safe_choice(k2, N_ic, min(batch_bc, N_ic))
            ib  = safe_choice(k3, N_bc, min(batch_bc, N_bc))

            params, opt_state, total, (pl, il, bl) = step(
                params, opt_state,
                xt_r_j[ir], xt_ic_j[ii], ic_tgt[ii], xt_bcL_j[ib], xt_bcR_j[ib])

            loss_hist.append(float(total))
            eps_hist.append(pde.viscosity(params))
            logs = {"loss": float(total), "pde": float(pl),
                    "ic": float(il), "bc": float(bl)}
            if train_av:
                logs["eps"] = eps_hist[-1]
            logger.on_epoch_end(ep, logs)
            restart.maybe_save(ep, params, opt_state,
                               {"loss_hist": loss_hist, "eps_hist": eps_hist},
                               arrays={"xt_r": np.asarray(xt_r_j),
                                       "key": np.asarray(key)})
    except StopIteration:
        pass

    restart.done()
    logger.on_train_end({"loss": loss_hist[-1] if loss_hist else float("nan")})
    eps_final = pde.viscosity(params)
    if train_av:
        print(f"  Learned artificial viscosity:  ε = {eps_final:.6e}")

    # ── Evaluate at t_final vs exact (compute non-dim, then re-dimensionalise) ──
    Nx    = 400
    xt_nd = np.linspace(0.0, 1.0, Nx, dtype=np.float32)            # x̃ grid
    xt_e  = jnp.array(np.stack([xt_nd, np.full(Nx, tf_nd, np.float32)], axis=1))
    pv_nd = np.array(pde.apply(params, xt_e))                      # (Nx, 3) non-dim
    re_nd, ue_nd, pe_nd = exact_riemann_1d(xt_nd, tf_nd, x0_nd, gamma,
                                           left_nd, right_nd)

    # Re-dimensionalise for physically-meaningful plots
    xg    = x_min + xt_nd * L
    rho_p, u_p, p_p = pv_nd[:, 0] * rho_ref, pv_nd[:, 1] * u_ref, pv_nd[:, 2] * p_ref
    rho_e, u_e, p_e = re_nd * rho_ref, ue_nd * u_ref, pe_nd * p_ref

    def _rel_l2(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    errs = {"rho": _rel_l2(rho_p, rho_e),
            "u":   _rel_l2(u_p,  u_e),
            "p":   _rel_l2(p_p,  p_e)}
    print(f"\nRel-L² vs exact at t={t_final}:  "
          f"ρ={errs['rho']:.3e}  u={errs['u']:.3e}  p={errs['p']:.3e}")

    # ── Plots: ρ, u, p profiles at t_final (physical units) ───────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, yp, ye, name in zip(
            axes, [rho_p, u_p, p_p], [rho_e, u_e, p_e], ["ρ", "u", "p"]):
        ax.plot(xg, ye, "k-",  lw=2.0, label="Exact")
        ax.plot(xg, yp, "b--", lw=1.8, label="PINN")
        ax.set_xlabel("x")
        ax.set_ylabel(name)
        ax.set_title(f"{name}  at t={t_final}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(f"Toro test 3 (blast wave) — PINN vs exact   "
                 f"(learned ε={eps_final:.4f})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "toro3_profiles.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Loss + ε trajectory ───────────────────────────────────────────────────
    fig2, (axl, axe) = plt.subplots(1, 2, figsize=(13, 4))
    axl.semilogy(loss_hist, lw=1.0)
    axl.set_xlabel("Epoch")
    axl.set_ylabel("Total loss")
    axl.set_title("Training loss")
    axl.grid(alpha=0.3, which="both")
    axe.plot(eps_hist, lw=1.2, color="#D55E00")
    axe.set_xlabel("Epoch")
    axe.set_ylabel("ε = softplus(log_av)")
    axe.set_title("Learned artificial viscosity")
    axe.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "toro3_loss_eps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(loss_hist))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    save_checkpoint(params, out_dir, metadata={
        "problem": "toro3",
        "network": net_cfg,
        "physics": {"gamma": gamma, "x0": x0, "t_final": t_final,
                    "left": list(left), "right": list(right),
                    "transform": transform, "rho_ref": rho_ref, "p_ref": p_ref,
                    "u_ref": u_ref, "art_visc": eps_final,
                    "trainable_visc": train_av},
        "results": {"rel_l2": errs, "n_epochs": len(loss_hist)},
    })
    print(f"\nOutputs saved to: {out_dir}/")

    return {"params": params, "loss_hist": loss_hist,
            "rel_l2": errs, "eps_final": eps_final, "n_epochs": len(loss_hist)}


if __name__ == "__main__":
    import sys
    import pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_toro3(load_config(cfg_path))
