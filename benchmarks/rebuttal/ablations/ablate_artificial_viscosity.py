"""Ablation of underPINN's artificial-viscosity options on the Toro-3 blast wave.

The paper advertises a *trainable* artificial-viscosity coefficient
(eps = softplus(log_av), learned jointly with the network weights) as a
contribution -- but every shock benchmark it reports actually runs with a
*fixed* coefficient (Toro3Evaluator and the ramp examples all pass
``art_visc=0.001`` with plain network params). The advertised feature is
therefore never exercised in the results, which is exactly the gap a reviewer
flagged. This script closes it by measuring three arms on the identical
problem:

  none        art_visc = 0        -- no dissipation at all
  fixed       art_visc = 0.001    -- what the paper's Toro3 result uses
  trainable   eps = softplus(log_av), learned  -- the advertised feature

Scored on relative L^2 against the exact Riemann solution. The learned eps is
reported so it can be compared against the hand-picked 0.001, which tells the
reader whether the tuning the feature automates was worth automating.

Run:
    python benchmarks/rebuttal/ablations/ablate_artificial_viscosity.py --epochs 20000
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

import jax                                                         # noqa: E402
import jax.numpy as jnp                                            # noqa: E402
import numpy as np                                                 # noqa: E402
import optax                                                       # noqa: E402

from common import (base_parser, jax_device_info, save_result,      # noqa: E402
                    timed, warn_if_cpu)

from underPINN.nn.mlp import MLP                                    # noqa: E402
from underPINN.pde.euler_1d_unsteady import Euler1DUnsteadyPDE      # noqa: E402
from underPINN.utils.metrics import relative_l2_error               # noqa: E402
from underPINN.utils.riemann import exact_riemann_1d                # noqa: E402

GAMMA = 1.4
X0, T_FINAL = 0.5, 0.012
LEFT, RIGHT = (1.0, 0.0, 1000.0), (1.0, 0.0, 0.01)
N_INT, N_IC, N_BC = 40000, 5000, 3000
BR, BI, BB = 2048, 400, 300
W_PDE, W_IC, W_BC = 1.0, 100.0, 10.0
LAYERS = [2, 128, 128, 128, 128, 128, 3]
LR = 1e-3
FIXED_AV = 0.001


def make_problem(seed: int):
    """Non-dimensionalised Toro-3 collocation sets (matches Toro3Evaluator)."""
    rho_ref, p_ref = max(LEFT[0], RIGHT[0]), max(LEFT[2], RIGHT[2])
    u_ref = float(np.sqrt(p_ref / rho_ref))
    t_ref = 1.0 / u_ref
    left_nd = (LEFT[0] / rho_ref, LEFT[1] / u_ref, LEFT[2] / p_ref)
    right_nd = (RIGHT[0] / rho_ref, RIGHT[1] / u_ref, RIGHT[2] / p_ref)
    tf_nd = T_FINAL / t_ref

    rng = np.random.default_rng(seed)
    xt_r = np.stack([rng.uniform(0.0, 1.0, N_INT),
                     rng.uniform(0.0, tf_nd, N_INT)], axis=1).astype("f4")
    x_ic = rng.uniform(0.0, 1.0, N_IC).astype("f4")
    le = x_ic < X0
    ic_tgt = np.stack([np.where(le, left_nd[0], right_nd[0]),
                       np.where(le, left_nd[1], right_nd[1]),
                       np.where(le, left_nd[2], right_nd[2])],
                      axis=1).astype("f4")
    xt_ic = np.stack([x_ic, np.zeros(N_IC, "f4")], axis=1)
    t_bc = rng.uniform(0.0, tf_nd, N_BC).astype("f4")
    xt_bcL = np.stack([np.zeros(N_BC, "f4"), t_bc], axis=1)
    xt_bcR = np.stack([np.ones(N_BC, "f4"), t_bc], axis=1)

    return dict(
        xt_r=jnp.array(xt_r), xt_ic=jnp.array(xt_ic),
        ic_tgt=jnp.array(ic_tgt),
        xt_bcL=jnp.array(xt_bcL), xt_bcR=jnp.array(xt_bcR),
        bcL_tgt=jnp.array(np.array(left_nd, "f4")),
        bcR_tgt=jnp.array(np.array(right_nd, "f4")),
        tf_nd=tf_nd, left_nd=left_nd, right_nd=right_nd,
    )


def run_arm(arm: str, epochs: int, seed: int, prob) -> dict:
    """*arm* is 'none' | 'fixed' | 'trainable'."""
    model = MLP(layers=LAYERS)
    art_visc = {"none": 0.0, "fixed": FIXED_AV, "trainable": FIXED_AV}[arm]
    pde = Euler1DUnsteadyPDE(model, gamma=GAMMA, art_visc=art_visc,
                             transform="exp")

    net_params = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 2)))
    if arm == "trainable":
        # Euler1DUnsteadyPDE switches to the learned-eps path when params is
        # the combined {"net": ..., "log_av": ...} pytree. Initialise log_av so
        # softplus(log_av) == FIXED_AV, i.e. training starts exactly where the
        # fixed arm sits and can move from there.
        log_av0 = float(np.log(np.expm1(FIXED_AV)))
        params = {"net": net_params, "log_av": jnp.array(log_av0, dtype="f4")}
    else:
        params = net_params

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=1e-2)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))
    state = opt.init(params)

    @jax.jit
    def step(params, state, r_b, ic_b, ic_t, bcL_b, bcR_b):
        def loss_fn(p):
            res = pde.residual(p, r_b)
            pde_l = jnp.mean(jnp.sum(res ** 2, axis=-1))
            ic_l = jnp.mean(jnp.sum((pde.apply(p, ic_b) - ic_t) ** 2, axis=-1))
            bc_l = (jnp.mean(jnp.sum(
                        (pde.apply(p, bcL_b) - prob["bcL_tgt"]) ** 2, axis=-1))
                    + jnp.mean(jnp.sum(
                        (pde.apply(p, bcR_b) - prob["bcR_tgt"]) ** 2, axis=-1)))
            return W_PDE * pde_l + W_IC * ic_l + W_BC * bc_l, pde_l
        (total, _pl), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = opt.update(grads, state)
        return optax.apply_updates(params, updates), state, total

    key = jax.random.PRNGKey(seed + 7)

    def train():
        p, s, key_ = params, state, key
        total = None
        for _ in range(epochs):
            key_, k1, k2, k3 = jax.random.split(key_, 4)
            ir = jax.random.randint(k1, (BR,), 0, N_INT)
            ii = jax.random.randint(k2, (BI,), 0, N_IC)
            ib = jax.random.randint(k3, (BB,), 0, N_BC)
            p, s, total = step(p, s, prob["xt_r"][ir], prob["xt_ic"][ii],
                               prob["ic_tgt"][ii], prob["xt_bcL"][ib],
                               prob["xt_bcR"][ib])
        total.block_until_ready()
        return p, float(total)

    (final_params, final_loss), wall = timed(train)

    # ── accuracy vs the exact Riemann solution ───────────────────────────────
    Nx = 400
    xg = np.linspace(0.0, 1.0, Nx, dtype="f4")
    pts = jnp.array(np.stack([xg, np.full(Nx, prob["tf_nd"], "f4")], axis=1))
    pred = np.array(pde.apply(final_params, pts))
    re, ue, pe = exact_riemann_1d(xg, prob["tf_nd"], X0, GAMMA,
                                  prob["left_nd"], prob["right_nd"])
    exact = np.stack([re, ue, pe], axis=1)
    rel_l2 = float(relative_l2_error(jnp.array(pred), jnp.array(exact)))

    eps_used = (float(jax.nn.softplus(final_params["log_av"]))
                if arm == "trainable" else art_visc)
    return {"arm": arm, "epochs": epochs, "wall_s": wall,
            "ms_per_epoch": 1e3 * wall / epochs, "final_loss": final_loss,
            "rel_l2": rel_l2, "eps_initial": art_visc, "eps_final": eps_used,
            "trainable": arm == "trainable"}


ARMS = {
    "none": "art_visc = 0 (no dissipation)",
    "fixed": f"art_visc = {FIXED_AV} fixed (what the paper's Toro3 uses)",
    "trainable": "eps = softplus(log_av), learned (the advertised feature)",
}


def main() -> int:
    ap = base_parser("Ablate artificial viscosity on the Toro-3 blast wave")
    ap.set_defaults(epochs=20000)
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    args = ap.parse_args()

    info = jax_device_info(require_gpu=not args.allow_cpu)
    warn_if_cpu(info)
    print(f"JAX backend: {info['platform']} ({info['device_name']})")
    print(f"Epochs per arm: {args.epochs}   seed: {args.seed}\n")

    prob = make_problem(args.seed)
    rows = {}
    for arm in args.arms:
        print(f"--- {arm}: {ARMS[arm]}")
        try:
            r = run_arm(arm, args.epochs, args.seed, prob)
            rows[arm] = r
            print(f"    {r['ms_per_epoch']:6.2f} ms/ep  loss={r['final_loss']:.4e}"
                  f"  rel_L2={r['rel_l2']:.4e}  eps={r['eps_final']:.6g}")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            rows[arm] = {"error": f"{type(e).__name__}: {e}"}

    ok = {k: v for k, v in rows.items() if "error" not in v}
    if ok:
        base = ok.get("fixed", {}).get("rel_l2")
        print("\n" + "=" * 80)
        print(f"{'arm':12s} {'rel L2':>11s} {'vs fixed':>10s} {'eps used':>12s}   "
              f"description")
        print("-" * 80)
        for arm in ARMS:
            if arm not in ok:
                continue
            r = ok[arm]
            rel = f"{base / r['rel_l2']:.2f}x" if base else "-"
            print(f"{arm:12s} {r['rel_l2']:11.4e} {rel:>10s} "
                  f"{r['eps_final']:12.6g}   {ARMS[arm]}")
        print("=" * 80)
        if "trainable" in ok and "fixed" in ok:
            lo = ok["trainable"]["eps_final"]
            print(f"\nLearned eps settled at {lo:.6g} vs the hand-picked "
                  f"{FIXED_AV:g}.")
            print("If these are close and the errors match, the trainable "
                  "coefficient reproduces\nhand tuning automatically; if they "
                  "diverge, report which one actually won.")

    save_result("ablation_artificial_viscosity_toro3", {
        "problem": "toro3_blast_wave", "epochs": args.epochs, "seed": args.seed,
        "device": info, "metric": "relative L2 vs exact Riemann solution",
        "arms": rows,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
