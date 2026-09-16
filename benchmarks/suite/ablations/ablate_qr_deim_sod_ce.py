"""A new benchmark case, not just a new ablation run: the classic Sod
shock tube as a **compression-expansion** test of the 1-D unsteady
compressible Euler equations.

Requested directly: build a case combining a compression region (shock)
and an expansion region (rarefaction fan) -- distinct from the existing
``ablate_qr_deim.py`` (Toro-3, a *severe* 5-decade-pressure-jump blast
wave, left=(1,0,1000)/right=(1,0,0.01)), which is harder-but-less-
canonical. Sod's problem -- left=(1,0,1), right=(0.125,0,0.1), a mild 10:1
pressure ratio -- is *the* textbook Euler Riemann test and gives exactly
the requested structure explicitly: left to right, a rarefaction fan
(EXPANSION -- density and pressure smoothly drop), a contact discontinuity,
and a shock (COMPRESSION -- density and pressure jump discontinuously).

Same methodology as ``ablate_qr_deim.py`` (reused directly, not
reinvented): ``underPINN.pde.euler_1d_unsteady.Euler1DUnsteadyPDE`` in
conservative form with a fixed artificial-viscosity coefficient, scored
against ``underPINN.utils.riemann.exact_riemann_1d``'s exact solution, five
collocation-strategy arms:

  none            static uniform collocation pool, no adaptive resampling
  rad             RAR-D/RAD magnitude-weighted resampling, full pool replaced
  qr_deim         QR-DEIM-R resampling, full pool replaced
  rad_hybrid      RAD, 80% pool fixed (uniform, for stability) / 20% adaptive
  qr_deim_hybrid  QR-DEIM-R, same 80/20 fixed/adaptive split

**Physics loss is tracked and reported separately from the total loss at
every epoch, not just at the end** -- ``run_arm`` returns both
``pde_loss_hist`` (the raw PDE-residual term alone, every epoch) and
``final_loss``/``final_pde_loss`` (total vs. physics-only, at the end),
and the solution plot below has a dedicated physics-loss-only panel next
to the usual total-loss curve, so the two are visually as well as
numerically distinguishable per arm.

Run:
    python benchmarks/suite/ablations/ablate_qr_deim_sod_ce.py --epochs 5000
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
import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")                                              # noqa: E402
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import optax                                                       # noqa: E402

from common import (base_parser, jax_device_info, load_raw_arrays,  # noqa: E402
                    load_results, RESULTS_DIR, save_raw_arrays,
                    save_result, timed, warn_if_cpu)

from underPINN.nn.mlp import MLP                                    # noqa: E402
from underPINN.pde.euler_1d_unsteady import Euler1DUnsteadyPDE      # noqa: E402
from underPINN.utils.metrics import relative_l2_error               # noqa: E402
from underPINN.utils.riemann import exact_riemann_1d                # noqa: E402
from underPINN.utils.sampling import qr_deim_resample, rad_resample # noqa: E402

GAMMA = 1.4
X0, T_FINAL = 0.5, 0.2          # classic Sod: waves stay well inside [0,1] at t=0.2
LEFT, RIGHT = (1.0, 0.0, 1.0), (0.125, 0.0, 0.1)   # 10:1 pressure ratio -- mild,
                                                    # vs. Toro-3's 100,000:1
N_INT, N_IC, N_BC = 20000, 3000, 2000
BR, BI, BB = 2048, 400, 300
W_PDE, W_IC, W_BC = 1.0, 100.0, 10.0
LAYERS = [2, 128, 128, 128, 128, 128, 3]
LR = 1e-3
FIXED_AV = 0.001   # same vetted value as Toro-3/examples/toro3 -- not retuned here


def make_problem(seed: int):
    """Non-dimensionalised Sod collocation sets -- identical scheme to
    ablate_qr_deim.py's make_problem, so the two cases are directly
    comparable in methodology (though not in physics: Sod's pressure ratio
    is 10:1 vs. Toro-3's 100,000:1, so the reference scales here are much
    closer to 1 and the non-dimensionalisation is closer to a no-op)."""
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

    def domain_sampler(n, s):
        r = np.random.default_rng(s)
        return np.stack([r.uniform(0.0, 1.0, n),
                         r.uniform(0.0, tf_nd, n)], axis=1).astype("f4")

    return dict(
        xt_r=jnp.array(xt_r), xt_ic=jnp.array(xt_ic),
        ic_tgt=jnp.array(ic_tgt),
        xt_bcL=jnp.array(xt_bcL), xt_bcR=jnp.array(xt_bcR),
        bcL_tgt=jnp.array(np.array(left_nd, "f4")),
        bcR_tgt=jnp.array(np.array(right_nd, "f4")),
        tf_nd=tf_nd, left_nd=left_nd, right_nd=right_nd,
        domain_sampler=domain_sampler,
    )


ARM_CONFIG = {
    # arm name -> (base resampling method or None, adaptive_frac). Same
    # majority-fixed/minority-adaptive hybrid design as ablate_qr_deim.py.
    "none":           (None,      1.0),
    "rad":            ("rad",     1.0),
    "qr_deim":        ("qr_deim", 1.0),
    "qr_deim_hybrid": ("qr_deim", 0.2),
    "rad_hybrid":     ("rad",     0.2),
}


def run_arm(arm: str, epochs: int, seed: int, prob) -> dict:
    """*arm* indexes :data:`ARM_CONFIG` for the resampling method (or
    ``None`` for no resampling -- i.e. plain uniform collocation) and the
    adaptive fraction of the pool. Every arm uses the same fixed
    artificial-viscosity coefficient (``FIXED_AV``); only the collocation
    strategy varies."""
    strategy, adaptive_frac = ARM_CONFIG[arm]
    model = MLP(layers=LAYERS)
    pde = Euler1DUnsteadyPDE(model, gamma=GAMMA, art_visc=FIXED_AV,
                             transform="exp")

    params = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 2)))
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
        (total, pl), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = opt.update(grads, state)
        return optax.apply_updates(params, updates), state, total, pl

    key = jax.random.PRNGKey(seed + 7)
    xt_r = prob["xt_r"]                    # (N_INT, 2), replaced by resampling
    xt_r_init = np.array(xt_r)             # kept for the migration plot
    resample_period = max(1, epochs // 5)  # ~5 resamplings over the run

    n_adapt = max(1, round(adaptive_frac * N_INT)) if strategy is not None else 0
    n_fixed = N_INT - n_adapt
    xt_r_fixed = xt_r[:n_fixed] if n_fixed > 0 else None

    # Physics loss tracked SEPARATELY from the total loss at every epoch --
    # not just a final-epoch number. Both are appended every step so the
    # two curves can be plotted and compared directly (see plot_solutions).
    loss_hist: list = []
    pde_hist: list = []

    def train():
        nonlocal xt_r
        p, s, key_ = params, state, key
        total, pl = None, None
        for ep in range(epochs):
            if strategy is not None and ep > 0 and ep % resample_period == 0:
                if strategy == "rad":
                    new_pts = rad_resample(
                        pde, p, prob["domain_sampler"],
                        n_keep=n_adapt, n_candidates=5 * n_adapt,
                        k=1.0, c=1.0, seed=seed + ep)
                else:  # "qr_deim"
                    new_pts = qr_deim_resample(
                        pde, p, prob["domain_sampler"],
                        n_keep=n_adapt, n_candidates=5 * n_adapt,
                        seed=seed + ep)
                adapt_pts = jnp.array(new_pts)
                xt_r = (jnp.concatenate([xt_r_fixed, adapt_pts], axis=0)
                       if xt_r_fixed is not None else adapt_pts)
            key_, k1, k2, k3 = jax.random.split(key_, 4)
            ir = jax.random.randint(k1, (BR,), 0, N_INT)
            ii = jax.random.randint(k2, (BI,), 0, N_IC)
            ib = jax.random.randint(k3, (BB,), 0, N_BC)
            p, s, total, pl = step(p, s, xt_r[ir], prob["xt_ic"][ii],
                                   prob["ic_tgt"][ii], prob["xt_bcL"][ib],
                                   prob["xt_bcR"][ib])
            loss_hist.append(float(total))
            pde_hist.append(float(pl))
        total.block_until_ready()
        return p, float(total), float(pl)

    (final_params, final_loss, final_pde_loss), wall = timed(train)

    Nx = 400
    xg = np.linspace(0.0, 1.0, Nx, dtype="f4")
    pts = jnp.array(np.stack([xg, np.full(Nx, prob["tf_nd"], "f4")], axis=1))
    pred = np.array(pde.apply(final_params, pts))
    re, ue, pe = exact_riemann_1d(xg, prob["tf_nd"], X0, GAMMA,
                                  prob["left_nd"], prob["right_nd"])
    exact = np.stack([re, ue, pe], axis=1)
    rel_l2 = float(relative_l2_error(jnp.array(pred), jnp.array(exact)))

    return {"strategy": arm, "base_method": strategy,
            "adaptive_frac": adaptive_frac, "n_fixed": n_fixed,
            "n_adapt": n_adapt, "epochs": epochs, "wall_s": wall,
            "ms_per_epoch": 1e3 * wall / epochs,
            "final_loss": final_loss, "final_pde_loss": final_pde_loss,
            "rel_l2": rel_l2,
            "_xy_init": xt_r_init, "_xy_final": np.array(xt_r),
            "_xg": xg, "_pred": pred, "_exact": exact,
            "_loss_hist": np.array(loss_hist, "f4"),
            "_pde_hist": np.array(pde_hist, "f4")}


ARMS = {
    "none": "static, uniform collocation pool, no adaptive resampling",
    "rad": "RAR-D/RAD magnitude-weighted resampling, full pool replaced",
    "qr_deim": "QR-DEIM-R resampling, full pool replaced",
    "qr_deim_hybrid": "QR-DEIM-R, 80% pool fixed (stability) / 20% adaptive (improvement)",
    "rad_hybrid": "RAD, 80% pool fixed (stability) / 20% adaptive (improvement)",
}


def plot_migration(rows: dict, prob: dict, out_path: str) -> None:
    """Initial vs. final (x, t) collocation scatter, one column per arm --
    same rendering convention as ablate_qr_deim.py (grey = fixed/uniform
    portion, red = adaptive portion; verified not to silently overplot a
    moving "uniform" cloud, see that script's docstring for the full
    rationale)."""
    ok = {k: v for k, v in rows.items() if "error" not in v and "_xy_init" in v}
    if not ok:
        return
    arms = [a for a in ARMS if a in ok]
    fig, axes = plt.subplots(2, len(arms), figsize=(4.5 * len(arms), 8),
                             sharex=True, sharey=True)
    if len(arms) == 1:
        axes = axes[:, None]
    for col, arm in enumerate(arms):
        xy_init, xy_final = ok[arm]["_xy_init"], ok[arm]["_xy_final"]
        n_fixed = ok[arm].get("n_fixed", len(xy_init))
        for row, (xy, label) in enumerate(
                [(xy_init, "initial"), (xy_final, "final")]):
            ax = axes[row, col]
            fixed, adapt = xy[:n_fixed], xy[n_fixed:]
            ax.scatter(fixed[:, 0], fixed[:, 1], s=1.5, alpha=0.15,
                      c="#a0aec0", label="fixed (uniform)")
            if len(adapt):
                ax.scatter(adapt[:, 0], adapt[:, 1], s=2.5, alpha=0.5,
                          c="#c53030", label="adaptive")
            ax.axvline(X0, color="k", ls="--", lw=1, alpha=0.6)
            ax.set_title(f"{arm} — {label}" if row == 0 else label,
                        fontsize=10)
            if row == 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("t (non-dim.)")
    axes[0, -1].legend(fontsize=7, loc="upper right", markerscale=4)
    fig.suptitle("Sod compression-expansion: interior collocation pool, "
                "initial vs. final (grey = fixed/uniform, red = adaptive; "
                "dashed line = initial discontinuity at x=X0)",
                fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCollocation migration plot -> {out_path}")


_ARM_STYLE = {"none": dict(c="#718096", ls=":"),
             "rad": dict(c="#dd6b20", ls="--"),
             "qr_deim": dict(c="#2b6cb0", ls="-."),
             "rad_hybrid": dict(c="#c05621", ls="-"),
             "qr_deim_hybrid": dict(c="#1a4971", ls="-")}
_FIELD_LABELS = [r"$\rho$", r"$u$", r"$p$"]


def plot_solutions(rows: dict, out_path: str) -> None:
    """Density/velocity/pressure profiles at t=t_final (exact vs. every
    arm's PINN prediction, all overlaid) plus a dedicated pair of loss
    panels -- total loss and, separately, the physics-residual-only loss
    -- so "does the physics equation itself get satisfied" is visible on
    its own curve, not blended into IC/BC terms."""
    ok = {k: v for k, v in rows.items() if "error" not in v and "_pred" in v}
    if not ok:
        return
    arms = [a for a in ARMS if a in ok]
    xg = ok[arms[0]]["_xg"]
    exact = ok[arms[0]]["_exact"]        # identical across arms (same reference)

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.2))
    for j, (ax, label) in enumerate(zip(axes[:3], _FIELD_LABELS)):
        ax.plot(xg, exact[:, j], "k-", lw=2, label="exact")
        for arm in arms:
            style = _ARM_STYLE.get(arm, {})
            ax.plot(xg, ok[arm]["_pred"][:, j], lw=1.6,
                   label=f"{arm} (L2={ok[arm]['rel_l2']:.3f})", **style)
        ax.axvline(X0, color="gray", lw=0.8, alpha=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[0].legend(fontsize=7, loc="best")

    # Physics loss, plotted separately from total loss -- two panels, not
    # one, so the physics-only convergence is directly legible per arm.
    ax_tot, ax_pde = axes[3], axes[4]
    for arm in arms:
        style = _ARM_STYLE.get(arm, {})
        lh, ph = ok[arm].get("_loss_hist"), ok[arm].get("_pde_hist")
        if lh is None or len(lh) == 0:
            continue
        ep = np.arange(1, len(lh) + 1)
        ax_tot.plot(ep, lh, lw=1.3, label=arm, **style)
        ax_pde.plot(ep, ph, lw=1.3, label=arm, **style)
    for ax, title in [(ax_tot, "total loss (PDE+IC+BC)"),
                      (ax_pde, "physics (PDE-residual) loss only")]:
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    ax_pde.legend(fontsize=7, loc="best")

    fig.suptitle("Sod compression-expansion (rarefaction + shock) solution "
                "profiles and loss curves, total vs. physics-only, "
                "across resampling strategies", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Solution + loss plot -> {out_path}")


def main() -> int:
    global FIXED_AV
    ap = base_parser("Sod compression-expansion Euler case: uniform vs. "
                     "QR-DEIM-R vs. RAD collocation, fixed artificial "
                     "viscosity, physics loss tracked separately")
    ap.set_defaults(epochs=5000)
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--art-visc", type=float, default=FIXED_AV,
                    help="fixed artificial-viscosity coefficient (default "
                         "matches the vetted Toro-3/examples value)")
    ap.add_argument("--replot", action="store_true",
                    help="skip training entirely and regenerate the plots "
                         "(and the summary table) from the previous run's "
                         "results/ablation_qr_deim_sod_ce_raw.npz + .json.")
    args = ap.parse_args()

    FIXED_AV = args.art_visc

    if args.replot:
        rows = load_raw_arrays("ablation_qr_deim_sod_ce")
        saved = load_results("ablation_qr_deim_sod_ce")["ablation_qr_deim_sod_ce"]
        for arm, scalars in saved["arms"].items():
            rows.setdefault(arm, {}).update(scalars)
        prob = make_problem(saved["seed"])
        print(f"Replotting from saved results (epochs={saved['epochs']}, "
             f"seed={saved['seed']}, {len(rows)} arms) -- no training run.\n")
        plot_migration(rows, prob, os.path.join(
            RESULTS_DIR, "qr_deim_sod_ce_collocation_migration.png"))
        plot_solutions(rows, os.path.join(
            RESULTS_DIR, "qr_deim_sod_ce_solutions.png"))
        return 0

    info = jax_device_info(require_gpu=not args.allow_cpu)
    warn_if_cpu(info)
    print(f"JAX backend: {info['platform']} ({info['device_name']})")
    print(f"Epochs per arm: {args.epochs}   seed: {args.seed}   "
         f"art_visc: {FIXED_AV}\n")
    print(f"Problem: Sod compression-expansion -- left=(rho,u,p)={LEFT}, "
         f"right={RIGHT}, gamma={GAMMA}, x0={X0}, t_final={T_FINAL}\n")

    prob = make_problem(args.seed)
    rows = {}
    for arm in args.arms:
        print(f"--- {arm}: {ARMS[arm]}")
        try:
            r = run_arm(arm, args.epochs, args.seed, prob)
            rows[arm] = r
            print(f"    {r['ms_per_epoch']:6.2f} ms/ep  "
                 f"total_loss={r['final_loss']:.4e}  "
                 f"physics_loss={r['final_pde_loss']:.4e}  "
                 f"rel_L2={r['rel_l2']:.4e}")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            rows[arm] = {"error": f"{type(e).__name__}: {e}"}

    ok = {k: v for k, v in rows.items() if "error" not in v}
    if ok:
        base = ok.get("none", {}).get("rel_l2") or ok.get("rad", {}).get("rel_l2")
        print("\n" + "=" * 100)
        print(f"{'strategy':16s} {'ms/ep':>8s} {'total loss':>12s} "
             f"{'physics loss':>13s} {'rel L2':>11s} {'vs base':>9s}")
        print("-" * 100)
        for arm in ARMS:
            if arm not in ok:
                continue
            r = ok[arm]
            rel = f"{base / r['rel_l2']:.2f}x" if base else "-"
            print(f"{arm:16s} {r['ms_per_epoch']:8.2f} {r['final_loss']:12.4e} "
                 f"{r['final_pde_loss']:13.4e} {r['rel_l2']:11.4e} {rel:>9s}")
        print("=" * 100)
        if "rad" in ok and "qr_deim" in ok:
            better = "qr_deim" if ok["qr_deim"]["rel_l2"] < ok["rad"]["rel_l2"] \
                else "rad"
            print(f"\n{better} reached lower relative L2 error on this problem.")

    plot_migration(rows, prob, os.path.join(
        RESULTS_DIR, "qr_deim_sod_ce_collocation_migration.png"))
    plot_solutions(rows, os.path.join(
        RESULTS_DIR, "qr_deim_sod_ce_solutions.png"))

    save_raw_arrays("ablation_qr_deim_sod_ce", rows)
    rows_for_json = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in rows.items()}
    save_result("ablation_qr_deim_sod_ce", {
        "problem": "sod_compression_expansion", "epochs": args.epochs,
        "seed": args.seed, "device": info,
        "metric": "relative L2 vs exact Riemann solution",
        "physics_loss_note": "final_pde_loss / _pde_hist track the PDE-"
                             "residual term alone, separate from final_loss "
                             "(total, incl. IC/BC terms)",
        "art_visc": FIXED_AV, "left": LEFT, "right": RIGHT, "x0": X0,
        "t_final": T_FINAL, "arms": rows_for_json,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
