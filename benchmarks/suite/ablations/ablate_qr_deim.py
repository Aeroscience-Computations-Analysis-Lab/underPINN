"""Ablation: QR-DEIM-R adaptive collocation vs. RAD vs. no resampling, on
the Toro-3 blast wave.

A reviewer asked for discussion of adaptive point-selection schemes
(QR-DEIM / QR-DEIM-R) alongside underPINN's existing RAR-D/RAD magnitude-
weighted resampling. ``underPINN/utils/sampling.py::qr_deim_resample`` adds
a QR-pivoted, DEIM-inspired resampler (see its docstring for exactly which
published ideas it follows and where it is our own construction rather than
a literal reproduction) as a genuine alternative to
``rad_resample``. This script measures whether it actually helps, on the
same 1-D Toro-3 problem (and the same fixed ``art_visc=0.001``) used by
``ablate_artificial_viscosity.py`` -- only the *resampling strategy* varies:

  none       static collocation pool, no adaptive resampling at all
  rad        underPINN's existing RAR-D/RAD magnitude-weighted resampling
  qr_deim    the new QR-DEIM-R resampler

Scored on relative L^2 against the exact Riemann solution, same as the
artificial-viscosity ablation, so the two ablations' numbers are directly
comparable.

Run:
    python benchmarks/suite/ablations/ablate_qr_deim.py --epochs 5000
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
X0, T_FINAL = 0.5, 0.012
LEFT, RIGHT = (1.0, 0.0, 1000.0), (1.0, 0.0, 0.01)
N_INT, N_IC, N_BC = 40000, 5000, 3000
BR, BI, BB = 2048, 400, 300
W_PDE, W_IC, W_BC = 1.0, 100.0, 10.0
LAYERS = [2, 128, 128, 128, 128, 128, 3]
LR = 1e-3
FIXED_AV = 0.001


def make_problem(seed: int):
    """Non-dimensionalised Toro-3 collocation sets -- identical to
    ablate_artificial_viscosity.py's make_problem, so the two ablations are
    directly comparable."""
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
    # arm name -> (base resampling method or None, adaptive_frac).
    # adaptive_frac=1.0 replaces the *entire* interior pool every resample
    # cycle; adaptive_frac<1.0 freezes the first N_INT*(1-frac) points for
    # the whole run (uniform, for stability) and only ever resamples the
    # remaining N_INT*frac (adaptive, for improvement) -- the same
    # majority-fixed/minority-adaptive split ablate_qr_deim_ramp_ns.py uses
    # and benchmarks/suite/physicsnemo/compare_underpinn_multi.py's
    # ``--adaptive-frac`` reproduces there.
    "none":           (None,      1.0),
    "rad":            ("rad",     1.0),
    "qr_deim":        ("qr_deim", 1.0),
    "qr_deim_hybrid": ("qr_deim", 0.2),
    "rad_hybrid":     ("rad",     0.2),
    # Isolates the artificial-viscosity axis from the collocation-strategy
    # axis: static (never-resampled) pool, same as "none", but the
    # dissipation coefficient is now a trained scalar (epsilon =
    # softplus(log_av), see Euler1DUnsteadyPDE) instead of the fixed
    # FIXED_AV every other arm uses. Answers item 9 of the revision plan --
    # "does trainable AV alone close any of the gap Section sec:qrdeim
    # attributes to the fixed-visc cap?" -- without also changing where the
    # collocation points sit, so the two effects (point placement vs.
    # dissipation) are not conflated.
    "trainable_av":   (None,      1.0),
}
TRAINABLE_AV_ARMS = {"trainable_av"}


def _shock_sharpness(xg: np.ndarray, pred: np.ndarray,
                     exact: np.ndarray) -> dict:
    """Localized shock-capture metrics for Toro-3, complementing the
    domain-averaged relative L2.

    Relative L2 over the whole domain is dominated by the smooth
    rarefaction fan and the large near-constant regions; it barely moves
    when the near-discontinuous contact + shock go from smeared to sharp.
    For a blast wave the question that matters is how faithfully those
    jumps are resolved, so we also report, on a common fine grid at
    ``t_final``:

      rho_peak       max density (the shocked-region spike; Toro-3's
                     exact peak is ~6.0 and PINNs characteristically
                     undershoot it)
      grad_rho_peak  peak |d rho / dx| for x > X0 -- a schlieren-like
                     measure: a sharper, taller spike means the jump is
                     resolved over fewer cells
      grad_p_peak    peak |d p / dx| for x > X0 -- pressure jumps at the
                     shock (not at the contact), so this isolates the
                     shock front specifically

    Unlike the Ramp NS ablation's |grad(rho)| peak, which has no
    closed-form reference and is only comparable between arms, Toro-3 has
    an exact Riemann solution: each quantity is sampled on the *same*
    grid for prediction and exact, and reported both as the raw peak and
    as the ratio pred/exact (1.0 = as sharp as the exact solution at this
    resolution; < 1.0 = smeared)."""
    rho_p, p_p = pred[:, 0], pred[:, 2]
    rho_e, p_e = exact[:, 0], exact[:, 2]
    dx = float(xg[1] - xg[0])
    g_rho_p = np.abs(np.gradient(rho_p, dx))
    g_rho_e = np.abs(np.gradient(rho_e, dx))
    g_p_p = np.abs(np.gradient(p_p, dx))
    g_p_e = np.abs(np.gradient(p_e, dx))
    right = xg > X0                      # right-moving contact + shock
    grp_p, grp_e = float(g_rho_p[right].max()), float(g_rho_e[right].max())
    gpp_p, gpp_e = float(g_p_p[right].max()), float(g_p_e[right].max())
    rpk_p, rpk_e = float(rho_p.max()), float(rho_e.max())
    return {
        "rho_peak_pred": rpk_p, "rho_peak_exact": rpk_e,
        "rho_peak_ratio": rpk_p / rpk_e,
        "grad_rho_peak_pred": grp_p, "grad_rho_peak_exact": grp_e,
        "grad_rho_peak_ratio": grp_p / grp_e,
        "grad_p_peak_pred": gpp_p, "grad_p_peak_exact": gpp_e,
        "grad_p_peak_ratio": gpp_p / gpp_e,
        "shock_pos_pred": float(xg[right][np.argmax(g_p_p[right])]),
        "shock_pos_exact": float(xg[right][np.argmax(g_p_e[right])]),
    }


def run_arm(arm: str, epochs: int, seed: int, prob) -> dict:
    """*arm* indexes :data:`ARM_CONFIG` for the resampling method (or
    ``None`` for no resampling) and the adaptive fraction of the pool."""
    strategy, adaptive_frac = ARM_CONFIG[arm]
    model = MLP(layers=LAYERS)
    pde = Euler1DUnsteadyPDE(model, gamma=GAMMA, art_visc=FIXED_AV,
                             transform="exp")

    params = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 2)))
    if arm in TRAINABLE_AV_ARMS:
        # Same combined-params trick examples/toro3/toro3.py uses: once
        # params carries "log_av", Euler1DUnsteadyPDE.apply/residual pick up
        # the trained epsilon = softplus(log_av) automatically (see
        # Euler1DUnsteadyPDE._is_combined) instead of the fixed FIXED_AV --
        # no other code in run_arm needs to change, since everything below
        # already goes through pde.apply/pde.residual rather than
        # model.apply directly.
        raw0 = Euler1DUnsteadyPDE.inverse_softplus(FIXED_AV)
        params = {"net": params, "log_av": jnp.asarray(raw0, jnp.float32)}
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
    xt_r = prob["xt_r"]                    # (N_INT, 2), replaced by resampling
    xt_r_init = np.array(xt_r)             # kept for the migration plot
    resample_period = max(1, epochs // 5)  # ~5 resamplings over the run

    n_adapt = max(1, round(adaptive_frac * N_INT)) if strategy is not None else 0
    n_fixed = N_INT - n_adapt
    xt_r_fixed = xt_r[:n_fixed] if n_fixed > 0 else None

    def train():
        nonlocal xt_r
        p, s, key_ = params, state, key
        total = None
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
            p, s, total = step(p, s, xt_r[ir], prob["xt_ic"][ii],
                               prob["ic_tgt"][ii], prob["xt_bcL"][ib],
                               prob["xt_bcR"][ib])
        total.block_until_ready()
        return p, float(total)

    (final_params, final_loss), wall = timed(train)

    # 800 points: fine enough to expose how many cells each arm smears the
    # contact/shock over, coarse enough that the exact solution's own
    # (grid-limited) peak |d rho / dx| stays within ~1 order of magnitude of
    # what a well-resolved PINN can reach, so the pred/exact sharpness ratio
    # keeps useful dynamic range instead of collapsing toward zero.
    Nx = 800
    xg = np.linspace(0.0, 1.0, Nx, dtype="f4")
    pts = jnp.array(np.stack([xg, np.full(Nx, prob["tf_nd"], "f4")], axis=1))
    pred = np.array(pde.apply(final_params, pts))
    re, ue, pe = exact_riemann_1d(xg, prob["tf_nd"], X0, GAMMA,
                                  prob["left_nd"], prob["right_nd"])
    exact = np.stack([re, ue, pe], axis=1)
    rel_l2 = float(relative_l2_error(jnp.array(pred), jnp.array(exact)))
    sharp = _shock_sharpness(xg, pred, exact)
    final_av = (pde.viscosity(final_params) if arm in TRAINABLE_AV_ARMS
                else FIXED_AV)

    return {"strategy": arm, "base_method": strategy,
            "adaptive_frac": adaptive_frac, "n_fixed": n_fixed,
            "n_adapt": n_adapt, "epochs": epochs, "wall_s": wall,
            "ms_per_epoch": 1e3 * wall / epochs, "final_loss": final_loss,
            "rel_l2": rel_l2, "final_art_visc": final_av, **sharp,
            "_xy_init": xt_r_init, "_xy_final": np.array(xt_r),
            "_xg": xg, "_pred": pred, "_exact": exact}


ARMS = {
    "none": "static collocation pool, no adaptive resampling",
    "rad": "RAR-D/RAD magnitude-weighted resampling, full pool replaced",
    "qr_deim": "QR-DEIM-R resampling, full pool replaced",
    "qr_deim_hybrid": "QR-DEIM-R, 80% pool fixed (stability) / 20% adaptive (improvement)",
    "rad_hybrid": "RAD, 80% pool fixed (stability) / 20% adaptive (improvement)",
    "trainable_av": "static pool (as 'none'), but epsilon = softplus(log_av) is trained jointly with the network instead of fixed",
}


def plot_migration(rows: dict, prob: dict, out_path: str) -> None:
    """Initial vs. final (x, t) collocation scatter, one column per arm.

    The fixed ("uniform, for stability") and adaptive ("resampled, for
    improvement") portions of the pool are colour-split (grey vs. blue) in
    *both* rows -- for a hybrid arm the grey points are, by construction,
    bit-identical between the "initial" and "final" panels (verified with
    ``np.array_equal`` in ``tests/``, not just asserted); only the blue
    points ever move. This is a rendering fix, not a data fix -- a reviewer
    asked why the "uniform" points looked like they moved in an earlier,
    single-colour version of this plot, and they don't; overplotting 40,000
    same-colour dots at low alpha just made a visible new structure (the
    adaptive points tracing the shock) read as if the *whole* cloud had
    changed, when 80% of it (the grey points here) never does.

    A dashed line marks x=X0, the initial discontinuity location -- not the
    shock's actual (moving) trajectory, just a fixed visual reference for
    where the interesting physics starts.
    """
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
    fig.suptitle("Toro-3: interior collocation pool, initial vs. final "
                "(grey = fixed/uniform, red = adaptive; dashed line = "
                "initial discontinuity at x=X0)",
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
    """Density/velocity/pressure profiles at t=t_final: exact (black) vs.
    every arm's PINN prediction overlaid on the same axes, directly
    comparable -- answers "which method gives the best solution" visually,
    not just via the scalar rel-L2 table above."""
    ok = {k: v for k, v in rows.items() if "error" not in v and "_pred" in v}
    if not ok:
        return
    arms = [a for a in ARMS if a in ok]
    xg = ok[arms[0]]["_xg"]
    exact = ok[arms[0]]["_exact"]        # identical across arms (same reference)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for j, (ax, label) in enumerate(zip(axes, _FIELD_LABELS)):
        ax.plot(xg, exact[:, j], "k-", lw=2, label="exact")
        for arm in arms:
            style = _ARM_STYLE.get(arm, {})
            ax.plot(xg, ok[arm]["_pred"][:, j], lw=1.6,
                   label=f"{arm} (L2={ok[arm]['rel_l2']:.3f})", **style)
        ax.axvline(X0, color="gray", lw=0.8, alpha=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Toro-3 solution profiles at t=t_final: exact vs. every "
                "resampling strategy", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Solution profile plot -> {out_path}")


def main() -> int:
    ap = base_parser("Ablate QR-DEIM-R vs RAD vs no resampling on Toro-3")
    ap.set_defaults(epochs=5000)
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--replot", action="store_true",
                    help="skip training entirely and regenerate the plots "
                         "(and the summary table) from the previous run's "
                         "results/ablation_qr_deim_toro3_raw.npz + "
                         ".json -- run this after --arms once, not the "
                         "full training loop again, whenever only the "
                         "plotting code changes.")
    args = ap.parse_args()

    if args.replot:
        rows = load_raw_arrays("ablation_qr_deim_toro3")
        saved = load_results("ablation_qr_deim_toro3")["ablation_qr_deim_toro3"]
        for arm, scalars in saved["arms"].items():
            rows.setdefault(arm, {}).update(scalars)
        prob = make_problem(saved["seed"])
        print(f"Replotting from saved results (epochs={saved['epochs']}, "
             f"seed={saved['seed']}, {len(rows)} arms) -- no training run.\n")
        plot_migration(rows, prob, os.path.join(
            RESULTS_DIR, "qr_deim_toro3_collocation_migration.png"))
        plot_solutions(rows, os.path.join(
            RESULTS_DIR, "qr_deim_toro3_solutions.png"))
        return 0

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
                  f"  rel_L2={r['rel_l2']:.4e}")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            rows[arm] = {"error": f"{type(e).__name__}: {e}"}

    ok = {k: v for k, v in rows.items() if "error" not in v}
    if ok:
        base = ok.get("none", {}).get("rel_l2") or ok.get("rad", {}).get("rel_l2")
        print("\n" + "=" * 80)
        print(f"{'strategy':10s} {'ms/ep':>8s} {'rel L2':>11s} {'vs base':>9s}   "
              f"description")
        print("-" * 80)
        for arm in ARMS:
            if arm not in ok:
                continue
            r = ok[arm]
            rel = f"{base / r['rel_l2']:.2f}x" if base else "-"
            print(f"{arm:10s} {r['ms_per_epoch']:8.2f} {r['rel_l2']:11.4e} "
                  f"{rel:>9s}   {ARMS[arm]}")
        print("=" * 80)
        if "rad" in ok and "qr_deim" in ok:
            better = "qr_deim" if ok["qr_deim"]["rel_l2"] < ok["rad"]["rel_l2"] \
                else "rad"
            print(f"\n{better} reached lower relative L2 error on this problem.")

        with_sharp = {k: v for k, v in ok.items() if "grad_rho_peak_ratio" in v}
        if with_sharp:
            print("\nLocalized shock capture (ratio to the exact Riemann "
                  "solution on the same grid; 1.0 = as sharp as exact, "
                  "< 1.0 = smeared):")
            print(f"{'strategy':16s} {'rho_peak':>10s} {'|drho/dx|':>10s} "
                  f"{'|dp/dx|':>10s}")
            for arm in ARMS:
                if arm not in with_sharp:
                    continue
                r = with_sharp[arm]
                print(f"{arm:16s} {r['rho_peak_ratio']:10.3f} "
                      f"{r['grad_rho_peak_ratio']:10.3f} "
                      f"{r['grad_p_peak_ratio']:10.3f}")
            sharpest = max(with_sharp,
                           key=lambda k: with_sharp[k]["grad_rho_peak_ratio"])
            print(f"Sharpest density jump on this run: '{sharpest}' "
                  f"(this can disagree with the rel-L2 ranking -- see "
                  f"Section on Ramp NS, where it does).")

    plot_migration(rows, prob, os.path.join(
        RESULTS_DIR, "qr_deim_toro3_collocation_migration.png"))
    plot_solutions(rows, os.path.join(
        RESULTS_DIR, "qr_deim_toro3_solutions.png"))

    # Save the raw point/field arrays separately (large -- npz, not JSON) so
    # the plots above can be regenerated later with --replot, without
    # rerunning training; then strip them before the human-readable JSON.
    save_raw_arrays("ablation_qr_deim_toro3", rows)
    rows_for_json = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in rows.items()}
    save_result("ablation_qr_deim_toro3", {
        "problem": "toro3_blast_wave", "epochs": args.epochs, "seed": args.seed,
        "device": info,
        "metric": "relative L2 vs exact Riemann solution (domain-averaged); "
        "plus localized shock-capture ratios rho_peak / |drho/dx| / |dp/dx| "
        "vs the exact solution on a common grid (see _shock_sharpness)",
        "art_visc": FIXED_AV, "arms": rows_for_json,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
