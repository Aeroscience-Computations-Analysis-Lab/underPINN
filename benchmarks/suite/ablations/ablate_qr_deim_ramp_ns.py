"""Ablation: QR-DEIM-R vs RAD vs no resampling, on the *real* flagship RAR-D
problem -- 2-D viscous compression-ramp SBLI (Ramp NS).

``ablate_qr_deim.py`` tested QR-DEIM-R against RAD on the 1-D Toro-3 blast
wave. Ramp NS is the paper's own actual production-scale RAR-D use case
(``dispatch_parity.py`` and the paper text both single it out: "For Ramp NS
specifically, the RAR-D collocation resampling ... calls non-jittable
NumPy-based rejection sampling"), with a much larger network (~66k
params vs Toro-3's), a much larger collocation pool, and a genuine 2-D
shock-boundary-layer interaction rather than a 1-D Riemann problem. This
script mirrors ``RampNSEvaluator.train()``
(``underPINN/benchmark_utils/evaluators.py``) *exactly* -- same geometry,
network, loss weights, batch sizes, RAR cadence -- varying only the
**adaptive interior pool's resampling strategy**:

  none       xy_adapt frozen at its initial draw -- no adaptive resampling
  rad        RampNSEvaluator's existing RAR-D/RAD resampling (unchanged)
  qr_deim    the new QR-DEIM-R resampler (underPINN/utils/sampling.py)

Scored exactly as RampNSEvaluator.evaluate() does: relative L2 of the
predicted Mach field against the analytic oblique-shock solution, restricted
to the outer flow (a near-wall band is excluded -- the boundary layer there
is real viscous physics no inviscid reference can capture).

Run:
    python benchmarks/suite/ablations/ablate_qr_deim_ramp_ns.py --epochs 5000
"""
from __future__ import annotations

import math
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

from underPINN.geometry.ramp import RampGeometry                    # noqa: E402
from underPINN.nn.mlp import MLP                                    # noqa: E402
from underPINN.pde.compressible_euler import CompressibleEulerPDE   # noqa: E402
from underPINN.pde.compressible_ns_2d import CompressibleNS2DPDE    # noqa: E402
from underPINN.utils.metrics import relative_l2_error               # noqa: E402
from underPINN.utils.sampling import qr_deim_resample, rad_resample # noqa: E402

M_INF, THETA_DEG, GAMMA, RE, PR = 3.0, 15.0, 1.4, 1.0e4, 0.72
L, H, RAMP_START, SLIP_END = 2.0, 1.0, 0.8, 0.15
LAYERS = [2, 128, 128, 128, 128, 128, 4]
LR = 1e-3
ART_VISC = 2e-3
N_ADAPT, RAR_X_MIN = 6000, 0.25
W_PDE, W_INLET, W_WALL, W_SLIP, W_UPPER = 1.0, 100.0, 100.0, 80.0, 20.0


def make_problem(seed: int):
    geom = RampGeometry(THETA_DEG, L=L, H=H,
                        ramp_start=RAMP_START, slip_end=SLIP_END)
    xy_uniform = geom.sample_interior(40000, seed=seed)
    xy_bl = geom.sample_boundary_layer(5000, beta=4.0, seed=seed + 7)
    xy_adapt0 = geom.sample_interior(N_ADAPT, seed=seed + 101, x_min=RAR_X_MIN)
    return dict(
        geom=geom,
        xy_uniform=xy_uniform, xy_bl=xy_bl, xy_adapt0=xy_adapt0,
        xy_in=jnp.array(np.array(geom.sample_inlet(200), "f4")),
        xy_w=jnp.array(np.array(geom.sample_noslip_wall(200), "f4")),
        xy_slip=jnp.array(np.array(geom.sample_slip_wall(100), "f4")),
        xy_up=jnp.array(np.array(geom.sample_upper(150), "f4")),
    )


def _outer_mask(geom, XX, YY, mask, band_frac=0.12):
    band = band_frac * H
    return mask & (YY > geom.y_wall(XX) + band)


# arm name -> (base resampling method or None, size of the adaptive
# sub-pool). xy_uniform (40,000) + xy_bl (5,000) = 45,000 points are always
# fixed for the whole run regardless of arm; only the adaptive sub-pool is
# ever resampled -- this design already keeps a majority fixed
# (N_ADAPT=6,000 -> ~11.8% adaptive of the 51,000-point total). The
# "_hybrid" arms raise that to exactly the 20% adaptive fraction used in
# benchmarks/suite/physicsnemo/compare_underpinn_multi.py's
# --adaptive-frac 0.2 (n_adapt s.t. n_adapt/(45000+n_adapt) = 0.2), to test
# that standardized split here too.
_N_FIXED = 40000 + 5000
N_ADAPT_HYBRID = round(0.2 / 0.8 * _N_FIXED)   # 11,250
ARM_CONFIG = {
    "none":           (None,      N_ADAPT),
    "rad":            ("rad",     N_ADAPT),
    "qr_deim":        ("qr_deim", N_ADAPT),
    "qr_deim_hybrid": ("qr_deim", N_ADAPT_HYBRID),
    "rad_hybrid":     ("rad",     N_ADAPT_HYBRID),
}


def run_arm(arm: str, epochs: int, seed: int, prob,
           resample_period: int = 500, art_visc: float = ART_VISC) -> dict:
    """*arm* indexes :data:`ARM_CONFIG` for the resampling method (or
    ``None``) and the adaptive sub-pool size. ``art_visc`` overrides the
    module-level default -- the fixed artificial-viscosity coefficient caps
    how sharp *any* arm's captured shock can get (it directly damps
    d^2U/dx^2), independent of collocation strategy; sweeping it (see the
    ``--art-visc`` CLI flag below) tests whether that cap, not the
    resampling method, is why every arm's peak |grad(rho)| clusters so
    closely together. ``ablate_artificial_viscosity.py`` runs the analogous
    sweep on Toro-3 (1-D); this is Ramp NS's own, added directly here."""
    strategy, n_adapt = ARM_CONFIG[arm]
    geom = prob["geom"]
    xy_uniform, xy_bl = prob["xy_uniform"], prob["xy_bl"]
    xy_adapt = np.array(
        geom.sample_interior(n_adapt, seed=seed + 101, x_min=RAR_X_MIN))
    xy_adapt_init = xy_adapt.copy()        # kept for the migration plot
    xy_r = jnp.array(np.concatenate([xy_uniform, xy_bl, xy_adapt], axis=0))

    model = MLP(layers=LAYERS)
    pde = CompressibleNS2DPDE(model, gamma=GAMMA, M_inf=M_INF, Re=RE, Pr=PR,
                              art_visc=art_visc)
    T0 = pde.total_temperature()
    rho_inf, u_inf, v_inf, T_inf = pde.freestream()

    params = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 2)))
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    sched = optax.cosine_decay_schedule(LR, epochs, alpha=1e-2)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))
    state = opt.init(params)

    xy_in, xy_w, xy_slip, xy_up = (
        prob["xy_in"], prob["xy_w"], prob["xy_slip"], prob["xy_up"])
    N_r = xy_r.shape[0]
    N_in, N_w = xy_in.shape[0], xy_w.shape[0]
    N_slip, N_up = xy_slip.shape[0], xy_up.shape[0]
    bR, bI, bW, bS, bU = (min(1536, N_r), min(250, N_in), min(250, N_w),
                          min(180, N_slip), min(180, N_up))

    @jax.jit
    def step(params, state, r_b, in_b, w_b, slip_b, up_b):
        def loss_fn(p):
            res = pde.residual(p, r_b)
            pde_l = jnp.mean(jnp.sum(res ** 2, axis=-1))
            pv_in = pde.apply(p, in_b)
            in_l = (jnp.mean((pv_in[:, 0] - rho_inf) ** 2)
                    + jnp.mean((pv_in[:, 1] - u_inf) ** 2)
                    + jnp.mean((pv_in[:, 2] - v_inf) ** 2)
                    + jnp.mean((pv_in[:, 3] - T_inf) ** 2))
            pv_w = pde.apply(p, w_b)
            wall_l = (jnp.mean(pv_w[:, 1] ** 2) + jnp.mean(pv_w[:, 2] ** 2)
                      + jnp.mean((pv_w[:, 3] - T0) ** 2))
            pv_s = pde.apply(p, slip_b)
            slip_l = jnp.mean(pv_s[:, 2] ** 2)
            pv_up = pde.apply(p, up_b)
            up_l = (jnp.mean((pv_up[:, 0] - rho_inf) ** 2)
                    + jnp.mean((pv_up[:, 1] - u_inf) ** 2)
                    + jnp.mean((pv_up[:, 2] - v_inf) ** 2)
                    + jnp.mean((pv_up[:, 3] - T_inf) ** 2))
            total = (W_PDE * pde_l + W_INLET * in_l + W_WALL * wall_l
                     + W_SLIP * slip_l + W_UPPER * up_l)
            return total, pde_l
        (total, _pl), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = opt.update(grads, state)
        return optax.apply_updates(params, updates), state, total

    rar_period = max(1, resample_period)
    key = jax.random.PRNGKey(seed + 11)

    def train():
        nonlocal xy_r, xy_adapt
        p, s, key_ = params, state, key
        total = None
        for ep in range(epochs):
            if strategy is not None and ep > 0 and ep % rar_period == 0:
                domain_sampler = (
                    lambda n, sd: geom.sample_interior(n, seed=sd, x_min=RAR_X_MIN))
                if strategy == "rad":
                    xy_adapt = rad_resample(
                        pde, p, domain_sampler,
                        n_keep=n_adapt, n_candidates=5 * n_adapt,
                        k=1.0, c=1.0, seed=seed + ep)
                else:  # "qr_deim"
                    xy_adapt = qr_deim_resample(
                        pde, p, domain_sampler,
                        n_keep=n_adapt, n_candidates=5 * n_adapt, seed=seed + ep)
                xy_r = jnp.array(
                    np.concatenate([xy_uniform, xy_bl, xy_adapt], axis=0))
            key_, k1, k2, k3, k4, k5 = jax.random.split(key_, 6)
            ir = jax.random.randint(k1, (bR,), 0, N_r)
            ii = jax.random.randint(k2, (bI,), 0, N_in)
            iw = jax.random.randint(k3, (bW,), 0, N_w)
            isl = jax.random.randint(k4, (bS,), 0, N_slip)
            iu = jax.random.randint(k5, (bU,), 0, N_up)
            p, s, total = step(p, s, xy_r[ir], xy_in[ii], xy_w[iw],
                               xy_slip[isl], xy_up[iu])
        total.block_until_ready()
        return p, float(total)

    (final_params, final_loss), wall = timed(train)

    # ── score: relative L2 of outer-flow Mach vs the analytic oblique-shock
    #    field, exactly as RampNSEvaluator.evaluate() does.
    XX, YY, mask = geom.make_grid(Nx=140, Ny=110)
    outer = _outer_mask(geom, XX, YY, mask)
    pts = jnp.array(np.stack([XX.ravel(), YY.ravel()], axis=1), "f4")
    mach_pred = np.array(pde.mach(final_params, pts)).reshape(XX.shape)

    euler = CompressibleEulerPDE(None, gamma=GAMMA)
    shock = euler.oblique_shock(M_INF, THETA_DEG)
    beta = math.radians(shock["beta_deg"])
    dx = np.maximum(XX - RAMP_START, 0.0)
    below = (YY <= dx * math.tan(beta)) & (XX >= RAMP_START)
    mach_exact = np.where(below, shock["M2"], M_INF)

    rel_l2 = float(relative_l2_error(jnp.array(mach_pred[outer]),
                                     jnp.array(mach_exact[outer])))

    # ── density-gradient magnitude |grad(rho)| -- a schlieren-like field:
    #    a genuine shock is a near-discontinuous jump in rho, so a sharper,
    #    taller |grad(rho)| spike right at the shock means the PINN is
    #    resolving that jump more faithfully; a method that smears the
    #    shock over a wider band shows a lower, broader peak instead. This
    #    has no "exact" counterpart to score against (the true field is a
    #    literal delta function at the shock), so it is reported as a
    #    per-arm peak value to compare arms against each other, not against
    #    a reference. ──
    def _rho_at(xy_i):
        return pde.apply(final_params, xy_i[None, :])[0, 0]

    rho_grad = jax.vmap(jax.grad(_rho_at))(pts)             # (N, 2)
    rho_grad_mag = np.array(jnp.linalg.norm(rho_grad, axis=-1)).reshape(XX.shape)

    # "near the shock" = within one grid-cell's worth of the analytic
    # oblique-shock line (same line plot_solutions/plot_migration draw),
    # restricted to x >= RAMP_START where that line is actually defined.
    band = 0.02 * H
    y_shock_of_x = np.maximum(XX - RAMP_START, 0.0) * math.tan(beta)
    near_shock = (XX >= RAMP_START) & (np.abs(YY - y_shock_of_x) <= band)
    rho_grad_peak_near_shock = (float(np.max(rho_grad_mag[near_shock]))
                                if near_shock.any() else float("nan"))

    return {"strategy": arm, "base_method": strategy, "n_adapt": n_adapt,
            "art_visc": art_visc, "epochs": epochs, "wall_s": wall,
            "ms_per_epoch": 1e3 * wall / epochs, "final_loss": final_loss,
            "rel_l2": rel_l2, "n_params": n_params,
            "rho_grad_peak_near_shock": rho_grad_peak_near_shock,
            "_xy_init": xy_adapt_init, "_xy_final": np.array(xy_adapt),
            # xy_uniform + xy_bl -- always fixed, identical across every arm
            # and every epoch, kept here only so plot_migration can draw it
            # as grey background context (never plotted as if it moved).
            "_xy_fixed": np.concatenate([np.array(xy_uniform), np.array(xy_bl)]),
            "_mach_pred": mach_pred, "_rho_grad_mag": rho_grad_mag,
            "_mask": mask}


ARMS = {
    "none": "adaptive pool frozen at its initial draw -- no resampling",
    "rad": f"RampNSEvaluator's existing RAR-D/RAD resampling ({N_ADAPT}/{_N_FIXED + N_ADAPT} adaptive)",
    "qr_deim": f"QR-DEIM-R resampling ({N_ADAPT}/{_N_FIXED + N_ADAPT} adaptive)",
    "qr_deim_hybrid": f"QR-DEIM-R, standardized 20% adaptive ({N_ADAPT_HYBRID}/{_N_FIXED + N_ADAPT_HYBRID})",
    "rad_hybrid": f"RAD, standardized 20% adaptive ({N_ADAPT_HYBRID}/{_N_FIXED + N_ADAPT_HYBRID})",
}


def plot_migration(rows: dict, geom, out_path: str) -> None:
    """Initial vs. final (x, y) scatter, one column per arm, with the ramp
    wall and the analytic oblique-shock line overlaid -- the same reference
    geometry as RampNSEvaluator.plot() /
    ``pdf_images/ramp_ns_rar_migration.pdf`` in the paper. 'none' has
    identical initial/final panels by construction (a useful sanity check,
    not just a filler column).

    The always-fixed background (xy_uniform + xy_bl, 45,000 points, grey) is
    drawn under the adaptive sub-pool (blue/red) in *both* rows -- it is the
    same 45,000 points, unchanged, in every panel. Plotting only the
    adaptive sub-pool without this context (an earlier version of this
    figure) can read as if resampling moves points it never touches; this
    makes explicit which points are actually eligible to move."""
    ok = {k: v for k, v in rows.items() if "error" not in v and "_xy_init" in v}
    if not ok:
        return
    arms = [a for a in ARMS if a in ok]

    euler = CompressibleEulerPDE(None, gamma=GAMMA)
    shock = euler.oblique_shock(M_INF, THETA_DEG)
    beta = math.radians(shock["beta_deg"])
    x_shock = np.array([RAMP_START,
                        min(L, RAMP_START + H / max(math.tan(beta), 1e-9))])
    y_shock = (x_shock - RAMP_START) * math.tan(beta)
    x_wall = np.linspace(0.0, L, 200)
    y_wall = geom.y_wall(x_wall)

    fig, axes = plt.subplots(2, len(arms), figsize=(4.5 * len(arms), 7),
                             sharex=True, sharey=True)
    if len(arms) == 1:
        axes = axes[:, None]
    for col, arm in enumerate(arms):
        xy_init, xy_final = ok[arm]["_xy_init"], ok[arm]["_xy_final"]
        xy_fixed = ok[arm].get("_xy_fixed")
        for row, (xy, label) in enumerate(
                [(xy_init, "initial"), (xy_final, "final")]):
            ax = axes[row, col]
            if xy_fixed is not None:
                ax.scatter(xy_fixed[:, 0], xy_fixed[:, 1], s=1.0, alpha=0.1,
                          c="#a0aec0", label="fixed (uniform+BL)")
            ax.scatter(xy[:, 0], xy[:, 1], s=2.5, alpha=0.4, c="#c53030",
                      label="adaptive")
            ax.plot(x_wall, y_wall, "k-", lw=1)
            ax.plot(x_shock, y_shock, "r--", lw=1.3, label="shock")
            ax.set_title(f"{arm} — {label}" if row == 0 else label,
                        fontsize=10)
            ax.set_xlim(0, L)
            ax.set_ylim(0, H)
            if row == 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("y")
    axes[0, -1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Ramp NS: collocation pool, initial vs. final "
                "(grey = always-fixed uniform+BL, red = adaptive; "
                "dashed = analytic shock line)",
                fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCollocation migration plot -> {out_path}")


def plot_solutions(rows: dict, geom, out_path: str) -> None:
    """Mach field per arm vs. its own density-gradient magnitude, one row
    per arm: [PINN Mach field] [|grad(rho)|, schlieren-like] -- same
    grid/shock-line convention as RampNSEvaluator.plot().

    The second panel replaces what used to be the exact-solution reference
    -- a reviewer asked to see the density gradient instead, to compare
    which arm resolves the shock as an actually sharp jump rather than a
    smeared one. Unlike Mach, there is no simple closed-form |grad(rho)|
    reference to score against (the true field is a delta function at the
    shock), so each row is annotated with its own peak |grad(rho)| in a band
    around the analytic shock line (``rho_grad_peak_near_shock``, computed
    in ``run_arm``) -- directly comparable *across* arms (higher = sharper),
    even without a target to compare *to*. A third |error|-vs-exact panel
    used to sit alongside these two; dropped (rel L2 in the Mach panel's
    title already carries that number, and a reviewer asked for the density
    gradient over it) rather than kept as a third, less load-bearing column.
    """
    ok = {k: v for k, v in rows.items() if "error" not in v and "_mach_pred" in v}
    if not ok:
        return
    arms = [a for a in ARMS if a in ok]

    XX, YY, mask = geom.make_grid(Nx=140, Ny=110)
    euler = CompressibleEulerPDE(None, gamma=GAMMA)
    shock = euler.oblique_shock(M_INF, THETA_DEG)
    beta = math.radians(shock["beta_deg"])
    x_shock = np.array([RAMP_START,
                        min(L, RAMP_START + H / max(math.tan(beta), 1e-9))])
    y_shock = (x_shock - RAMP_START) * math.tan(beta)
    vmin, vmax = 0.0, M_INF + 0.2
    # |grad(rho)| colour scale, shared across arms so panel-to-panel colour
    # intensity is directly comparable. Clipped to the near-shock peaks
    # (not the field's raw max) -- the raw max is dominated by the
    # inlet/wall corner singularity at (x, y) = (0, 0) (a genuine geometric
    # singularity: the BC jumps discontinuously there between the inlet's
    # freestream condition and the wall's no-slip condition, RampGeometry's
    # own sample_interior(x_min=...) already routes around it for the same
    # reason), which is many times larger than anything at the actual
    # compression shock and, uncapped, saturates the whole colour scale so
    # the shock band itself barely registers. `extend="max"` marks the
    # clip visually rather than silently.
    near_shock_peaks = [v["rho_grad_peak_near_shock"] for v in ok.values()
                        if "rho_grad_peak_near_shock" in v]
    grad_vmax = 1.3 * max(near_shock_peaks) if near_shock_peaks else 1.0
    grad_levels = np.linspace(0.0, grad_vmax, 51)

    fig, axes = plt.subplots(len(arms), 2, figsize=(9, 3.6 * len(arms)),
                             squeeze=False)
    for row, arm in enumerate(arms):
        mach_pred = ok[arm]["_mach_pred"].copy().astype(float)
        mach_full = mach_pred.copy()
        mach_full[~ok[arm]["_mask"]] = np.nan
        grad_full = ok[arm].get("_rho_grad_mag")
        peak = ok[arm].get("rho_grad_peak_near_shock", float("nan"))
        if grad_full is not None:
            grad_full = grad_full.copy().astype(float)
            grad_full[~ok[arm]["_mask"]] = np.nan

        cf0 = axes[row, 0].contourf(XX, YY, mach_full, levels=50, cmap="jet",
                                    vmin=vmin, vmax=vmax)
        fig.colorbar(cf0, ax=axes[row, 0])
        axes[row, 0].set_title(f"{arm} — PINN Mach "
                              f"(rel_L2={ok[arm]['rel_l2']:.3f})", fontsize=9)

        if grad_full is not None:
            cf1 = axes[row, 1].contourf(XX, YY, grad_full, levels=grad_levels,
                                        cmap="inferno", extend="max")
            fig.colorbar(cf1, ax=axes[row, 1])
        axes[row, 1].set_title(f"|grad(rho)| (peak near shock={peak:.2f})",
                              fontsize=9)

        for ax in axes[row]:
            ax.plot(x_shock, y_shock, "k--", lw=1.2)
            ax.set_xlim(0, L)
            ax.set_ylim(0, H)
            ax.set_xlabel("x")
        axes[row, 0].set_ylabel("y")

    fig.suptitle("Ramp NS: PINN Mach and density-gradient magnitude "
                "(sharper = closer to a true shock), per resampling "
                "strategy", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Solution field plot -> {out_path}")


def main() -> int:
    ap = base_parser("Ablate QR-DEIM-R vs RAD vs no resampling on Ramp NS (SBLI)")
    ap.set_defaults(epochs=30000)
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--resample-period", type=int, default=500,
                    help="epochs between resamplings for the 'rad'/'qr_deim' "
                         "arms (default: 500; 'none' never resamples)")
    ap.add_argument("--replot", action="store_true",
                    help="skip training entirely and regenerate the plots "
                         "from the previous run's "
                         "results/ablation_qr_deim_ramp_ns_raw.npz + "
                         ".json -- run this after --arms once, not the "
                         "full 30,000-epoch training loop again, whenever "
                         "only the plotting code changes.")
    ap.add_argument("--art-visc", type=float, default=ART_VISC,
                    help=f"fixed artificial-viscosity coefficient (default "
                         f"{ART_VISC}), passed to CompressibleNS2DPDE for "
                         "every arm. Lower values relax the numerical "
                         "dissipation that stabilises the shock (see "
                         "CompressibleNS2DPDE's docstring) and are expected "
                         "to raise peak |grad(rho)| -- i.e. sharpen the "
                         "captured shock -- at the risk of Gibbs "
                         "oscillations / training instability if pushed too "
                         "low. Sweep this to find the actual sharpness "
                         "ceiling before crediting/blaming the resampling "
                         "method for the current small (~5%) inter-arm "
                         "spread.")
    ap.add_argument("--tag", default="",
                    help="suffix for the saved result/plot filenames (e.g. "
                         "'_av1e-3'), so a --art-visc sweep doesn't "
                         "overwrite the default run's results.")
    args = ap.parse_args()
    result_name = "ablation_qr_deim_ramp_ns" + args.tag
    migration_png = f"qr_deim_ramp_ns_collocation_migration{args.tag}.png"
    solutions_png = f"qr_deim_ramp_ns_solutions{args.tag}.png"

    if args.replot:
        rows = load_raw_arrays(result_name)
        saved = load_results(result_name)[result_name]
        for arm, scalars in saved["arms"].items():
            rows.setdefault(arm, {}).update(scalars)
        prob = make_problem(saved["seed"])
        print(f"Replotting from saved results (epochs={saved['epochs']}, "
             f"seed={saved['seed']}, {len(rows)} arms) -- no training run.\n")
        plot_migration(rows, prob["geom"], os.path.join(
            RESULTS_DIR, migration_png))
        plot_solutions(rows, prob["geom"], os.path.join(
            RESULTS_DIR, solutions_png))
        return 0

    info = jax_device_info(require_gpu=not args.allow_cpu)
    warn_if_cpu(info)
    print(f"JAX backend: {info['platform']} ({info['device_name']})")
    print(f"Epochs per arm: {args.epochs}   resample_period: "
          f"{args.resample_period}   art_visc: {args.art_visc}   "
          f"seed: {args.seed}\n")

    prob = make_problem(args.seed)
    rows = {}
    for arm in args.arms:
        print(f"--- {arm}: {ARMS[arm]}")
        try:
            r = run_arm(arm, args.epochs, args.seed, prob,
                       resample_period=args.resample_period,
                       art_visc=args.art_visc)
            rows[arm] = r
            print(f"    {r['ms_per_epoch']:6.2f} ms/ep  loss={r['final_loss']:.4e}"
                  f"  rel_L2(outer Mach)={r['rel_l2']:.4e}")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            rows[arm] = {"error": f"{type(e).__name__}: {e}"}

    ok = {k: v for k, v in rows.items() if "error" not in v}
    if ok:
        base = ok.get("none", {}).get("rel_l2")
        print("\n" + "=" * 84)
        print(f"{'strategy':10s} {'ms/ep':>8s} {'rel L2':>11s} {'vs none':>9s}   "
              f"description")
        print("-" * 84)
        for arm in ARMS:
            if arm not in ok:
                continue
            r = ok[arm]
            rel = f"{base / r['rel_l2']:.2f}x" if base else "-"
            print(f"{arm:10s} {r['ms_per_epoch']:8.2f} {r['rel_l2']:11.4e} "
                  f"{rel:>9s}   {ARMS[arm]}")
        print("=" * 84)
        best = min(ok, key=lambda k: ok[k]["rel_l2"])
        print(f"\nBest solution on this run: '{best}' "
              f"(rel_L2={ok[best]['rel_l2']:.4e}).")

        with_grad = {k: v for k, v in ok.items() if "rho_grad_peak_near_shock" in v}
        if with_grad:
            print("\nPeak |grad(rho)| near the compression shock "
                 "(higher = sharper-resolved shock, not scored against a "
                 "reference -- see plot_solutions):")
            for arm in sorted(with_grad, key=lambda k: -with_grad[k]["rho_grad_peak_near_shock"]):
                print(f"  {arm:16s} {with_grad[arm]['rho_grad_peak_near_shock']:8.3f}")
            sharpest = max(with_grad, key=lambda k: with_grad[k]["rho_grad_peak_near_shock"])
            print(f"Sharpest shock on this run: '{sharpest}'.")

    plot_migration(rows, prob["geom"], os.path.join(
        RESULTS_DIR, migration_png))
    plot_solutions(rows, prob["geom"], os.path.join(
        RESULTS_DIR, solutions_png))

    # Save the raw point/field arrays separately (large -- npz, not JSON) so
    # the plots above can be regenerated later with --replot, without
    # rerunning the 30,000-epoch training loop; then strip them before the
    # human-readable JSON.
    save_raw_arrays(result_name, rows)
    rows_for_json = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in rows.items()}
    save_result(result_name, {
        "problem": "ramp_ns_sbli", "epochs": args.epochs, "seed": args.seed,
        "resample_period": args.resample_period, "art_visc": args.art_visc,
        "device": info, "metric": "relative L2 of outer-flow Mach vs "
        "analytic oblique-shock field",
        "network_layers": LAYERS, "arms": rows_for_json,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
