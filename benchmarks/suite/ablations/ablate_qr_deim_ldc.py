"""Ablation: QR-DEIM-R vs RAD vs no resampling, on the 2-D lid-driven cavity
(LDC) at Re=100 -- a smooth (no shock), steady, incompressible-flow problem
with corner singularities where the moving lid meets the stationary walls.

Companion to ``ablate_qr_deim.py`` (Toro-3, 1-D blast wave) and
``ablate_qr_deim_ramp_ns.py`` (2-D viscous SBLI). Those two are shock
problems whose captured-jump sharpness is capped by a fixed
artificial-viscosity coefficient, which limits how much *any* collocation
strategy can help. LDC has no such cap: its solution is genuinely smooth
away from the two top corners, so this is a cleaner test of whether
adaptive collocation helps at all in this framework. Mirrors
``examples/LDC/run_ldc.py``'s geometry, network (FBPINN + SimpleGate,
single subdomain), Re, and loss weights, varying only the interior pool's
resampling strategy:

  none       static interior pool, no adaptive resampling
  rad        RAR-D/RAD magnitude-weighted resampling
  qr_deim    QR-DEIM-R deterministic resampling

each at full-pool replacement and at an 80/20 hybrid split, as in the
other two scripts. Scored against the Fluent Re=100 reference field
(``examples/LDC/re100.csv``, 201x201): full-field relative L2 of the
velocity magnitude, plus the two standard Ghia-style centreline profiles
(u along x=0.5, v along y=0.5).

Run:
    python benchmarks/suite/ablations/ablate_qr_deim_ldc.py --epochs 10000
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
import pandas as pd                                                # noqa: E402

from common import (base_parser, jax_device_info, save_result,     # noqa: E402
                    timed, warn_if_cpu)

from underPINN.nn.fbpinn import FBPINN                              # noqa: E402
from underPINN.nn.attention import SimpleGate                      # noqa: E402
from underPINN.pde.navier_stokes import NavierStokesPDE            # noqa: E402
from underPINN.utils.metrics import relative_l2_error              # noqa: E402
from underPINN.utils.sampling import qr_deim_resample, rad_resample  # noqa: E402

RE = 100.0
LAYERS = [2, 64, 64, 64, 64, 3]
LR = 1e-3
N_INT, N_EDGE = 10000, 300           # interior pool; boundary points per wall
BR, BB = 2048, 256
W_PDE, W_BC, W_LID = 1.0, 100.0, 50.0
REF_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))),
                       "examples", "LDC", "re100.csv")


def _build_model():
    return FBPINN(
        layers=LAYERS,
        shifts=jnp.array([[0.5, 0.5]]),
        xs_min=jnp.array([[0.0, 0.0]]),
        xs_max=jnp.array([[1.0, 1.0]]),
        smins=jnp.array([[0.4, 0.4]]),
        smaxs=jnp.array([[0.4, 0.4]]),
        attention_cls=SimpleGate,
    )


def make_problem(seed: int):
    rng = np.random.default_rng(seed)
    xy_int = rng.uniform(0.0, 1.0, (N_INT, 2)).astype("f4")
    t = np.linspace(0.0, 1.0, N_EDGE, dtype="f4")
    xy_lid = np.stack([t, np.ones_like(t)], axis=1)                 # y=1, u=1 v=0
    xy_wall = np.concatenate([                                       # u=v=0
        np.stack([np.zeros_like(t), t], axis=1),                     # x=0
        np.stack([np.ones_like(t), t], axis=1),                      # x=1
        np.stack([t, np.zeros_like(t)], axis=1),                     # y=0
    ], axis=0)

    def domain_sampler(n, s):
        r = np.random.default_rng(s)
        return r.uniform(0.0, 1.0, (n, 2)).astype("f4")

    # ── Fluent Re=100 reference field (201x201 uniform grid) ──
    df = pd.read_csv(REF_CSV, skipinitialspace=True)
    df = df.sort_values(by=["y-coordinate", "x-coordinate"]).reset_index(drop=True)
    n = int(round(len(df) ** 0.5))
    u_ref = df["x-velocity"].values.reshape(n, n).astype("f4")       # [iy, ix]
    v_ref = df["y-velocity"].values.reshape(n, n).astype("f4")
    xr = df["x-coordinate"].values.reshape(n, n)[0]
    yr = df["y-coordinate"].values.reshape(n, n)[:, 0]

    return dict(xy_int=jnp.array(xy_int), xy_lid=jnp.array(xy_lid),
                xy_wall=jnp.array(xy_wall), domain_sampler=domain_sampler,
                u_ref=u_ref, v_ref=v_ref, xr=xr, yr=yr, n_ref=n)


ARM_CONFIG = {
    "none":           (None,      1.0),
    "rad":            ("rad",     1.0),
    "qr_deim":        ("qr_deim", 1.0),
    "qr_deim_hybrid": ("qr_deim", 0.2),
    "rad_hybrid":     ("rad",     0.2),
}
ARMS = {
    "none": "static interior pool, no adaptive resampling",
    "rad": "RAR-D/RAD magnitude-weighted resampling, full pool replaced",
    "qr_deim": "QR-DEIM-R resampling, full pool replaced",
    "qr_deim_hybrid": "QR-DEIM-R, 80% pool fixed / 20% adaptive",
    "rad_hybrid": "RAD, 80% pool fixed / 20% adaptive",
}


def run_arm(arm: str, epochs: int, seed: int, prob, resample_period: int) -> dict:
    strategy, adaptive_frac = ARM_CONFIG[arm]
    model = _build_model()
    pde = NavierStokesPDE(model, Re=RE)
    params = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 2)))

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=1e-2)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))
    state = opt.init(params)

    xy_lid, xy_wall = prob["xy_lid"], prob["xy_wall"]
    N_lid, N_wall = xy_lid.shape[0], xy_wall.shape[0]

    @jax.jit
    def step(p, s, r_b, lid_b, wall_b):
        def loss_fn(pp):
            res = pde.residual(pp, r_b)
            pde_l = jnp.mean(jnp.sum(res ** 2, axis=-1))
            uvp_lid = model.apply(pp, lid_b)
            lid_l = (jnp.mean((uvp_lid[:, 0] - 1.0) ** 2)
                     + jnp.mean(uvp_lid[:, 1] ** 2))
            uvp_w = model.apply(pp, wall_b)
            wall_l = jnp.mean(uvp_w[:, 0] ** 2) + jnp.mean(uvp_w[:, 1] ** 2)
            return W_PDE * pde_l + W_LID * lid_l + W_BC * wall_l
        loss, g = jax.value_and_grad(loss_fn)(p)
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, loss

    key = jax.random.PRNGKey(seed + 7)
    xy_r = prob["xy_int"]
    n_adapt = max(1, round(adaptive_frac * N_INT)) if strategy is not None else 0
    n_fixed = N_INT - n_adapt
    xy_fixed = xy_r[:n_fixed] if n_fixed > 0 else None

    def train():
        nonlocal xy_r
        p, s, k = params, state, key
        loss = None
        for ep in range(epochs):
            if strategy is not None and ep > 0 and ep % resample_period == 0:
                if strategy == "rad":
                    new = rad_resample(pde, p, prob["domain_sampler"],
                                       n_keep=n_adapt, n_candidates=5 * n_adapt,
                                       k=1.0, c=1.0, seed=seed + ep)
                else:
                    new = qr_deim_resample(pde, p, prob["domain_sampler"],
                                           n_keep=n_adapt,
                                           n_candidates=5 * n_adapt, seed=seed + ep)
                ap_ = jnp.asarray(new)
                xy_r = (jnp.concatenate([xy_fixed, ap_], axis=0)
                        if xy_fixed is not None else ap_)
            k, k1, k2, k3 = jax.random.split(k, 4)
            ir = jax.random.randint(k1, (BR,), 0, N_INT)
            il = jax.random.randint(k2, (BB,), 0, N_lid)
            iw = jax.random.randint(k3, (BB,), 0, N_wall)
            p, s, loss = step(p, s, xy_r[ir], xy_lid[il], xy_wall[iw])
        loss.block_until_ready()
        return p, float(loss)

    (final_params, final_loss), wall = timed(train)

    # ── score against the Fluent Re=100 field ──
    n = prob["n_ref"]
    XX, YY = np.meshgrid(prob["xr"], prob["yr"], indexing="xy")     # [iy, ix]
    pts = jnp.array(np.stack([XX.ravel(), YY.ravel()], axis=1), "f4")
    uvp = np.array(model.apply(final_params, pts))
    u_p = uvp[:, 0].reshape(n, n)
    v_p = uvp[:, 1].reshape(n, n)
    u_ref, v_ref = prob["u_ref"], prob["v_ref"]

    mag_p = np.sqrt(u_p ** 2 + v_p ** 2)
    mag_r = np.sqrt(u_ref ** 2 + v_ref ** 2)
    rel_l2_mag = float(relative_l2_error(jnp.array(mag_p), jnp.array(mag_r)))
    rel_l2_u = float(relative_l2_error(jnp.array(u_p), jnp.array(u_ref)))
    rel_l2_v = float(relative_l2_error(jnp.array(v_p), jnp.array(v_ref)))

    ic = n // 2                                                     # centre index
    rel_l2_ucl = float(relative_l2_error(jnp.array(u_p[:, ic]),
                                         jnp.array(u_ref[:, ic])))   # u vs y at x=0.5
    rel_l2_vcl = float(relative_l2_error(jnp.array(v_p[ic, :]),
                                         jnp.array(v_ref[ic, :])))   # v vs x at y=0.5

    return {"strategy": arm, "base_method": strategy,
            "adaptive_frac": adaptive_frac, "n_fixed": n_fixed, "n_adapt": n_adapt,
            "epochs": epochs, "wall_s": wall, "ms_per_epoch": 1e3 * wall / epochs,
            "final_loss": final_loss,
            "rel_l2": rel_l2_mag, "rel_l2_u": rel_l2_u, "rel_l2_v": rel_l2_v,
            "rel_l2_u_centreline": rel_l2_ucl, "rel_l2_v_centreline": rel_l2_vcl}


def main() -> int:
    ap = base_parser("Ablate QR-DEIM-R vs RAD vs no resampling on the lid-driven cavity")
    ap.set_defaults(epochs=10000)
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--resample-period", type=int, default=500)
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
            r = run_arm(arm, args.epochs, args.seed, prob, args.resample_period)
            rows[arm] = r
            print(f"    {r['ms_per_epoch']:6.2f} ms/ep  loss={r['final_loss']:.4e}"
                  f"  rel_L2(|U|)={r['rel_l2']:.4e}  u_cl={r['rel_l2_u_centreline']:.4e}"
                  f"  v_cl={r['rel_l2_v_centreline']:.4e}")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            rows[arm] = {"error": f"{type(e).__name__}: {e}"}

    ok = {k: v for k, v in rows.items() if "error" not in v}
    if ok:
        base = ok.get("none", {}).get("rel_l2")
        print("\n" + "=" * 78)
        print(f"{'strategy':16s} {'ms/ep':>8s} {'rel L2 |U|':>11s} {'u_cl':>10s} "
              f"{'v_cl':>10s} {'vs none':>9s}")
        print("-" * 78)
        for arm in ARMS:
            if arm not in ok:
                continue
            r = ok[arm]
            rel = f"{base / r['rel_l2']:.2f}x" if base else "-"
            print(f"{arm:16s} {r['ms_per_epoch']:8.2f} {r['rel_l2']:11.4e} "
                  f"{r['rel_l2_u_centreline']:10.4e} {r['rel_l2_v_centreline']:10.4e} "
                  f"{rel:>9s}")
        print("=" * 78)

    save_result("ablation_qr_deim_ldc", {
        "problem": "lid_driven_cavity_re100", "epochs": args.epochs,
        "seed": args.seed, "device": info,
        "metric": "relative L2 of velocity magnitude vs Fluent Re=100 field; "
        "plus u (x=0.5) and v (y=0.5) centreline profiles",
        "Re": RE, "arms": rows,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
