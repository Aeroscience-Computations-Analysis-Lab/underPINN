"""underPINN's own JAX PINN on the *identical* 1-D Burgers setup used by
``burgers1d/burgers1d.py`` (NVIDIA PhysicsNeMo Sym) -- same physics, same
architecture, same collocation/IC/BC counts, same IC (u0_mode=1, the
classic ``-sin(pi x)``), scored against the identical Cole-Hopf exact
reference (``underPINN.utils.operator_datagen.burgers1d_exact``), so the
two frameworks' numbers are directly comparable rather than each reporting
its own metric.

Run with the *base* environment's JAX (not the physicsnemo venv):
    python benchmarks/suite/physicsnemo/compare_underpinn.py --epochs 5000
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "baselines"))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
import numpy as np                                            # noqa: E402
import optax                                                  # noqa: E402
from flax import linen as fnn                                 # noqa: E402

from common import base_parser, jax_device_info, save_result, timed, warn_if_cpu  # noqa: E402
from burgers_baselines import (LAYERS, LR, N_BC, N_IC, N_R,    # noqa: E402
                               NU, T_MAX, W_BC, W_IC, make_data)

from underPINN.utils.operator_datagen import burgers1d_exact   # noqa: E402
from underPINN.utils.metrics import relative_l2_error          # noqa: E402
from underPINN.utils.sampling import qr_deim_resample, rad_resample  # noqa: E402

_RESAMPLERS = {"qr_deim": qr_deim_resample, "rad": rad_resample}


class MLPNet(fnn.Module):
    @fnn.compact
    def __call__(self, xt):
        h = xt
        for w in LAYERS[1:-1]:
            h = jnp.tanh(fnn.Dense(w)(h))
        return fnn.Dense(LAYERS[-1])(h)


class _BurgersResidual:
    """Minimal ``pde.residual(params, xt)`` wrapper around the same inline
    Cole-Hopf-PDE computation ``loss_fn`` below does, so
    ``qr_deim_resample``/``rad_resample`` (which expect a ``pde`` object with
    a ``.residual`` method) can be reused as-is here without duplicating the
    ``burgers_baselines.py`` PyTorch-side residual machinery."""

    def __init__(self, model, nu):
        self.model = model
        self.nu = nu

    def residual(self, params, xt):
        # Fused vjp+jvp -- same redundant jacfwd+hessian+apply pattern
        # profiled and fixed in underPINN/pde/burgers.py, reproduced here
        # since this class is a self-contained duplicate of that residual
        # (not an import of it) rather than left stale, matching the fix
        # now applied to the timed step() below.
        def u_single(xy_i):
            return self.model.apply(params, xy_i[None, :])[0, 0]

        def per_point(xy_i):
            u_val, vjp_fn = jax.vjp(u_single, xy_i)
            grad_vec = vjp_fn(1.0)[0]

            def grad_only(z):
                return jax.vjp(u_single, z)[1](1.0)[0]

            _, jvp_out = jax.jvp(grad_only, (xy_i,), (jnp.array([1.0, 0.0]),))
            return u_val, grad_vec[0], grad_vec[1], jvp_out[0]

        u, ux, ut, uxx = jax.vmap(per_point)(xt)
        return ut + u * ux - self.nu * uxx


def run_underpinn_jax(epochs: int, seed: int, sampling: str = "uniform",
                      resample_period: int = 500,
                      adaptive_frac: float = 1.0) -> dict:
    data = make_data(seed)
    x_r, t_r, x_ic, u_ic, x_bc, t_bc = data

    model = MLPNet()
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 2)))
    pde = _BurgersResidual(model, NU)

    def u_single(p, xt):
        return model.apply(p, xt[None, :])[0, 0]

    # Fused vjp+jvp in place of the separate jax.jacfwd + jax.hessian +
    # plain model.apply calls this loss_fn used to make -- the same
    # redundant-AD-transform pattern profiled and fixed in
    # underPINN/pde/burgers.py, reproduced here since loss_fn is a
    # self-contained residual computation, not an import of that class.
    # Verified against the legacy jacfwd/hessian/apply formulation in
    # tests/test_pde_burgers_residual.py (this is the identical residual
    # expression, only the calling convention around `model.apply` differs
    # -- (p, xt) here as two args vs. (params, xt) closed over there).
    def _per_point(p, xt_i):
        def scalar_fn(z):
            return u_single(p, z)

        u_val, vjp_fn = jax.vjp(scalar_fn, xt_i)
        grad_vec = vjp_fn(1.0)[0]

        def grad_only(z):
            return jax.vjp(scalar_fn, z)[1](1.0)[0]

        _, jvp_out = jax.jvp(grad_only, (xt_i,), (jnp.array([1.0, 0.0]),))
        return u_val, grad_vec[0], grad_vec[1], jvp_out[0]

    def fused_burgers(p, xt_pool):
        return jax.vmap(_per_point, in_axes=(None, 0))(p, xt_pool)

    XR0 = jnp.stack([jnp.array(x_r), jnp.array(t_r)], axis=1)
    XI = jnp.stack([jnp.array(x_ic), jnp.zeros_like(jnp.array(x_ic))], axis=1)
    UI = jnp.array(u_ic)
    XB = jnp.stack([jnp.array(x_bc), jnp.array(t_bc)], axis=1)

    def domain_sampler(n, sd):
        r = np.random.default_rng(sd)
        return np.stack([r.uniform(-1.0, 1.0, n), r.uniform(0.0, T_MAX, n)],
                        axis=1).astype(np.float32)

    def loss_fn(p, xr_pool):
        u, ux, ut, uxx = fused_burgers(p, xr_pool)
        res = ut + u * ux - NU * uxx
        pde_l = jnp.mean(res ** 2)
        ic_l = jnp.mean((model.apply(p, XI)[:, 0] - UI) ** 2)
        bc_l = jnp.mean(model.apply(p, XB)[:, 0] ** 2)
        return pde_l + W_IC * ic_l + W_BC * bc_l

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=1e-2)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))

    # Burgers here is trained full-batch every step (matches PhysicsNeMo's
    # `fixed_dataset=True, batch_size=full pool`, see the README table), so
    # `xr_pool` is the *entire* N_R-point pool, not a per-step minibatch --
    # "adaptive" resampling replaces the whole pool every `resample_period`
    # epochs rather than picking a fresh minibatch index every step.
    @jax.jit
    def step(p, s, xr_pool):
        loss, g = jax.value_and_grad(loss_fn)(p, xr_pool)
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, loss

    state0 = opt.init(params0)
    _p, _s, warm = step(params0, state0, XR0)
    warm.block_until_ready()          # compile once, untimed
    resample_fn = _RESAMPLERS.get(sampling)

    # adaptive_frac < 1.0: freeze the first N_R*(1-adaptive_frac) points of
    # the pool for the whole run (uniform, for stability) and only ever
    # resample the remaining N_R*adaptive_frac (adaptive, for improvement) --
    # same majority-fixed/minority-adaptive split as
    # ablations/ablate_qr_deim_ramp_ns.py, see _timed_train_adaptive in
    # compare_underpinn_multi.py for the fuller rationale.
    if resample_fn is not None:
        n_adapt = max(1, round(adaptive_frac * N_R))
        n_fixed = N_R - n_adapt
    else:
        n_adapt, n_fixed = 0, N_R
    fixed_pool = XR0[:n_fixed] if n_fixed > 0 else None

    def train():
        p, s, xr_pool, loss = params0, state0, XR0, None
        for ep in range(epochs):
            if resample_fn is not None and ep > 0 and ep % resample_period == 0:
                kwargs = {"k": 1.0, "c": 1.0} if sampling == "rad" else {}
                new_pts = resample_fn(pde, p, domain_sampler, n_keep=n_adapt,
                                      n_candidates=5 * n_adapt,
                                      seed=seed + ep, **kwargs)
                adapt_pool = jnp.asarray(new_pts)
                xr_pool = (jnp.concatenate([fixed_pool, adapt_pool], axis=0)
                          if fixed_pool is not None else adapt_pool)
            p, s, loss = step(p, s, xr_pool)
        loss.block_until_ready()
        return p, float(loss)

    (final_params, final_loss), wall = timed(train)

    # ── score against the identical Cole-Hopf reference physicsnemo uses ──
    Nx_eval, Nt_eval = 101, 41
    x_eval = np.linspace(-1.0, 1.0, Nx_eval)
    t_eval = np.linspace(0.0, T_MAX, Nt_eval)
    u_exact = burgers1d_exact(x_eval, t_eval, nu=NU, u0_mode=1)

    XX, TT = np.meshgrid(x_eval, t_eval, indexing="ij")
    xt_query = jnp.array(np.stack([XX.ravel(), TT.ravel()], axis=1).astype(np.float32))
    u_pred = np.array(model.apply(final_params, xt_query)[:, 0]).reshape(Nx_eval, Nt_eval)

    rel_l2 = float(relative_l2_error(u_pred, u_exact))
    return {"framework": "underpinn-jax-jit", "epochs": epochs, "wall_s": wall,
            "ms_per_epoch": 1e3 * wall / epochs, "final_loss": final_loss,
            "rel_l2_vs_cole_hopf": rel_l2, "sampling": sampling,
            "adaptive_frac": adaptive_frac,
            "n_r": N_R, "n_ic": N_IC, "n_bc": N_BC, "layers": LAYERS}


def main() -> int:
    ap = base_parser("underPINN JAX Burgers, scored on the same Cole-Hopf "
                     "reference as the physicsnemo comparison")
    ap.set_defaults(epochs=5000)
    ap.add_argument("--sampling", default="uniform",
                    choices=["uniform", "qr_deim", "rad"],
                    help="'uniform': fixed full-batch pool drawn once "
                         "(default). 'qr_deim'/'rad': periodically replace "
                         "the whole pool with an adaptive resample based on "
                         "the live PDE residual.")
    ap.add_argument("--resample-period", type=int, default=500)
    ap.add_argument("--adaptive-frac", type=float, default=1.0,
                    help="fraction of the pool subject to resampling "
                         "(ignored for --sampling uniform); <1.0 keeps the "
                         "rest permanently fixed (uniform, for stability).")
    args = ap.parse_args()

    info = jax_device_info(require_gpu=not args.allow_cpu)
    warn_if_cpu(info)
    print(f"JAX backend: {info['platform']} ({info['device_name']})")
    print(f"Epochs: {args.epochs}   seed: {args.seed}   "
         f"sampling: {args.sampling}   resample_period: "
         f"{args.resample_period}   adaptive_frac: {args.adaptive_frac}\n")

    r = run_underpinn_jax(args.epochs, args.seed, sampling=args.sampling,
                          resample_period=args.resample_period,
                          adaptive_frac=args.adaptive_frac)
    print(f"underPINN (JAX jit): {r['ms_per_epoch']:.3f} ms/ep  "
         f"final_loss={r['final_loss']:.4e}  "
         f"rel_L2(Cole-Hopf)={r['rel_l2_vs_cole_hopf']:.4e}")

    frac_tag = "" if args.adaptive_frac >= 1.0 else f"_frac{args.adaptive_frac:g}"
    tag = ("physicsnemo_compare_underpinn" if args.sampling == "uniform"
          else f"physicsnemo_compare_underpinn_{args.sampling}{frac_tag}")
    save_result(tag, {
        "problem": "burgers_1d", "device": info, **r,
    })

    # Merge with physicsnemo's own result.json, if present, into one table.
    pn_path = os.path.join(_HERE, "burgers1d", "result.json")
    if os.path.exists(pn_path):
        with open(pn_path) as fh:
            pn = json.load(fh)
        print("\n" + "=" * 78)
        print(f"{'framework':24s} {'ms/epoch':>10s} {'rel_L2 (Cole-Hopf)':>20s}")
        print("-" * 78)
        print(f"{'underpinn-jax-jit':24s} {r['ms_per_epoch']:10.3f} "
             f"{r['rel_l2_vs_cole_hopf']:20.4e}")
        print(f"{'nvidia-physicsnemo-sym':24s} {pn['ms_per_epoch']:10.3f} "
             f"{pn['rel_l2_vs_cole_hopf']:20.4e}")
        print("=" * 78)
        faster = "underPINN" if r["ms_per_epoch"] < pn["ms_per_epoch"] else "PhysicsNeMo"
        more_acc = ("underPINN" if r["rel_l2_vs_cole_hopf"] < pn["rel_l2_vs_cole_hopf"]
                    else "PhysicsNeMo")
        print(f"\n{faster} is faster per epoch; {more_acc} reached lower "
             f"error at {args.epochs} epochs on this run.")
    else:
        print(f"\n(physicsnemo result not found at {pn_path} -- run "
             f"burgers1d/burgers1d.py first for the side-by-side table)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
