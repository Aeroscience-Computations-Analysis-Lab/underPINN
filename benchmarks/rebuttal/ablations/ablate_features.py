"""Quantitative ablation of underPINN's advertised architectural features.

A reviewer noted that FBPINNs, gated attention, Fourier embeddings and RBA are
advertised but never ablated -- the paper claims they help without measuring
whether they do. This script measures each one against a matched plain-MLP
baseline on 1-D Burgers, scored on relative L^2 error against a high-fidelity
RK45 reference (``_burgers_reference``, the same reference BurgersEvaluator
uses), so the comparison is accuracy, not just loss value.

Every arm holds constant: the collocation sets, the loss weighting
(pde + 100*ic + 10*bc), Adam(1e-3) with cosine decay, the epoch budget, and
the seed. Only the ablated factor changes. Parameter counts are reported per
arm, because an architecture that wins only by being larger has not earned
its place in the paper.

Arms
----
mlp                plain tanh MLP  (baseline)
gated_mlp          GatedMLP -- the "gated attention" claim
fourier_mlp        FourierMLP -- trainable spectral embedding claim
fbpinn             FBPINN domain decomposition claim
mlp_rba            baseline + residual-based adaptivity (RBA) claim

Run:
    python benchmarks/rebuttal/ablations/ablate_features.py --epochs 5000
    python benchmarks/rebuttal/ablations/ablate_features.py --arms mlp fbpinn
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

from underPINN.benchmark_utils.evaluators import _burgers_reference  # noqa: E402
from underPINN.losses.loss import PINNLoss                          # noqa: E402
from underPINN.nn.fbpinn import FBPINN                              # noqa: E402
from underPINN.nn.mlp import MLP, FourierMLP, GatedMLP              # noqa: E402
from underPINN.pde.burgers import BurgersPDE                        # noqa: E402
from underPINN.utils.metrics import relative_l2_error               # noqa: E402

NU = 0.01
T_MAX = 1.5
N_R, N_IC, N_BC = 20000, 200, 300
W_IC, W_BC = 100.0, 10.0
LR = 1e-3
BASE_LAYERS = [2, 64, 64, 64, 64, 64, 1]


def make_data(seed: int):
    rng = np.random.default_rng(seed)
    x_r = jnp.array(rng.uniform(-1.0, 1.0, N_R).astype("f4"))
    t_r = jnp.array(rng.uniform(0.0, T_MAX, N_R).astype("f4"))
    x_ic = jnp.array(np.linspace(-1, 1, N_IC, dtype="f4"))
    u_ic = jnp.array(-np.sin(np.pi * np.asarray(x_ic)))
    t_bc_half = rng.uniform(0.0, T_MAX, N_BC).astype("f4")
    x_bc = jnp.array(np.tile([-1.0, 1.0], N_BC).astype("f4"))
    t_bc = jnp.array(np.tile(t_bc_half, 2))
    u_bc = jnp.zeros(2 * N_BC, dtype="f4")
    return x_r, t_r, x_ic, u_ic, x_bc, t_bc, u_bc


def build_fbpinn(n_sub: int = 4):
    """Overlapping 1-D-in-x decomposition across x in [-1, 1] (t left global)."""
    edges = np.linspace(-1.0, 1.0, n_sub + 1)
    width = edges[1] - edges[0]
    overlap = 0.3 * width
    shifts, xs_min, xs_max, smins, smaxs = [], [], [], [], []
    for i in range(n_sub):
        lo, hi = edges[i] - overlap, edges[i + 1] + overlap
        shifts.append([0.5 * (lo + hi), 0.5 * T_MAX])
        xs_min.append([lo, -1.0])            # t window spans the whole domain
        xs_max.append([hi, T_MAX + 1.0])
        smins.append([0.3 * overlap, 1.0])
        smaxs.append([0.3 * overlap, 1.0])
    sub_layers = [2, 32, 32, 32, 1]          # smaller nets, n_sub of them
    return FBPINN(
        layers=sub_layers,
        shifts=jnp.array(np.array(shifts, dtype="f4")),
        xs_min=jnp.array(np.array(xs_min, dtype="f4")),
        xs_max=jnp.array(np.array(xs_max, dtype="f4")),
        smins=jnp.array(np.array(smins, dtype="f4")),
        smaxs=jnp.array(np.array(smaxs, dtype="f4")),
    )


ARMS = {
    "mlp": dict(build=lambda: MLP(layers=BASE_LAYERS), rba=False,
                desc="plain tanh MLP (baseline)"),
    "gated_mlp": dict(build=lambda: GatedMLP(layers=BASE_LAYERS), rba=False,
                      desc="GatedMLP -- gated attention claim"),
    "fourier_mlp": dict(build=lambda: FourierMLP(layers=BASE_LAYERS,
                                                 n_fourier=16, sigma=1.0),
                        rba=False,
                        desc="FourierMLP -- trainable spectral embedding"),
    "fbpinn": dict(build=build_fbpinn, rba=False,
                   desc="FBPINN -- 4-subdomain decomposition"),
    "mlp_rba": dict(build=lambda: MLP(layers=BASE_LAYERS), rba=True,
                    desc="baseline + residual-based adaptivity (RBA)"),
}


def count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def evaluate_rel_l2(model, params) -> float:
    """Relative L^2 against the RK45 reference, averaged over 5 time slices."""
    x_ref, t_ref_grid, U_ref = _burgers_reference(NU)
    preds, refs = [], []
    for t_val in [0.5, 0.75, 1.0, 1.25, 1.5]:
        idx = int(np.argmin(np.abs(t_ref_grid - t_val)))
        pts = jnp.stack([jnp.array(x_ref.astype("f4")),
                         jnp.full(len(x_ref), t_val, "f4")], axis=1)
        preds.append(np.array(model.apply(params, pts)[:, 0]))
        refs.append(U_ref[:, idx].astype("f4"))
    return float(relative_l2_error(np.concatenate(preds), np.concatenate(refs)))


def run_arm(name: str, epochs: int, seed: int, data) -> dict:
    spec = ARMS[name]
    x_r, t_r, x_ic, u_ic, x_bc, t_bc, u_bc = data

    model = spec["build"]()
    pde = BurgersPDE(model, nu=NU)
    loss = PINNLoss(model, pde, ic_weight=W_IC, bc_weight=W_BC, rba=spec["rba"])
    params = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 2)))

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=1e-2)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))

    @jax.jit
    def step(p, s):
        # PINNLoss returns (total, (pde, ic, bc, reg)) -- hence has_aux=True.
        (val, _aux), g = jax.value_and_grad(
            lambda q: loss(q, x_r, t_r, x_ic, u_ic, x_bc, t_bc, u_bc),
            has_aux=True)(p)
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, val

    state = opt.init(params)
    _p, _s, warm = step(params, state)
    warm.block_until_ready()                    # compile once, untimed

    def train():
        p, s, val = params, state, None
        for _ in range(epochs):
            p, s, val = step(p, s)
        val.block_until_ready()
        return p, float(val)

    (final_params, final_loss), wall = timed(train)
    return {
        "description": spec["desc"],
        "rba": spec["rba"],
        "n_params": count_params(final_params),
        "epochs": epochs,
        "wall_s": wall,
        "ms_per_epoch": 1e3 * wall / epochs,
        "final_loss": final_loss,
        "rel_l2": evaluate_rel_l2(model, final_params),
    }


def main() -> int:
    ap = base_parser("Ablate underPINN's advertised features on 1-D Burgers")
    ap.add_argument("--arms", nargs="*", default=list(ARMS),
                    choices=list(ARMS), help="which arms to run")
    args = ap.parse_args()

    info = jax_device_info(require_gpu=not args.allow_cpu)
    warn_if_cpu(info)
    print(f"JAX backend: {info['platform']} ({info['device_name']})")
    print(f"Epochs per arm: {args.epochs}   seed: {args.seed}\n")

    data = make_data(args.seed)
    rows = {}
    for name in args.arms:
        print(f"--- {name}: {ARMS[name]['desc']}")
        try:
            r = run_arm(name, args.epochs, args.seed, data)
            rows[name] = r
            print(f"    params={r['n_params']:>7d}  "
                  f"{r['ms_per_epoch']:6.2f} ms/ep  "
                  f"loss={r['final_loss']:.4e}  rel_L2={r['rel_l2']:.4e}")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            rows[name] = {"error": f"{type(e).__name__}: {e}"}

    ok = {k: v for k, v in rows.items() if "error" not in v}
    if ok:
        base = ok.get("mlp", {}).get("rel_l2")
        print("\n" + "=" * 84)
        print(f"{'arm':14s} {'params':>8s} {'ms/ep':>8s} {'rel L2':>11s} "
              f"{'vs mlp':>9s}   description")
        print("-" * 84)
        for name in ARMS:
            if name not in ok:
                continue
            r = ok[name]
            rel = f"{base / r['rel_l2']:.2f}x" if base else "-"
            print(f"{name:14s} {r['n_params']:8d} {r['ms_per_epoch']:8.2f} "
                  f"{r['rel_l2']:11.4e} {rel:>9s}   {r['description']}")
        print("=" * 84)
        print("\n'vs mlp' > 1 means the feature reduced error relative to the "
              "plain-MLP\nbaseline; < 1 means it made accuracy worse on this "
              "problem. Read it\nalongside the parameter count.")

    save_result("ablation_features_burgers", {
        "problem": "burgers_1d", "epochs": args.epochs, "seed": args.seed,
        "device": info, "metric": "relative L2 vs RK45 reference",
        "arms": rows,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
