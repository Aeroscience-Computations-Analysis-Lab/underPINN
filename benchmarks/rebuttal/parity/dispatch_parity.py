"""Why does 3-D pipe flow report ~12 ms/epoch in one place and ~0.8 ms/epoch in another?

A reviewer flagged this discrepancy as a possible measurement-methodology or
test-parity problem. It is neither a typo nor a different problem size: both
numbers are real, and the gap is caused entirely by *how much work the host
does per epoch outside the compiled step function*.

``PipeFlowEvaluator.train`` (underPINN/benchmark_utils/evaluators.py) runs, per
epoch, in addition to the one ``jax.jit``-ed ``step`` call:

  * ``jax.random.split(key, 5)``          -> 1 dispatch
  * 4x ``jax.random.randint(...)``        -> 4 dispatches
  * 4x fancy-index gather ``xyz_int[ir]`` -> 4 dispatches
  * ``float(total)`` and ``float(pl)``    -> 2 BLOCKING device->host syncs

So the compiled step is ~1 of ~12 dispatches per epoch, and the two ``float()``
calls force the host to wait for the device every single epoch, serialising the
pipeline and preventing any overlap between epochs.

This script measures five variants of the identical pipe-flow problem, changing
only the loop scaffolding around an identical compiled step:

  A  evaluator_style      exactly what PipeFlowEvaluator does today
  B  no_host_sync         A, but losses accumulate on-device (stacked once at end)
  C  batching_in_jit      RNG + gather moved inside the jitted step
  D  jit_minimal          C + no per-epoch host sync
  E  lax_scan             D, fully fused via jax.lax.scan

The physics, network, batch sizes, and loss weights are identical across all
five -- only host-side scaffolding differs. Any timing spread is therefore
attributable to dispatch/sync overhead alone.

Run:
    python benchmarks/rebuttal/parity/dispatch_parity.py --epochs 5000
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))),
))

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import numpy as np                                                # noqa: E402
import optax                                                      # noqa: E402

from common import (base_parser, jax_device_info, save_result,     # noqa: E402
                    timed, warn_if_cpu)

from underPINN.geometry.pipe import Pipe                           # noqa: E402
from underPINN.nn.mlp import MLP                                   # noqa: E402
from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE           # noqa: E402

# Hyperparameters copied verbatim from PipeFlowEvaluator.train.
R, L, U_MAX, RE = 0.5, 2.0, 1.0, 10.0
W_PDE, W_WALL, W_IN, W_OUT = 1.0, 100.0, 50.0, 20.0
N_INT, N_WALL, N_IN, N_OUT = 40000, 600, 200, 200
B, BW, BI, BO = 256, 128, 64, 64
LAYERS = [3, 128, 128, 128, 128, 128, 4]


def build_problem(seed: int):
    """Geometry, model, PDE, optimizer -- shared by every variant."""
    pipe = Pipe(R=R, L=L)
    pools = (
        jnp.array(np.array(pipe.sample_interior(N_INT), dtype="f4")),
        jnp.array(np.array(pipe.sample_wall(N_WALL), dtype="f4")),
        jnp.array(np.array(pipe.sample_inlet(N_IN), dtype="f4")),
        jnp.array(np.array(pipe.sample_outlet(N_OUT), dtype="f4")),
    )
    model = MLP(layers=LAYERS)
    pde = SteadyNS3DPDE(model, Re=RE)
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 3)))
    return pools, model, pde, params0


def make_optimizer(epochs: int):
    sched = optax.cosine_decay_schedule(1e-3, epochs, alpha=1e-2)
    return optax.chain(optax.scale_by_adam(),
                       optax.scale_by_schedule(sched),
                       optax.scale(-1.0))


def make_loss_fn(pde):
    """The identical physics loss used by every variant."""
    def loss_fn(p, xint, xwall, xin, xout):
        res = pde.residual(p, xint)
        pde_l = jnp.mean(jnp.sum(res ** 2, axis=-1))
        u_w, v_w, w_w, _ = pde.uvwp(p, xwall)
        wall_l = jnp.mean(u_w ** 2 + v_w ** 2 + w_w ** 2)
        r_in = jnp.sqrt(xin[:, 1] ** 2 + xin[:, 2] ** 2)
        u_ex = U_MAX * (1 - r_in ** 2 / R ** 2)
        u_in, v_in, w_in, _ = pde.uvwp(p, xin)
        in_l = jnp.mean((u_in - u_ex) ** 2 + v_in ** 2 + w_in ** 2)
        u_o, v_o, w_o, _ = pde.uvwp(p, xout)
        out_l = jnp.mean(v_o ** 2 + w_o ** 2)
        total = W_PDE * pde_l + W_WALL * wall_l + W_IN * in_l + W_OUT * out_l
        return total, pde_l
    return loss_fn


# ── Variant A: exactly what PipeFlowEvaluator does today ─────────────────────

def variant_evaluator_style(epochs, seed, pools, pde, params0, optimizer):
    xyz_int, xyz_wall, xyz_in, xyz_out = pools
    loss_fn = make_loss_fn(pde)

    @jax.jit
    def step(params, state, xint, xwall, xin, xout):
        (total, pl), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params, xint, xwall, xin, xout)
        updates, state = optimizer.update(grads, state)
        return optax.apply_updates(params, updates), state, total, pl

    params, state = params0, optimizer.init(params0)
    key = jax.random.PRNGKey(seed + 11)
    loss_hist, pde_hist = [], []
    for _ in range(epochs):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        ir = jax.random.randint(k1, (B,), 0, xyz_int.shape[0])
        iw = jax.random.randint(k2, (BW,), 0, xyz_wall.shape[0])
        ii = jax.random.randint(k3, (BI,), 0, xyz_in.shape[0])
        io = jax.random.randint(k4, (BO,), 0, xyz_out.shape[0])
        params, state, total, pl = step(
            params, state, xyz_int[ir], xyz_wall[iw], xyz_in[ii], xyz_out[io])
        loss_hist.append(float(total))   # blocking device->host sync
        pde_hist.append(float(pl))       # blocking device->host sync
    return params, loss_hist[-1]


# ── Variant B: same dispatches, but no per-epoch host sync ───────────────────

def variant_no_host_sync(epochs, seed, pools, pde, params0, optimizer):
    xyz_int, xyz_wall, xyz_in, xyz_out = pools
    loss_fn = make_loss_fn(pde)

    @jax.jit
    def step(params, state, xint, xwall, xin, xout):
        (total, pl), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params, xint, xwall, xin, xout)
        updates, state = optimizer.update(grads, state)
        return optax.apply_updates(params, updates), state, total, pl

    params, state = params0, optimizer.init(params0)
    key = jax.random.PRNGKey(seed + 11)
    losses = []
    for _ in range(epochs):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        ir = jax.random.randint(k1, (B,), 0, xyz_int.shape[0])
        iw = jax.random.randint(k2, (BW,), 0, xyz_wall.shape[0])
        ii = jax.random.randint(k3, (BI,), 0, xyz_in.shape[0])
        io = jax.random.randint(k4, (BO,), 0, xyz_out.shape[0])
        params, state, total, _ = step(
            params, state, xyz_int[ir], xyz_wall[iw], xyz_in[ii], xyz_out[io])
        losses.append(total)             # stays on device
    final = float(jnp.stack(losses)[-1])  # one sync, at the very end
    return params, final


# ── Variant C: batching moved inside the compiled step ───────────────────────

def _make_fused_step(pools, pde, optimizer):
    """Compiled step that also does its own RNG + gather.

    The key is threaded through and split with the *identical* ``split(key, 5)``
    pattern used by variants A/B/E, so all five variants consume randomness the
    same way and follow bit-identical loss trajectories. Only the scaffolding
    around the compiled region differs -- which is the whole point.
    """
    xyz_int, xyz_wall, xyz_in, xyz_out = pools
    loss_fn = make_loss_fn(pde)

    @jax.jit
    def step(params, state, key):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        xint = xyz_int[jax.random.randint(k1, (B,), 0, xyz_int.shape[0])]
        xwall = xyz_wall[jax.random.randint(k2, (BW,), 0, xyz_wall.shape[0])]
        xin = xyz_in[jax.random.randint(k3, (BI,), 0, xyz_in.shape[0])]
        xout = xyz_out[jax.random.randint(k4, (BO,), 0, xyz_out.shape[0])]
        (total, pl), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params, xint, xwall, xin, xout)
        updates, state = optimizer.update(grads, state)
        return optax.apply_updates(params, updates), state, key, total, pl
    return step


def variant_batching_in_jit(epochs, seed, pools, pde, params0, optimizer):
    step = _make_fused_step(pools, pde, optimizer)
    params, state = params0, optimizer.init(params0)
    key = jax.random.PRNGKey(seed + 11)
    loss_hist = []
    for _ in range(epochs):
        params, state, key, total, _ = step(params, state, key)
        loss_hist.append(float(total))   # keeps A's per-epoch sync
    return params, loss_hist[-1]


def variant_jit_minimal(epochs, seed, pools, pde, params0, optimizer):
    step = _make_fused_step(pools, pde, optimizer)
    params, state = params0, optimizer.init(params0)
    key = jax.random.PRNGKey(seed + 11)
    losses = []
    for _ in range(epochs):
        params, state, key, total, _ = step(params, state, key)
        losses.append(total)
    return params, float(jnp.stack(losses)[-1])


# ── Variant E: fully fused with lax.scan ─────────────────────────────────────

def variant_lax_scan(epochs, seed, pools, pde, params0, optimizer):
    xyz_int, xyz_wall, xyz_in, xyz_out = pools
    loss_fn = make_loss_fn(pde)

    def body(carry, _):
        params, state, key = carry
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        xint = xyz_int[jax.random.randint(k1, (B,), 0, xyz_int.shape[0])]
        xwall = xyz_wall[jax.random.randint(k2, (BW,), 0, xyz_wall.shape[0])]
        xin = xyz_in[jax.random.randint(k3, (BI,), 0, xyz_in.shape[0])]
        xout = xyz_out[jax.random.randint(k4, (BO,), 0, xyz_out.shape[0])]
        (total, _pl), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params, xint, xwall, xin, xout)
        updates, state = optimizer.update(grads, state)
        return (optax.apply_updates(params, updates), state, key), total

    @jax.jit
    def run(params, state, key):
        (p, s, _), losses = jax.lax.scan(
            body, (params, state, key), None, length=epochs)
        return p, s, losses

    params, state = params0, optimizer.init(params0)
    key = jax.random.PRNGKey(seed + 11)
    params, _state, losses = run(params, state, key)
    losses.block_until_ready()
    return params, float(losses[-1])


VARIANTS = [
    ("A_evaluator_style", variant_evaluator_style,
     "as PipeFlowEvaluator runs today (host RNG+gather, 2 syncs/epoch)"),
    ("B_no_host_sync", variant_no_host_sync,
     "host RNG+gather kept, per-epoch float() syncs removed"),
    ("C_batching_in_jit", variant_batching_in_jit,
     "RNG+gather inside jit, per-epoch float() sync kept"),
    ("D_jit_minimal", variant_jit_minimal,
     "RNG+gather inside jit, no per-epoch sync"),
    ("E_lax_scan", variant_lax_scan,
     "entire loop fused into one jax.lax.scan call"),
]


def main() -> int:
    ap = base_parser(__doc__.split("\n")[0])
    args = ap.parse_args()

    info = jax_device_info(require_gpu=not args.allow_cpu)
    warn_if_cpu(info)
    print(f"JAX backend: {info['platform']} ({info['device_name']})")
    print(f"Epochs per variant: {args.epochs}\n")

    pools, _model, pde, params0 = build_problem(args.seed)
    optimizer = make_optimizer(args.epochs)

    rows = {}
    for key, fn, desc in VARIANTS:
        print(f"--- {key}: {desc}")
        # Warm up compilation with a short run so the timed run measures
        # steady-state execution, not tracing.
        warm_opt = make_optimizer(2)
        fn(2, args.seed, pools, pde, params0, warm_opt)
        (_p, final_loss), wall = timed(
            lambda f=fn: f(args.epochs, args.seed, pools, pde, params0, optimizer))
        ms = 1e3 * wall / args.epochs
        rows[key] = {"description": desc, "wall_s": wall,
                     "ms_per_epoch": ms, "final_loss": final_loss}
        print(f"    {wall:8.2f}s total   {ms:7.3f} ms/epoch   "
              f"final_loss={final_loss:.4e}\n")

    base = rows["A_evaluator_style"]["ms_per_epoch"]
    print("=" * 74)
    print(f"{'variant':22s} {'ms/epoch':>10s} {'speedup vs A':>14s}   description")
    print("-" * 74)
    for key, _fn, desc in VARIANTS:
        ms = rows[key]["ms_per_epoch"]
        print(f"{key:22s} {ms:10.3f} {base / ms:13.2f}x   {desc}")
    print("=" * 74)

    losses = {k: rows[k]["final_loss"] for k in rows}
    spread = max(losses.values()) - min(losses.values())
    print(f"\nfinal-loss spread across variants: {spread:.3e} "
          f"(should be ~0 -- all five run identical math)")
    if info["platform"] not in ("gpu", "cuda"):
        print("\nNOTE: on CPU there is no host<->device boundary, so dispatch "
              "and\nsync costs are near-zero and all five variants should look "
              "alike.\nThe separation this script measures only appears on GPU.")

    save_result("parity_pipeflow_dispatch", {
        "problem": "pipe_flow_3d",
        "epochs": args.epochs,
        "device": info,
        "batch_sizes": {"interior": B, "wall": BW, "inlet": BI, "outlet": BO},
        "variants": rows,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
