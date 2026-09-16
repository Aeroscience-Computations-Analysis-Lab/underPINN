"""3-D steady Hagen-Poiseuille pipe flow (underPINN side), matched to
``../jinns/pipe_flow_jinns.py`` for a genuine jinns comparison -- distinct
from ``compare_underpinn_multi.py::run_pipe_flow``'s full-scale PhysicsNeMo
comparison (100,000 interior points, GatedMLP, 70,000 epochs), reduced and
architecture-adjusted specifically to make a fair jinns comparison
tractable:

  * network: **plain MLP**, not GatedMLP -- GatedMLP has no jinns
    equivalent (same disclosed gap as FourierMLP for Burgers/Wave/
    Helmholtz elsewhere in this comparison), so this uses the same
    architecture on both sides rather than silently comparing an
    unmatched one
  * scale: N_interior=20,000 (down from 100,000), epochs=10,000 (down from
    70,000) -- the full 70,000-epoch/100,000-point PhysicsNeMo-matched
    config would make a from-scratch jinns port prohibitively slow to
    verify; this reduced scale is disclosed as such, not presented as the
    same comparison at a smaller sample size
  * same physics, same geometry: reuses underPINN's own
    ``underPINN.geometry.pipe.Pipe`` sampler directly (same class, same
    seeds) so both sides train on statistically identical collocation,
    wall, inlet, and outlet point sets, not just matched counts

Run:
    python pipe_flow_matched_jinns.py --epochs 10000
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
import numpy as np                                            # noqa: E402
import optax                                                  # noqa: E402

from common import base_parser, jax_device_info, save_result, timed, warn_if_cpu  # noqa: E402

from underPINN.nn.mlp import MLP                              # noqa: E402
from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE       # noqa: E402
from underPINN.geometry.pipe import Pipe                      # noqa: E402
from underPINN.utils.sampling import safe_choice              # noqa: E402

RE, R, L, U_MAX, X_LO = 40.0, 0.5, 7.0, 2.0, -3.5
N_INTERIOR, N_WALL, N_INLET, N_OUTLET = 20000, 3000, 500, 500
BATCH_R, BATCH_BC = 2048, 512
W_PDE, W_WALL, W_INLET, W_OUTLET = 1.0, 100.0, 50.0, 20.0
LAYERS = [3, 192, 192, 192, 192, 4]
LR = 1e-3


def inlet_velocity(xyz):
    r2 = xyz[:, 1] ** 2 + xyz[:, 2] ** 2
    return U_MAX * (1.0 - r2 / R ** 2)


def run_underpinn(epochs: int, seed: int) -> dict:
    pipe = Pipe(R=R, L=L, x_lo=X_LO)
    xyz_r0 = jnp.array(pipe.sample_interior(N_INTERIOR, seed=seed))
    xyz_w = jnp.array(pipe.sample_wall(N_WALL, seed=seed + 1))
    xyz_in = jnp.array(pipe.sample_inlet(N_INLET, seed=seed + 2))
    xyz_out = jnp.array(pipe.sample_outlet(N_OUTLET, seed=seed + 3))

    model = MLP(layers=LAYERS)
    pde = SteadyNS3DPDE(model, Re=RE)
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 3)))

    def loss_fn(p, xr, xw, xin, xout):
        res = pde.residual(p, xr)
        pde_l = jnp.mean(jnp.sum(res ** 2, axis=-1))
        out_w = model.apply(p, xw)
        wall_l = jnp.mean(out_w[:, 0] ** 2 + out_w[:, 1] ** 2 + out_w[:, 2] ** 2)
        out_in = model.apply(p, xin)
        u_ex = inlet_velocity(xin)
        in_l = (jnp.mean((out_in[:, 0] - u_ex) ** 2)
               + jnp.mean(out_in[:, 1] ** 2) + jnp.mean(out_in[:, 2] ** 2))
        out_out = model.apply(p, xout)
        outlet_l = jnp.mean(out_out[:, 3] ** 2)
        return (W_PDE * pde_l + W_WALL * wall_l + W_INLET * in_l
               + W_OUTLET * outlet_l)

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=0.01)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))

    N_w, N_in, N_out = xyz_w.shape[0], xyz_in.shape[0], xyz_out.shape[0]

    @jax.jit
    def step(p, s, key):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        ir = safe_choice(k1, N_INTERIOR, BATCH_R)
        iw = safe_choice(k2, N_w, BATCH_BC)
        iin = safe_choice(k3, N_in, min(BATCH_BC, N_in))
        iout = safe_choice(k4, N_out, min(BATCH_BC, N_out))
        loss, g = jax.value_and_grad(loss_fn)(
            p, xyz_r0[ir], xyz_w[iw], xyz_in[iin], xyz_out[iout])
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, key, loss

    state0 = opt.init(params0)
    key0 = jax.random.PRNGKey(seed + 99)
    _p, _s, _k, warm = step(params0, state0, key0)
    jax.block_until_ready(warm)

    def train():
        p, s, key, loss = params0, state0, key0, None
        for _ in range(epochs):
            p, s, key, loss = step(p, s, key)
        jax.block_until_ready(loss)
        return p, float(loss)

    (final_params, final_loss), wall = timed(train)

    rng = np.random.default_rng(99)
    n_val = 3000
    rr = R * np.sqrt(rng.uniform(0.0, 1.0, n_val))
    th = rng.uniform(0.0, 2 * np.pi, n_val)
    y_v = (rr * np.cos(th)).astype(np.float32)
    z_v = (rr * np.sin(th)).astype(np.float32)
    x_v = rng.uniform(X_LO, X_LO + L, n_val).astype(np.float32)
    xyz_val = jnp.array(np.stack([x_v, y_v, z_v], axis=1))
    u_p, v_p, w_p, p_p = pde.exact_poiseuille(xyz_val, R=R, U_max=U_MAX, L=L, x_lo=X_LO)
    out_val = model.apply(final_params, xyz_val)

    def rel_l2(pred, exact):
        return float(jnp.linalg.norm(pred - exact) / (jnp.linalg.norm(exact) + 1e-10))

    rel_l2_u = rel_l2(out_val[:, 0], u_p)
    rel_l2_p = rel_l2(out_val[:, 3], p_p)
    return {"problem": "pipe_flow_3d_matched_jinns", "epochs": epochs, "wall_s": wall,
           "ms_per_epoch": 1e3 * wall / epochs, "final_loss": final_loss,
           "rel_l2_axial_velocity": rel_l2_u, "rel_l2_pressure": rel_l2_p}


def main() -> int:
    ap = base_parser("underPINN plain-MLP pipe flow, reduced scale, "
                     "matched for a jinns comparison")
    ap.set_defaults(epochs=10000)
    args = ap.parse_args()

    info = jax_device_info(require_gpu=not args.allow_cpu)
    warn_if_cpu(info)
    print(f"JAX backend: {info['platform']} ({info['device_name']})")
    print(f"Epochs: {args.epochs}   seed: {args.seed}\n")

    r = run_underpinn(args.epochs, args.seed)
    print(f"underPINN: {r['ms_per_epoch']:.3f} ms/ep  final_loss={r['final_loss']:.4e}  "
         f"rel_L2(u)={r['rel_l2_axial_velocity']:.4e}  rel_L2(p)={r['rel_l2_pressure']:.4e}")

    save_result("pipe_flow_matched_jinns_underpinn", {"device": info, **r})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
