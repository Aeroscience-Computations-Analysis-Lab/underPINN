"""3-D steady Hagen-Poiseuille pipe flow, solved with jinns
(https://gitlab.com/mia_jinns/jinns) -- a sixth jinns comparison, matching
``../physicsnemo/pipe_flow_matched_jinns.py``'s underPINN side (itself a
reduced-scale, plain-MLP variant of
``compare_underpinn_multi.py::run_pipe_flow``'s full PhysicsNeMo
comparison; see that script's module docstring for the full rationale on
why this is reduced/architecture-adjusted rather than the exact 100,000-
point/GatedMLP/70,000-epoch config):

  * same physics   : (u.grad)u + grad(p) - nu*laplacian(u) = 0, div(u)=0,
                     nu=1/Re, Re=40, cylindrical pipe R=0.5, L=7,
                     x in [-3.5, 3.5], no-slip wall, parabolic inlet
                     (U_max=2), p=0 outlet -- exact Hagen-Poiseuille
                     solution scored the same way underPINN itself does
  * same geometry  : reuses underPINN's own ``underPINN.geometry.pipe
                     .Pipe`` sampler directly (same class, same seeds) --
                     not a separately-coded geometry that merely matches
                     point counts, but the literal same interior/wall/
                     inlet/outlet arrays underPINN trains on
  * same network   : plain MLP, 4 hidden layers x 192 units, tanh --
                     GatedMLP has no jinns equivalent, disclosed rather
                     than silently substituted (matches how FourierMLP was
                     handled for Burgers/Wave/Helmholtz elsewhere in this
                     comparison)
  * same batching  : minibatched every step (batch_r=2048,
                     batch_bc=512 for wall/inlet/outlet each)
  * same optimizer : plain Adam with a cosine decay schedule
  * same loss weights: W_PDE=1, W_WALL=100, W_INLET=50, W_OUTLET=20

**Two real jinns API gaps, found rather than assumed away, that make this
comparison structurally different from the other five in this
directory:**

1. jinns' convective-term utility, ``_u_dot_nabla_times_u_rev``, is
   hard-coded to 2-D inputs only (``assert x.shape[0] == 2`` in its own
   source) -- unusable for this 3-D problem. The convective term
   ``(u.grad)u`` is therefore hand-written here with ``jax.jacfwd``
   instead, while the diffusion term still uses jinns' own
   ``vectorial_laplacian_rev`` (which *does* support arbitrary spatial
   dimension, confirmed from source) and the continuity term uses jinns'
   own ``divergence_rev`` (likewise dimension-general).
2. This problem's geometry (a cylinder, with wall/inlet/outlet boundary
   *conditions* that differ by region) does not fit jinns'
   ``CubicMeshPDEStatio``/``Dirichlet`` machinery, which assumes an
   axis-aligned box with one boundary condition per facet. Like
   ``ode_harmonic_jinns.py``, this script therefore bypasses
   ``jinns.solve`` and writes a manual composite loss + training loop
   (``jax.value_and_grad`` + ``optax``) around jinns' PINN class and its
   dimension-general vector-field operators -- genuine jinns machinery for
   the parts that fit, hand-written for the two parts (3-D convection,
   non-box geometry) that do not, and disclosed as such rather than
   presented as an unmodified ``jinns.solve`` run.

Run (from this directory, using the jinns venv):
    ./.venv/bin/python pipe_flow_jinns.py --epochs 30000
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import equinox as eqx                                          # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jinns                                                    # noqa: E402
import numpy as np                                              # noqa: E402
import optax                                                    # noqa: E402
from jax import random                                          # noqa: E402
from jinns.loss._operators import divergence_rev, vectorial_laplacian_rev  # noqa: E402

from common import base_parser, jax_device_info, save_result, warn_if_cpu  # noqa: E402

from underPINN.geometry.pipe import Pipe                       # noqa: E402
from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE        # noqa: E402

RE, R, L, U_MAX, X_LO = 40.0, 0.5, 7.0, 2.0, -3.5
N_INTERIOR, N_WALL, N_INLET, N_OUTLET = 20000, 3000, 500, 500
BATCH_R, BATCH_BC = 2048, 512
W_PDE, W_WALL, W_INLET, W_OUTLET = 1.0, 100.0, 50.0, 20.0
LAYERS_HIDDEN, LAYER_SIZE = 4, 192
LR = 1e-3


def inlet_velocity(xyz):
    r2 = xyz[:, 1] ** 2 + xyz[:, 2] ** 2
    return U_MAX * (1.0 - r2 / R ** 2)


def _convective_3d(x, u, params):
    """(u.grad)u for a 3-D velocity field -- jinns' own
    _u_dot_nabla_times_u_rev is 2-D only (see module docstring), so this
    is hand-written via jax.jacfwd, matching underPINN's own
    SteadyNS3DPDE.residual formula exactly."""
    def vel(x):
        return u(x, params)[:3]
    J = jax.jacfwd(vel)(x)      # (3,3): J[i,j] = d(vel_i)/dx_j
    v = vel(x)                  # (3,)
    return J @ v


def pipe_ns_residual(x, u, params, nu):
    conv = _convective_3d(x, u, params)
    grad_p = jax.grad(lambda x: u(x, params)[3])(x)
    lap = vectorial_laplacian_rev(x, u, params, dim_out=3, eq_type="PDEStatio")
    mom = conv + grad_p - nu * lap
    cont = divergence_rev(x, u, params, eq_type="PDEStatio")
    return jnp.concatenate([cont[None], mom])


def run_jinns(epochs: int, seed: int) -> dict:
    key = random.PRNGKey(seed)

    eqx_list = ((eqx.nn.Linear, 3, LAYER_SIZE), (jax.nn.tanh,))
    for _ in range(LAYERS_HIDDEN - 1):
        eqx_list += ((eqx.nn.Linear, LAYER_SIZE, LAYER_SIZE), (jax.nn.tanh,))
    eqx_list += ((eqx.nn.Linear, LAYER_SIZE, 4),)

    key, subkey = random.split(key)
    u_pinn, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="PDEStatio")
    init_params = jinns.parameters.Params(nn_params=init_nn_params, eq_params={})

    # reuse underPINN's own geometry sampler directly -- same class, same
    # seeds, so both sides train on statistically identical point sets.
    pipe = Pipe(R=R, L=L, x_lo=X_LO)
    xyz_r0 = jnp.array(pipe.sample_interior(N_INTERIOR, seed=seed))
    xyz_w = jnp.array(pipe.sample_wall(N_WALL, seed=seed + 1))
    xyz_in = jnp.array(pipe.sample_inlet(N_INLET, seed=seed + 2))
    xyz_out = jnp.array(pipe.sample_outlet(N_OUTLET, seed=seed + 3))
    N_w, N_in, N_out = xyz_w.shape[0], xyz_in.shape[0], xyz_out.shape[0]

    nu = 1.0 / RE

    def loss_fn(params, xr, xw, xin, xout):
        res = jax.vmap(lambda x: pipe_ns_residual(x, u_pinn, params, nu))(xr)
        pde_l = jnp.mean(jnp.sum(res ** 2, axis=-1))

        out_w = jax.vmap(lambda x: u_pinn(x, params))(xw)
        wall_l = jnp.mean(out_w[:, 0] ** 2 + out_w[:, 1] ** 2 + out_w[:, 2] ** 2)

        out_in = jax.vmap(lambda x: u_pinn(x, params))(xin)
        u_ex = inlet_velocity(xin)
        in_l = (jnp.mean((out_in[:, 0] - u_ex) ** 2)
               + jnp.mean(out_in[:, 1] ** 2) + jnp.mean(out_in[:, 2] ** 2))

        out_out = jax.vmap(lambda x: u_pinn(x, params))(xout)
        outlet_l = jnp.mean(out_out[:, 3] ** 2)

        return (W_PDE * pde_l + W_WALL * wall_l + W_INLET * in_l
               + W_OUTLET * outlet_l)

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=0.01)
    tx = optax.adam(sched)
    opt_state = tx.init(init_params)

    def safe_choice(key, n, batch):
        replace = batch > n
        return random.choice(key, n, (batch,), replace=replace)

    @jax.jit
    def step(params, opt_state, key):
        key, k1, k2, k3, k4 = random.split(key, 5)
        ir = safe_choice(k1, N_INTERIOR, BATCH_R)
        iw = safe_choice(k2, N_w, min(BATCH_BC, N_w))
        iin = safe_choice(k3, N_in, min(BATCH_BC, N_in))
        iout = safe_choice(k4, N_out, min(BATCH_BC, N_out))
        loss, grads = jax.value_and_grad(loss_fn)(
            params, xyz_r0[ir], xyz_w[iw], xyz_in[iin], xyz_out[iout])
        updates, opt_state = tx.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, key, loss

    key0 = random.PRNGKey(seed + 99)
    p, s, k = init_params, opt_state, key0
    for _ in range(2):
        p, s, k, warm_loss = step(p, s, k)
    jax.block_until_ready(warm_loss)

    params, opt_state, key = init_params, opt_state, key0
    t0 = time.perf_counter()
    loss = None
    for _ in range(epochs):
        params, opt_state, key, loss = step(params, opt_state, key)
    jax.block_until_ready(loss)
    wall = time.perf_counter() - t0
    final_loss = float(loss)

    # score exactly as underPINN's own run_pipe_flow / pipe_flow_matched_jinns.py
    # does (3000 random points, rng seed=99, same axis labeling)
    rng = np.random.default_rng(99)
    n_val = 3000
    rr = R * np.sqrt(rng.uniform(0.0, 1.0, n_val))
    th = rng.uniform(0.0, 2 * np.pi, n_val)
    y_v = (rr * np.cos(th)).astype(np.float32)
    z_v = (rr * np.sin(th)).astype(np.float32)
    x_v = rng.uniform(X_LO, X_LO + L, n_val).astype(np.float32)
    xyz_val = jnp.array(np.stack([x_v, y_v, z_v], axis=1))

    pde_exact = SteadyNS3DPDE(model=None, Re=RE)
    u_p, v_p, w_p, p_p = pde_exact.exact_poiseuille(xyz_val, R=R, U_max=U_MAX, L=L, x_lo=X_LO)
    out_val = jax.vmap(lambda x: u_pinn(x, params))(xyz_val)

    def rel_l2(pred, exact):
        return float(jnp.linalg.norm(pred - exact) / (jnp.linalg.norm(exact) + 1e-10))

    rel_l2_u = rel_l2(out_val[:, 0], u_p)
    rel_l2_p = rel_l2(out_val[:, 3], p_p)

    return {"framework": "jinns", "jinns_version": "1.10.0", "epochs": epochs,
            "wall_s": wall, "ms_per_epoch": 1e3 * wall / epochs,
            "final_loss": final_loss, "rel_l2_axial_velocity": rel_l2_u,
            "rel_l2_pressure": rel_l2_p, "n_interior": N_INTERIOR,
            "batch_r": BATCH_R, "batch_bc": BATCH_BC,
            "layers_hidden": LAYERS_HIDDEN, "layer_size": LAYER_SIZE}


def main() -> int:
    ap = base_parser("jinns JAX 3-D pipe flow, scored on the same exact "
                     "Poiseuille reference as underPINN")
    ap.set_defaults(epochs=30000)
    args = ap.parse_args()

    info = jax_device_info(require_gpu=not args.allow_cpu)
    warn_if_cpu(info)
    print(f"JAX backend: {info['platform']} ({info['device_name']})")
    print(f"Epochs: {args.epochs}   seed: {args.seed}\n")

    r = run_jinns(args.epochs, args.seed)
    print(f"jinns: {r['ms_per_epoch']:.3f} ms/ep  final_loss={r['final_loss']:.4e}  "
         f"rel_L2(u)={r['rel_l2_axial_velocity']:.4e}  rel_L2(p)={r['rel_l2_pressure']:.4e}")

    save_result("jinns_compare_pipe_flow", {
        "problem": "pipe_flow_3d", "device": info, **r,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
