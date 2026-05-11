"""
3-D Pipe Flow — Hagen-Poiseuille (PINN)
========================================

Solves the steady 3-D incompressible Navier-Stokes equations inside a
cylindrical pipe of radius R = 0.5 and length L = 2.0.

Governing equations (non-dimensional, ρ = 1):
    ∇·u = 0
    (u·∇)u + ∇p = (1/Re) Δu

Exact solution — Hagen-Poiseuille:
    u(y,z) = U_max [1 − (y²+z²) / R²]
    v = w  = 0
    p(x)   = (dP/dx)(x − L),   dP/dx = −4ν U_max / R²

Boundary conditions:
    Inlet  (x=0) : u = parabolic profile,  v = w = 0
    Wall   (r=R) : u = v = w = 0          (no-slip)
    Outlet (x=L) : p = 0                  (pressure reference)

Network: MLP([3, 64, 64, 64, 64, 4])  →  (u, v, w, p)

NOTE: 3-D Hessian computation is expensive on CPU; expect ~10-20 min
      for 5 000 epochs.  On a GPU, this runs ~5-10×  faster.

Outputs
-------
pipe_flow_loss.png          — training loss curves
pipe_flow_cross_section.png — u(y,z) contour at x = L/2
pipe_flow_profiles.png      — radial + axial velocity profiles vs exact
pipe_flow_pressure.png      — centreline pressure vs exact (linear)
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.nn.mlp import MLP
from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE
from underPINN.geometry.pipe import Pipe
from underPINN.utils.io import save_predictions


# ── Problem parameters ────────────────────────────────────────────────────────
R     = 0.5      # pipe radius
L     = 2.0      # pipe length
RE    = 10.0     # Reynolds number (Poiseuille is exact for all Re)
U_MAX = 1.0      # centreline velocity

# ── Loss weights ──────────────────────────────────────────────────────────────
W_PDE    = 1.0
W_WALL   = 100.0
W_INLET  = 50.0
W_OUTLET = 20.0

# ── Training settings ─────────────────────────────────────────────────────────
EPOCHS    = 5000
LR        = 1e-3
BATCH_R   = 256     # collocation mini-batch  (keep small for 3-D Hessian)
BATCH_BC  = 128     # boundary mini-batch
LOG_EVERY = 500

LAYERS = [3, 64, 64, 64, 64, 4]


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_data(pipe: Pipe):
    xyz_r   = jnp.array(pipe.sample_interior(5000, seed=0))
    xyz_w   = jnp.array(pipe.sample_wall(1500,    seed=1))
    xyz_in  = jnp.array(pipe.sample_inlet(400,    seed=2))
    xyz_out = jnp.array(pipe.sample_outlet(400,   seed=3))
    return xyz_r, xyz_w, xyz_in, xyz_out


def inlet_velocity(xyz_in):
    """Parabolic (Poiseuille) profile at the inlet disk."""
    r2 = xyz_in[:, 1] ** 2 + xyz_in[:, 2] ** 2
    return U_MAX * (1.0 - r2 / R ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────

def build_loss_fn(model, pde):
    def loss_fn(params, xyz_r, xyz_w, xyz_in, xyz_out):
        # ── PDE residuals ────────────────────────────────────────────────────
        cont, mx, my, mz = pde.residual(params, xyz_r)
        pde_loss = (jnp.mean(cont ** 2) + jnp.mean(mx ** 2)
                    + jnp.mean(my ** 2) + jnp.mean(mz ** 2))

        # ── Wall no-slip: u = v = w = 0 ─────────────────────────────────────
        out_w    = model.apply(params, xyz_w)
        wall_loss = jnp.mean(out_w[:, 0] ** 2
                             + out_w[:, 1] ** 2
                             + out_w[:, 2] ** 2)

        # ── Inlet: u = parabolic, v = w = 0 ─────────────────────────────────
        out_in       = model.apply(params, xyz_in)
        u_in_exact   = inlet_velocity(xyz_in)
        inlet_loss   = (jnp.mean((out_in[:, 0] - u_in_exact) ** 2)
                        + jnp.mean(out_in[:, 1] ** 2)
                        + jnp.mean(out_in[:, 2] ** 2))

        # ── Outlet: p = 0 ────────────────────────────────────────────────────
        out_out      = model.apply(params, xyz_out)
        outlet_loss  = jnp.mean(out_out[:, 3] ** 2)

        total = (W_PDE * pde_loss + W_WALL * wall_loss
                 + W_INLET * inlet_loss + W_OUTLET * outlet_loss)
        return total, (pde_loss, wall_loss, inlet_loss, outlet_loss)

    return loss_fn


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(model, pde, xyz_r, xyz_w, xyz_in, xyz_out):
    key    = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, 3)))

    lr_sched = optax.cosine_decay_schedule(LR, decay_steps=EPOCHS, alpha=1e-2)
    opt      = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_sched),
        optax.scale(-1.0),
    )
    opt_state = opt.init(params)

    loss_fn = build_loss_fn(model, pde)

    @jax.jit
    def step(params, state, xyz_r, xyz_w, xyz_in, xyz_out):
        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, xyz_r, xyz_w, xyz_in, xyz_out)
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)
        return params, state, total, aux

    hist = {"total": [], "pde": [], "wall": [], "inlet": [], "outlet": []}
    key  = jax.random.PRNGKey(42)

    N_r, N_w = xyz_r.shape[0], xyz_w.shape[0]
    N_in, N_out = xyz_in.shape[0], xyz_out.shape[0]

    for ep in range(EPOCHS):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        idx_r   = jax.random.choice(k1, N_r,   (BATCH_R,),              replace=False)
        idx_w   = jax.random.choice(k2, N_w,   (BATCH_BC,),             replace=False)
        idx_in  = jax.random.choice(k3, N_in,  (min(BATCH_BC, N_in),),  replace=False)
        idx_out = jax.random.choice(k4, N_out, (min(BATCH_BC, N_out),), replace=False)

        params, opt_state, total, (pde_l, wall_l, in_l, out_l) = step(
            params, opt_state,
            xyz_r[idx_r], xyz_w[idx_w], xyz_in[idx_in], xyz_out[idx_out],
        )

        hist["total"].append(float(total))
        hist["pde"].append(float(pde_l))
        hist["wall"].append(float(wall_l))
        hist["inlet"].append(float(in_l))
        hist["outlet"].append(float(out_l))

        if ep % LOG_EVERY == 0 or ep == EPOCHS - 1:
            print(f"Epoch {ep:5d} | total {total:.3e} | pde {pde_l:.3e} "
                  f"| wall {wall_l:.3e} | inlet {in_l:.3e} | outlet {out_l:.3e}")

    return params, hist


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_rel_l2(model, params, pde, pipe, n: int = 3000):
    """Compute relative L² errors against the Hagen-Poiseuille exact solution."""
    xyz = jnp.array(pipe.sample_interior(n, seed=99))
    u_p, v_p, w_p, p_p = pde.exact_poiseuille(xyz, R=R, U_max=U_MAX, L=L)
    out = model.apply(params, xyz)

    def rel_l2(pred, exact):
        return float(jnp.linalg.norm(pred - exact) / (jnp.linalg.norm(exact) + 1e-10))

    return {
        "u": rel_l2(out[:, 0], u_p),
        "v": rel_l2(out[:, 1], v_p),
        "w": rel_l2(out[:, 2], w_p),
        "p": rel_l2(out[:, 3], p_p),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss(hist):
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = {"total": "k", "pde": "steelblue", "wall": "tomato",
               "inlet": "seagreen", "outlet": "orchid"}
    for k, c in colours.items():
        ax.semilogy(hist[k], lw=1.3, label=k, color=c)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("3-D Pipe Flow — Training Loss")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig("pipe_flow_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: pipe_flow_loss.png")


def plot_cross_section(model, params):
    """2-D contour of u(y,z) at x = L/2: PINN vs Poiseuille exact."""
    Ng = 64
    yg = np.linspace(-R, R, Ng, dtype=np.float32)
    zg = np.linspace(-R, R, Ng, dtype=np.float32)
    YY, ZZ = np.meshgrid(yg, zg, indexing="ij")
    inside  = YY ** 2 + ZZ ** 2 <= R ** 2

    x_mid = L / 2.0
    xyz_flat = np.column_stack([
        np.full(Ng * Ng, x_mid, dtype=np.float32),
        YY.ravel(),
        ZZ.ravel(),
    ])
    out  = np.array(model.apply(params, jnp.array(xyz_flat)))
    u_nn = out[:, 0].reshape(Ng, Ng)
    u_nn[~inside] = np.nan

    r2   = YY ** 2 + ZZ ** 2
    u_ex = U_MAX * (1.0 - r2 / R ** 2)
    u_ex[~inside] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    vmin, vmax = 0.0, U_MAX
    for ax, u, title in zip(axes, [u_ex, u_nn], ["Exact (Poiseuille)", "PINN"]):
        cf = ax.contourf(yg, zg, u.T, 40, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        plt.colorbar(cf, ax=ax, label="u")
        circ = plt.Circle((0, 0), R, fill=False, ec="k", lw=1.2)
        ax.add_patch(circ)
        ax.set_title(f"u(y, z) at x = {x_mid:.1f} — {title}")
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.set_aspect("equal")

    fig.suptitle("3-D Pipe Flow: axial velocity cross-section")
    fig.tight_layout()
    fig.savefig("pipe_flow_cross_section.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: pipe_flow_cross_section.png")


def plot_profiles(model, params):
    """Radial profile at x = L/2 and axial centreline profile."""
    Nr  = 80
    r_arr = np.linspace(0, R, Nr, dtype=np.float32)

    # Radial: along y-axis (z=0) at x=L/2
    xyz_rad = np.column_stack([
        np.full(Nr, L / 2, dtype=np.float32),
        r_arr,
        np.zeros(Nr, dtype=np.float32),
    ])
    out_r       = np.array(model.apply(params, jnp.array(xyz_rad)))
    u_r_exact   = U_MAX * (1.0 - r_arr ** 2 / R ** 2)

    # Axial centreline (y=z=0)
    Nx  = 80
    x_arr = np.linspace(0, L, Nx, dtype=np.float32)
    xyz_ax = np.column_stack([
        x_arr,
        np.zeros(Nx, dtype=np.float32),
        np.zeros(Nx, dtype=np.float32),
    ])
    out_x = np.array(model.apply(params, jnp.array(xyz_ax)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(r_arr, u_r_exact, "k-",  lw=2.0, label="Exact")
    ax.plot(r_arr, out_r[:, 0], "r--", lw=1.8, label="PINN")
    ax.set_xlabel("r  (along y-axis, z = 0)")
    ax.set_ylabel("u")
    ax.set_title(f"Radial velocity profile at x = {L/2:.1f}")
    ax.legend()
    ax.grid(ls="--", alpha=0.4)

    ax = axes[1]
    ax.axhline(U_MAX, color="k", lw=2.0, label=f"Exact (u = {U_MAX})")
    ax.plot(x_arr, out_x[:, 0], "r--", lw=1.8, label="PINN")
    ax.set_xlabel("x  (axial)")
    ax.set_ylabel("u  (centreline)")
    ax.set_title("Centreline axial velocity vs x  (r = 0)")
    ax.legend()
    ax.grid(ls="--", alpha=0.4)

    fig.suptitle("3-D Pipe Flow — Velocity Profiles")
    fig.tight_layout()
    fig.savefig("pipe_flow_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: pipe_flow_profiles.png")


def plot_pressure(model, params):
    """Centreline pressure p(x) vs exact linear profile."""
    Nx = 80
    x_arr = np.linspace(0, L, Nx, dtype=np.float32)
    xyz_ax = np.column_stack([
        x_arr,
        np.zeros(Nx, dtype=np.float32),
        np.zeros(Nx, dtype=np.float32),
    ])
    out_x   = np.array(model.apply(params, jnp.array(xyz_ax)))

    nu      = 1.0 / RE
    dpdx    = -4.0 * nu * U_MAX / R ** 2
    p_exact = dpdx * (x_arr - L)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_arr, p_exact, "k-",  lw=2.0, label="Exact (linear)")
    ax.plot(x_arr, out_x[:, 3], "r--", lw=1.8, label="PINN")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title(f"Centreline pressure — 3-D Pipe Flow  (Re = {RE})")
    ax.legend()
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("pipe_flow_pressure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: pipe_flow_pressure.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("JAX devices:", jax.devices())
    print(f"\nProblem: Hagen-Poiseuille  R={R}  L={L}  Re={RE}  U_max={U_MAX}")
    print(f"Network: MLP{LAYERS}   Epochs: {EPOCHS}")

    pipe  = Pipe(R=R, L=L)
    model = MLP(layers=LAYERS)
    pde   = SteadyNS3DPDE(model, Re=RE)

    print("\n── Sampling collocation data ──")
    xyz_r, xyz_w, xyz_in, xyz_out = make_data(pipe)
    print(f"Interior: {xyz_r.shape[0]}   Wall: {xyz_w.shape[0]}   "
          f"Inlet: {xyz_in.shape[0]}   Outlet: {xyz_out.shape[0]}")

    print("\n── Training ──")
    params, hist = train(model, pde, xyz_r, xyz_w, xyz_in, xyz_out)

    print("\n── Relative L² errors vs Hagen-Poiseuille exact ──")
    errs = eval_rel_l2(model, params, pde, pipe)
    for k, v in errs.items():
        print(f"  {k:>2s} : {v:.3e}")

    # ── Save predictions at interior collocation points ───────────────────────
    uvwp = np.array(model.apply(params, xyz_r))
    u_ex, v_ex, w_ex, p_ex = pde.exact_poiseuille(xyz_r, R=R, U_max=U_MAX, L=L)
    save_predictions(
        ".",
        coords  = {"x": np.array(xyz_r[:, 0]),
                   "y": np.array(xyz_r[:, 1]),
                   "z": np.array(xyz_r[:, 2])},
        outputs = {"u_pred": uvwp[:, 0], "v_pred": uvwp[:, 1],
                   "w_pred": uvwp[:, 2], "p_pred": uvwp[:, 3]},
        exact   = {"u_exact": np.array(u_ex), "v_exact": np.array(v_ex),
                   "w_exact": np.array(w_ex), "p_exact": np.array(p_ex)},
    )

    print("\n── Plotting ──")
    plot_loss(hist)
    plot_cross_section(model, params)
    plot_profiles(model, params)
    plot_pressure(model, params)

    print("\nDone — 3-D pipe flow complete.")


if __name__ == "__main__":
    main()
