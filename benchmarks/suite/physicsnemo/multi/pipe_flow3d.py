"""3-D steady Hagen-Poiseuille pipe flow, PhysicsNeMo Sym vs underPINN --
matches examples/pipe_flow/pipe_flow.yaml: steady incompressible NS in a
cylinder, Re=40, R=0.5, L=7.0, U_max=2.0, exact parabolic profile.

Uses PhysicsNeMo's built-in NavierStokes(nu=1/Re, rho=1) -- identical in
form to underPINN.pde.navier_stokes_3d.SteadyNS3DPDE's nondimensional
``(u.grad)u = -grad p + (1/Re) lap u`` (rho=1) once nu=1/Re.

PhysicsNeMo's Cylinder primitive is z-axis-aligned (not x-aligned like
underPINN's Pipe geometry) -- rather than rotate the geometry, this keeps
PhysicsNeMo's native axes and treats z as the flow/axial direction (w as
axial velocity, x/y as the cross-section) throughout; the physics is
axis-label-agnostic, so this is a relabeling, not an approximation.

Not matched (documented): underPINN uses a GatedMLP [3,192,192,192,192,4];
this uses a plain FullyConnectedArch of the same depth/width (PhysicsNeMo
has no direct drop-in for underPINN's dual-encoder gating).

Run:
    env -u SLURM_PROCID ../.venv/bin/python pipe_flow3d.py training.max_steps=70000 jit=false
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from sympy import Symbol

import physicsnemo.sym
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.geometry.primitives_3d import Cylinder
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Activation
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.solver import Solver

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", ".."))

RE, R, L, U_MAX = 40.0, 0.5, 7.0, 2.0
Z_LO, Z_HI = -L / 2.0, L / 2.0
NU = 1.0 / RE
N_INTERIOR, N_WALL, N_INLET, N_OUTLET = 100000, 15000, 2000, 2000
W_PDE, W_WALL, W_INLET, W_OUTLET = 1.0, 100.0, 50.0, 20.0
LAYERS_HIDDEN, LAYER_SIZE = 4, 192
EPS = 1e-4


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg) -> None:
    t0_setup = time.perf_counter()

    ns = NavierStokes(nu=NU, rho=1, dim=3, time=False)
    net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        layer_size=LAYER_SIZE, nr_layers=LAYERS_HIDDEN,
        activation_fn=Activation.TANH,
        weight_norm=False,  # match underPINN's plain MLP -- PhysicsNeMo's
        # FullyConnectedArch defaults to weight_norm=True (Salimans & Kingma
        # 2016) on every hidden layer, an undocumented architectural
        # difference from a genuinely plain MLP, not something underPINN's
        # MLP/FourierMLP/GatedMLP do.
    )
    nodes = ns.make_nodes() + [net.make_node(name="pipe_net")]

    geo = Cylinder(center=(0, 0, 0), radius=R, height=L)
    domain = Domain()

    x_sym, y_sym, z_sym = Symbol("x"), Symbol("y"), Symbol("z")
    r2 = x_sym ** 2 + y_sym ** 2

    # batch_size below matches underPINN's own per-step minibatch sizes
    # (batch_r=2048, batch_bc=1024 in pipe_flow.yaml) with fixed_dataset=False
    # so PhysicsNeMo draws a fresh sample every step, the same per-step
    # compute cost as underPINN's random-index minibatching from its fixed
    # 100k/15k/2k/2k pools (not a bit-identical resampling scheme -- fresh
    # continuous points vs. fresh indices into a fixed pool -- but the same
    # per-step point count, which is what the throughput comparison turns
    # on). An earlier version used the full pool size with fixed_dataset=True
    # (true full-batch), which was both an unintended mismatch against
    # underPINN's actual minibatched examples and projected to ~27 hours at
    # the full 70,000-epoch budget.
    BATCH_WALL, BATCH_BC, BATCH_R = 1024, 1024, 2048

    wall = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"u": 0, "v": 0, "w": 0}, batch_size=BATCH_WALL,
        criteria=(z_sym > Z_LO + EPS) & (z_sym < Z_HI - EPS),   # lateral surface only
        lambda_weighting={"u": W_WALL, "v": W_WALL, "w": W_WALL},
        fixed_dataset=True,  # thin criteria filter is fragile under continuous resampling
    )
    domain.add_constraint(wall, "wall")

    inlet = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"u": 0, "v": 0, "w": U_MAX * (1.0 - r2 / R ** 2)},
        batch_size=BATCH_BC,
        criteria=(z_sym < Z_LO + EPS),
        lambda_weighting={"u": W_INLET, "v": W_INLET, "w": W_INLET},
        fixed_dataset=True,  # thin criteria filter is fragile under continuous resampling
    )
    domain.add_constraint(inlet, "inlet")

    outlet = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"p": 0}, batch_size=BATCH_BC,
        criteria=(z_sym > Z_HI - EPS),
        lambda_weighting={"p": W_OUTLET},
        fixed_dataset=True,  # thin criteria filter is fragile under continuous resampling
    )
    domain.add_constraint(outlet, "outlet")

    interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=BATCH_R,
        lambda_weighting={"continuity": W_PDE, "momentum_x": W_PDE,
                         "momentum_y": W_PDE, "momentum_z": W_PDE},
        fixed_dataset=False,
    )
    domain.add_constraint(interior, "interior")

    slv = Solver(cfg, domain)
    setup_s = time.perf_counter() - t0_setup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0_train = time.perf_counter()
    slv.solve()
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0_train
    epochs = cfg.training.max_steps

    # ── score against the exact Hagen-Poiseuille profile (z playing the
    #    axial-x role; w playing the axial-u role -- see module docstring) ──
    rng = np.random.default_rng(99)
    n_val = 3000
    rr = R * np.sqrt(rng.uniform(0.0, 1.0, n_val))
    th = rng.uniform(0.0, 2 * np.pi, n_val)
    x_v = (rr * np.cos(th)).astype(np.float32)
    y_v = (rr * np.sin(th)).astype(np.float32)
    z_v = rng.uniform(Z_LO, Z_HI, n_val).astype(np.float32)
    r2_v = x_v ** 2 + y_v ** 2
    w_exact = U_MAX * (1.0 - r2_v / R ** 2)
    dpdz = -4.0 * NU * U_MAX / R ** 2
    p_exact = dpdz * (z_v - Z_HI)

    net.eval()
    with torch.no_grad():
        xt = torch.tensor(x_v, dtype=torch.float32, device=device)[:, None]
        yt = torch.tensor(y_v, dtype=torch.float32, device=device)[:, None]
        zt = torch.tensor(z_v, dtype=torch.float32, device=device)[:, None]
        out = net.forward({"x": xt, "y": yt, "z": zt})
        w_pred = out["w"].detach().cpu().numpy()[:, 0]
        p_pred = out["p"].detach().cpu().numpy()[:, 0]

    def rel_l2(pred, exact):
        return float(np.linalg.norm(pred - exact) / (np.linalg.norm(exact) + 1e-10))

    rel_l2_w = rel_l2(w_pred, w_exact)
    rel_l2_p = rel_l2(p_pred, p_exact)
    print(f"\n[physicsnemo pipe_flow3d] device={device}  epochs={epochs}  "
         f"setup={setup_s:.2f}s  train={train_s:.2f}s  "
         f"ms/epoch={1e3 * train_s / epochs:.3f}  "
         f"rel_L2(w)={rel_l2_w:.4e}  rel_L2(p)={rel_l2_p:.4e}")

    out = {"framework": "nvidia-physicsnemo-sym", "problem": "pipe_flow_3d",
          "epochs": epochs, "device": str(device),
          "setup_s": setup_s, "train_s": train_s,
          "ms_per_epoch": 1e3 * train_s / epochs,
          "rel_l2_axial_velocity": rel_l2_w, "rel_l2_pressure": rel_l2_p}
    out_path = os.path.join(_HERE, "result_pipe_flow3d.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Result saved -> {out_path}")


if __name__ == "__main__":
    run()
