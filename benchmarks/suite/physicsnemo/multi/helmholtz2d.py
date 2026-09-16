"""2-D Helmholtz equation, PhysicsNeMo Sym vs underPINN -- matches
examples/helmholtz/helmholtz.py: Delta u + k^2 u = f on [0,1]^2, k=4,
u=0 on all edges, f = -(2 pi^2 - k^2) sin(pi x) sin(pi y),
exact u = sin(pi x) sin(pi y).

No built-in Helmholtz PDE in physicsnemo.sym.eq.pdes -- a short custom
PDE, same pattern as Burgers1D in burgers1d/burgers1d.py.

Not matched (documented): underPINN uses a trainable-sigma FourierMLP
[2,128,128,128,1]; this uses a plain FullyConnectedArch of the same
depth/width.

Run:
    env -u SLURM_PROCID ../.venv/bin/python helmholtz2d.py training.max_steps=10000 jit=false
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from sympy import Function, Symbol, sin, pi as sym_pi

import physicsnemo.sym
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Activation
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.solver import Solver

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", ".."))

K = 4.0
N_R, N_BC = 8000, 600      # N_BC per edge
BC_W = 100.0
LAYERS_HIDDEN, LAYER_SIZE = 3, 128


class Helmholtz2D(PDE):
    """Delta u + k^2 u - f = 0 -- identical to underPINN.pde.helmholtz.HelmholtzPDE."""

    name = "Helmholtz2D"

    def __init__(self, k: float = 1.0):
        x, y = Symbol("x"), Symbol("y")
        u = Function("u")(x, y)
        f = -(2.0 * sym_pi ** 2 - k ** 2) * sin(sym_pi * x) * sin(sym_pi * y)
        self.equations = {"helmholtz": u.diff(x, 2) + u.diff(y, 2) + k ** 2 * u - f}


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg) -> None:
    t0_setup = time.perf_counter()

    helm = Helmholtz2D(k=K)
    net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y")], output_keys=[Key("u")],
        layer_size=LAYER_SIZE, nr_layers=LAYERS_HIDDEN,
        activation_fn=Activation.TANH,
        weight_norm=False,  # match underPINN's plain MLP -- PhysicsNeMo's
        # FullyConnectedArch defaults to weight_norm=True (Salimans & Kingma
        # 2016) on every hidden layer, an undocumented architectural
        # difference from a genuinely plain MLP, not something underPINN's
        # MLP/FourierMLP/GatedMLP do.
    )
    nodes = helm.make_nodes() + [net.make_node(name="helmholtz_net")]

    geo = Rectangle((0, 0), (1, 1))
    domain = Domain()

    # batch sizes match underPINN's per-step minibatch sizes (batch_r=2048,
    # batch_b=256 in helmholtz/config.yaml); fixed_dataset=False so
    # PhysicsNeMo draws a fresh sample every step (see pipe_flow3d.py for
    # the longer note on this not being bit-identical resampling semantics).
    BATCH_BC, BATCH_R = 256, 2048

    bc = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"u": 0}, batch_size=BATCH_BC,
        lambda_weighting={"u": BC_W},
        fixed_dataset=False,
    )
    domain.add_constraint(bc, "bc")

    interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo,
        outvar={"helmholtz": 0}, batch_size=BATCH_R,
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

    N_eval = 101
    x_eval = np.linspace(0.0, 1.0, N_eval)
    y_eval = np.linspace(0.0, 1.0, N_eval)
    XX, YY = np.meshgrid(x_eval, y_eval, indexing="ij")
    u_exact = np.sin(np.pi * XX) * np.sin(np.pi * YY)

    net.eval()
    with torch.no_grad():
        x_t = torch.tensor(XX.ravel(), dtype=torch.float32, device=device)[:, None]
        y_t = torch.tensor(YY.ravel(), dtype=torch.float32, device=device)[:, None]
        u_pred = net.forward({"x": x_t, "y": y_t})["u"].detach().cpu().numpy()
    u_pred = u_pred.reshape(N_eval, N_eval)

    rel_l2 = float(np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-10))
    print(f"\n[physicsnemo helmholtz2d] device={device}  epochs={epochs}  "
         f"setup={setup_s:.2f}s  train={train_s:.2f}s  "
         f"ms/epoch={1e3 * train_s / epochs:.3f}  rel_L2={rel_l2:.4e}")

    out = {"framework": "nvidia-physicsnemo-sym", "problem": "helmholtz_2d",
          "epochs": epochs, "device": str(device),
          "setup_s": setup_s, "train_s": train_s,
          "ms_per_epoch": 1e3 * train_s / epochs, "rel_l2": rel_l2}
    out_path = os.path.join(_HERE, "result_helmholtz2d.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Result saved -> {out_path}")


if __name__ == "__main__":
    run()
