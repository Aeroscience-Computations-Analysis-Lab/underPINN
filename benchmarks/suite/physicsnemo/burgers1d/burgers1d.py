"""1-D viscous Burgers, solved with NVIDIA PhysicsNeMo (Sym) -- a genuine
third-party PINN framework comparison, not just eager-vs-compiled PyTorch.

Matches underPINN's own Burgers baseline
(benchmarks/suite/baselines/burgers_baselines.py) as closely as
PhysicsNeMo's constraint-based API allows:

  * same physics   : u_t + u*u_x = nu*u_xx, nu=0.01, domain x in [-1,1],
                     t in [0, 1.5], IC u(x,0)=-sin(pi x), BC u(+-1,t)=0
  * same network   : 5 hidden layers x 64 units, tanh
  * same batch     : full-batch every step (fixed_dataset=True,
                     batch_size = the full pool size, matching
                     burgers_baselines.py's un-minibatched convention)
  * same reference : scored against the identical Cole-Hopf exact solution
                     underPINN itself uses (underPINN.utils.operator_datagen
                     .burgers1d_exact) -- not a separately-computed number.

Not matched (documented, not hidden): PhysicsNeMo's default Adam LR
schedule is exponential-decay (tf_exponential_lr), not underPINN's cosine
decay; IC/BC loss weights use PhysicsNeMo's lambda_weighting mechanism at
the same numeric values (100, 10) but the two frameworks may not combine
per-constraint losses identically. This case is run to completion once as
a real, working install -- it is not tuned or cherry-picked.

Run (from this directory, using the physicsnemo venv):
    ../.venv/bin/python burgers1d.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from sympy import Function, Symbol

import physicsnemo.sym
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.hydra import to_absolute_path
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Activation
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.solver import Solver

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))    # -> benchmarks/suite/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", ".."))  # -> repo root

NU = 0.01
T_MAX = 1.5
N_R, N_IC, N_BC = 20000, 200, 300
W_IC, W_BC = 100.0, 10.0
LAYERS_HIDDEN, LAYER_SIZE = 5, 64


class Burgers1D(PDE):
    """u_t + u*u_x - nu*u_xx = 0 -- identical to underPINN.pde.burgers.BurgersPDE."""

    name = "Burgers1D"

    def __init__(self, nu: float = 0.01):
        x, t = Symbol("x"), Symbol("t")
        u = Function("u")(x, t)
        self.equations = {"burgers": u.diff(t) + u * u.diff(x) - nu * u.diff(x, 2)}


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg) -> None:
    t0_setup = time.perf_counter()

    burgers = Burgers1D(nu=NU)
    net = FullyConnectedArch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("u")],
        layer_size=LAYER_SIZE,
        nr_layers=LAYERS_HIDDEN,
        activation_fn=Activation.TANH,
        weight_norm=False,  # match underPINN's plain MLP -- PhysicsNeMo's
        # FullyConnectedArch defaults to weight_norm=True (Salimans & Kingma
        # 2016) on every hidden layer, an undocumented architectural
        # difference from a genuinely plain MLP, not something underPINN's
        # MLP/FourierMLP/GatedMLP do.
    )
    nodes = burgers.make_nodes() + [net.make_node(name="burgers_net")]

    geo = Line1D(-1, 1)
    domain = Domain()

    # PhysicsNeMo's outvar takes a SymPy expression, not a per-point array, so
    # the IC target -sin(pi x) is expressed symbolically rather than sampled
    # in Python -- exact, not an approximation of underPINN's IC handling.
    from sympy import pi, sin
    x_sym = Symbol("x")
    ic = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo,
        outvar={"u": -sin(pi * x_sym)},
        batch_size=N_IC,
        parameterization={Symbol("t"): 0.0},
        lambda_weighting={"u": W_IC},
        fixed_dataset=True,
    )
    domain.add_constraint(ic, "ic")

    bc = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"u": 0},
        batch_size=N_BC,
        parameterization={Symbol("t"): (0.0, T_MAX)},
        lambda_weighting={"u": W_BC},
        fixed_dataset=True,
    )
    domain.add_constraint(bc, "bc")

    interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo,
        outvar={"burgers": 0},
        batch_size=N_R,
        parameterization={Symbol("t"): (0.0, T_MAX)},
        fixed_dataset=True,
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

    # ── Score against the identical Cole-Hopf exact reference underPINN uses ──
    from underPINN.utils.operator_datagen import burgers1d_exact
    from underPINN.utils.metrics import relative_l2_error

    Nx_eval, Nt_eval = 101, 41
    x_eval = np.linspace(-1.0, 1.0, Nx_eval)
    t_eval = np.linspace(0.0, T_MAX, Nt_eval)
    u_exact = burgers1d_exact(x_eval, t_eval, nu=NU, u0_mode=1)   # (Nx, Nt)

    XX, TT = np.meshgrid(x_eval, t_eval, indexing="ij")
    net.eval()
    with torch.no_grad():
        x_t = torch.tensor(XX.ravel(), dtype=torch.float32, device=device)[:, None]
        t_t = torch.tensor(TT.ravel(), dtype=torch.float32, device=device)[:, None]
        u_pred = net.forward({"x": x_t, "t": t_t})["u"].detach().cpu().numpy()
    u_pred = u_pred.reshape(Nx_eval, Nt_eval)

    rel_l2 = float(relative_l2_error(u_pred, u_exact))
    print(f"\n[physicsnemo burgers1d] device={device}  epochs={epochs}  "
         f"setup={setup_s:.2f}s  train={train_s:.2f}s  "
         f"ms/epoch={1e3 * train_s / epochs:.3f}  rel_L2(Cole-Hopf)={rel_l2:.4e}")

    import json
    out = {
        "framework": "nvidia-physicsnemo-sym", "physicsnemo_sym_version":
            __import__("physicsnemo.sym").sym.__version__,
        "problem": "burgers_1d", "nu": NU, "epochs": epochs,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "setup_s": setup_s, "train_s": train_s,
        "ms_per_epoch": 1e3 * train_s / epochs,
        "rel_l2_vs_cole_hopf": rel_l2,
        "layers_hidden": LAYERS_HIDDEN, "layer_size": LAYER_SIZE,
        "n_r": N_R, "n_ic": N_IC, "n_bc": N_BC,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "result.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Result saved -> {out_path}")


if __name__ == "__main__":
    run()
