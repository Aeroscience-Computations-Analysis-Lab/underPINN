"""1-D Toro test 3 (Woodward-Colella blast wave, left half): unsteady
compressible Euler with a 5-decade pressure jump, solved with NVIDIA
PhysicsNeMo (Sym) -- matching
``benchmarks/suite/ablations/ablate_qr_deim.py`` (the "none" arm's setup;
that script is where "test QR-DEIM-R hybrid 80/20 on Toro3" is actually run,
on underPINN's own JAX side) as closely as PhysicsNeMo's constraint-based
API allows, so the two frameworks' *baseline* (no adaptive resampling)
numbers are directly comparable:

  * same physics   : 1-D unsteady compressible Euler (mass/momentum/energy,
                     conservative form) + fixed artificial viscosity
                     art_visc=0.001 on d^2U/dx^2, gamma=1.4
  * same problem   : x in [0,1], x0=0.5, t_final=0.012,
                     LEFT (rho,u,p)=(1,0,1000), RIGHT (rho,u,p)=(1,0,0.01),
                     non-dimensionalised identically (rho_ref=max(rho),
                     p_ref=max(p), u_ref=sqrt(p_ref/rho_ref), t_ref=1/u_ref)
                     -- so LEFT_nd=(1,0,1), RIGHT_nd=(1,0,1e-5)
  * same transform : exp/log positivity (rho=exp(f_rho), p=exp(f_p)), the
                     "transform=exp" mode underPINN's Euler1DUnsteadyPDE uses
  * same network   : 5 hidden layers x 128 units, tanh
  * same batching  : per-step minibatches (batch_r=2048, batch_ic=400,
                     batch_bc=300-per-side), matching the ablation's actual
                     minibatched (not full-batch) convention -- this is the
                     same full-batch-vs-minibatch distinction already found
                     and fixed for the other 5 problems in ../multi/, so it
                     is applied correctly here from the start.
  * same reference : scored against the identical exact Riemann solution
                     underPINN itself uses (underPINN.utils.riemann
                     .exact_riemann_1d), on the same t=t_final grid.

Not matched (documented, not hidden): PhysicsNeMo's default LR schedule is
exponential decay, not underPINN's cosine decay (same caveat as every other
problem in this suite). The IC/BC targets are expressed as a single sympy
``Piecewise`` per constraint (selecting LEFT vs RIGHT by x-position)
rather than underPINN's per-point NumPy array -- exact, not an
approximation, just a different mechanism for the same discontinuous
target. This case has no adaptive-resampling analogue on the PhysicsNeMo
side (PhysicsNeMo has no built-in QR-DEIM-R/RAD); it exists purely as the
*uniform-sampling* framework-vs-framework reference point the ablation's
own "none" arm is compared against.

Run (from this directory, using the physicsnemo venv):
    ../.venv/bin/python toro3.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from sympy import Function, Number, Piecewise, Symbol, exp

import physicsnemo.sym
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Activation
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.solver import Solver

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))    # -> benchmarks/suite/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", ".."))  # -> repo root

GAMMA = 1.4
X0, T_FINAL = 0.5, 0.012
LEFT, RIGHT = (1.0, 0.0, 1000.0), (1.0, 0.0, 0.01)
FIXED_AV = 0.001
BATCH_R, BATCH_IC, BATCH_BC_PER_SIDE = 2048, 400, 300
W_PDE, W_IC, W_BC = 1.0, 100.0, 10.0
LAYERS_HIDDEN, LAYER_SIZE = 5, 128

# ── identical non-dimensionalisation to ablate_qr_deim.py::make_problem ──
_rho_ref = max(LEFT[0], RIGHT[0])
_p_ref = max(LEFT[2], RIGHT[2])
_u_ref = float(np.sqrt(_p_ref / _rho_ref))
_t_ref = 1.0 / _u_ref
LEFT_ND = (LEFT[0] / _rho_ref, LEFT[1] / _u_ref, LEFT[2] / _p_ref)
RIGHT_ND = (RIGHT[0] / _rho_ref, RIGHT[1] / _u_ref, RIGHT[2] / _p_ref)
TF_ND = T_FINAL / _t_ref


class Euler1DUnsteady(PDE):
    """1-D unsteady compressible Euler, conservative form, with fixed
    artificial viscosity on d^2U/dx^2 -- identical formulation to
    underPINN.pde.euler_1d_unsteady.Euler1DUnsteadyPDE (transform="exp",
    fixed art_visc). The network predicts raw (f_rho, u, f_p); rho=exp(f_rho)
    and p=exp(f_p) give the positivity-transformed physical state, exactly
    as the JAX side's ``_pos()`` does for ``transform="exp"``.
    """

    name = "Euler1DUnsteady"

    def __init__(self, gamma: float = 1.4, art_visc: float = 0.0):
        x, t = Symbol("x"), Symbol("t")
        f_rho = Function("f_rho")(x, t)
        u = Function("u")(x, t)
        f_p = Function("f_p")(x, t)

        rho = exp(f_rho)
        p = exp(f_p)
        E = p / (gamma - 1) + Number(0.5) * rho * u ** 2

        U = [rho, rho * u, E]
        F = [rho * u, rho * u ** 2 + p, (E + p) * u]

        self.equations = {}
        for name, Ui, Fi in zip(("mass", "momentum", "energy"), U, F):
            eq = Ui.diff(t) + Fi.diff(x)
            if art_visc > 0.0:
                eq = eq - art_visc * Ui.diff(x, 2)
            self.equations[name] = eq


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg) -> None:
    t0_setup = time.perf_counter()

    euler = Euler1DUnsteady(gamma=GAMMA, art_visc=FIXED_AV)
    net = FullyConnectedArch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("f_rho"), Key("u"), Key("f_p")],
        layer_size=LAYER_SIZE,
        nr_layers=LAYERS_HIDDEN,
        activation_fn=Activation.TANH,
        weight_norm=False,  # see burgers1d.py -- PhysicsNeMo's
        # FullyConnectedArch defaults to weight_norm=True, a confound found
        # and fixed earlier in this comparison suite; explicitly disabled
        # everywhere here to match underPINN's plain MLP.
    )
    nodes = euler.make_nodes() + [net.make_node(name="euler_net")]

    geo = Line1D(0.0, 1.0)
    domain = Domain()
    x_sym, t_sym = Symbol("x"), Symbol("t")

    f_rho_l, u_l, f_p_l = (0.0, LEFT_ND[1], float(np.log(LEFT_ND[2])))
    f_rho_r, u_r, f_p_r = (0.0, RIGHT_ND[1], float(np.log(RIGHT_ND[2])))
    # LEFT_ND[0] == RIGHT_ND[0] == 1.0 here (rho starts uniform for Toro-3;
    # only p is discontinuous), so f_rho is the same constant on both sides
    # -- still expressed as a Piecewise for correctness/generality rather
    # than assuming that cancellation.
    f_rho_pw = Piecewise((f_rho_l, x_sym < X0), (f_rho_r, True))
    u_pw = Piecewise((u_l, x_sym < X0), (u_r, True))
    f_p_pw = Piecewise((f_p_l, x_sym < X0), (f_p_r, True))

    ic = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo,
        outvar={"f_rho": f_rho_pw, "u": u_pw, "f_p": f_p_pw},
        batch_size=BATCH_IC,
        parameterization={t_sym: 0.0},
        lambda_weighting={"f_rho": W_IC, "u": W_IC, "f_p": W_IC},
        fixed_dataset=False,
    )
    domain.add_constraint(ic, "ic")

    # Line1D(0,1)'s boundary is exactly the two endpoints {x=0, x=1}; the
    # same Piecewise selects LEFT at x=0 and RIGHT at x=1 automatically --
    # no criteria= split needed (unlike the pipe-flow cylinder's thin
    # surface filters, this is two disjoint points, not a fragile band).
    bc = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"f_rho": f_rho_pw, "u": u_pw, "f_p": f_p_pw},
        batch_size=2 * BATCH_BC_PER_SIDE,
        parameterization={t_sym: (0.0, TF_ND)},
        lambda_weighting={"f_rho": W_BC, "u": W_BC, "f_p": W_BC},
        fixed_dataset=False,
    )
    domain.add_constraint(bc, "bc")

    interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo,
        outvar={"mass": 0, "momentum": 0, "energy": 0},
        batch_size=BATCH_R,
        parameterization={t_sym: (0.0, TF_ND)},
        lambda_weighting={"mass": W_PDE, "momentum": W_PDE, "energy": W_PDE},
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

    # ── Score against the identical exact Riemann solution underPINN uses ──
    from underPINN.utils.riemann import exact_riemann_1d
    from underPINN.utils.metrics import relative_l2_error

    Nx = 400
    xg = np.linspace(0.0, 1.0, Nx, dtype="f4")
    net.eval()
    with torch.no_grad():
        x_t = torch.tensor(xg, dtype=torch.float32, device=device)[:, None]
        t_t = torch.full_like(x_t, TF_ND)
        out = net.forward({"x": x_t, "t": t_t})
        rho_p = torch.exp(out["f_rho"]).detach().cpu().numpy()[:, 0]
        u_p = out["u"].detach().cpu().numpy()[:, 0]
        p_p = torch.exp(out["f_p"]).detach().cpu().numpy()[:, 0]
    pred = np.stack([rho_p, u_p, p_p], axis=1)

    re, ue, pe = exact_riemann_1d(xg, TF_ND, X0, GAMMA, LEFT_ND, RIGHT_ND)
    exact = np.stack([re, ue, pe], axis=1)
    rel_l2 = float(relative_l2_error(pred, exact))

    print(f"\n[physicsnemo toro3] device={device}  epochs={epochs}  "
         f"setup={setup_s:.2f}s  train={train_s:.2f}s  "
         f"ms/epoch={1e3 * train_s / epochs:.3f}  rel_L2(exact Riemann)={rel_l2:.4e}")

    out_json = {
        "framework": "nvidia-physicsnemo-sym", "physicsnemo_sym_version":
            __import__("physicsnemo.sym").sym.__version__,
        "problem": "toro3_blast_wave", "gamma": GAMMA, "art_visc": FIXED_AV,
        "epochs": epochs, "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "setup_s": setup_s, "train_s": train_s,
        "ms_per_epoch": 1e3 * train_s / epochs,
        "rel_l2": rel_l2,
        "layers_hidden": LAYERS_HIDDEN, "layer_size": LAYER_SIZE,
        "batch_r": BATCH_R, "batch_ic": BATCH_IC,
        "batch_bc_per_side": BATCH_BC_PER_SIDE,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "result.json")
    with open(out_path, "w") as fh:
        json.dump(out_json, fh, indent=2)
    print(f"Result saved -> {out_path}")


if __name__ == "__main__":
    run()
