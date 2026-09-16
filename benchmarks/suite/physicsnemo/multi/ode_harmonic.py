"""ODE harmonic oscillator, PhysicsNeMo Sym vs underPINN -- matches
examples/ode/ode_test.py's harmonic_oscillator case: u'' + omega^2 u = 0,
omega=2, u(0)=1, u'(0)=0, exact u = cos(omega t).

No 0-D domain primitive in physicsnemo.sym.geometry -- reuses Line1D and
labels its coordinate "x" throughout (matching the PDE's own symbol), with
"x" playing the role of time for this one case only.

Run:
    env -u SLURM_PROCID ../.venv/bin/python ode_harmonic.py training.max_steps=3000 jit=false
"""
from __future__ import annotations

import json
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
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Activation
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.solver import Solver

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", ".."))

OMEGA = 2.0
T_MAX = 5.0
U0, V0 = 1.0, 0.0
N_R = 10000
IC_W, IC_DOT_W = 100.0, 100.0
LAYERS_HIDDEN, LAYER_SIZE = 3, 64


class HarmonicOscillator(PDE):
    """u'' + omega^2 u = 0 -- identical to
    underPINN.pde.ode.HarmonicOscillatorODE. ``x`` plays the role of ``t``."""

    name = "HarmonicOscillator"

    def __init__(self, omega: float = 1.0):
        x = Symbol("x")
        u = Function("u")(x)
        self.equations = {"harmonic": u.diff(x, 2) + omega ** 2 * u}


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg) -> None:
    t0_setup = time.perf_counter()

    ho = HarmonicOscillator(omega=OMEGA)
    net = FullyConnectedArch(
        input_keys=[Key("x")], output_keys=[Key("u")],
        layer_size=LAYER_SIZE, nr_layers=LAYERS_HIDDEN,
        activation_fn=Activation.TANH,
        weight_norm=False,  # match underPINN's plain MLP -- PhysicsNeMo's
        # FullyConnectedArch defaults to weight_norm=True (Salimans & Kingma
        # 2016) on every hidden layer, an undocumented architectural
        # difference from a genuinely plain MLP, not something underPINN's
        # MLP/FourierMLP/GatedMLP do.
    )
    nodes = ho.make_nodes() + [net.make_node(name="ho_net")]

    geo = Line1D(0.0, T_MAX)
    domain = Domain()

    x_sym = Symbol("x")
    # IC at x=0 is the *left endpoint* of Line1D(0, T_MAX) -- a boundary
    # constraint (a discrete 2-point curve set: x=0 and x=T_MAX), filtered
    # to just x=0. (A PointwiseInteriorConstraint with criteria=x<eps would
    # be a probability-zero event under continuous interior sampling and
    # essentially never actually match any points.)
    ic = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"u": U0, "u__x": V0}, batch_size=16,
        criteria=(x_sym < 1e-6),
        lambda_weighting={"u": IC_W, "u__x": IC_DOT_W},
        fixed_dataset=True,
    )
    domain.add_constraint(ic, "ic")

    # batch size matches underPINN's TrainingConfig default batch_r=4096
    # (ode/config.yaml doesn't override it); fixed_dataset=False so
    # PhysicsNeMo draws a fresh sample every step (see pipe_flow3d.py for
    # the longer note on this not being bit-identical resampling semantics).
    interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo,
        outvar={"harmonic": 0}, batch_size=4096,
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

    t_eval = np.linspace(0.0, T_MAX, 2000)
    u_exact = U0 * np.cos(OMEGA * t_eval)

    net.eval()
    with torch.no_grad():
        x_t = torch.tensor(t_eval, dtype=torch.float32, device=device)[:, None]
        u_pred = net.forward({"x": x_t})["u"].detach().cpu().numpy()[:, 0]

    rel_l2 = float(np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-10))
    print(f"\n[physicsnemo ode_harmonic] device={device}  epochs={epochs}  "
         f"setup={setup_s:.2f}s  train={train_s:.2f}s  "
         f"ms/epoch={1e3 * train_s / epochs:.3f}  rel_L2={rel_l2:.4e}")

    out = {"framework": "nvidia-physicsnemo-sym", "problem": "ode_harmonic",
          "epochs": epochs, "device": str(device),
          "setup_s": setup_s, "train_s": train_s,
          "ms_per_epoch": 1e3 * train_s / epochs, "rel_l2": rel_l2}
    out_path = os.path.join(_HERE, "result_ode_harmonic.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Result saved -> {out_path}")


if __name__ == "__main__":
    run()
