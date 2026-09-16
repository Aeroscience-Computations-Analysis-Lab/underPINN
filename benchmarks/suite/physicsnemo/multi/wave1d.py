"""1-D wave equation, PhysicsNeMo Sym vs underPINN -- matches
examples/wave/wave.py: u_tt = c^2 u_xx, c=1, x in [-1,1], t in [0,2],
IC u(x,0)=sin(pi x), u_t(x,0)=0, BC u(+-1,t)=0, exact sin(pi x)cos(c pi t).

Not matched (documented): underPINN uses a trainable-sigma FourierMLP
[2,64,64,64,1]; this uses a plain FullyConnectedArch of the same depth/width
(PhysicsNeMo has no direct drop-in for a *trainable* Fourier-feature input
encoding matching underPINN's custom layer).

Run:
    env -u SLURM_PROCID ../.venv/bin/python wave1d.py training.max_steps=5000 jit=false
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from sympy import Symbol, cos, sin, pi as sym_pi

import physicsnemo.sym
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.eq.pdes.wave_equation import WaveEquation
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Activation
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.solver import Solver

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", ".."))

C = 1.0
T_MAX = 2.0
N_R, N_IC, N_BC = 10000, 300, 300
IC_W, IC_DOT_W, BC_W = 100.0, 100.0, 10.0
LAYERS_HIDDEN, LAYER_SIZE = 3, 64


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg) -> None:
    t0_setup = time.perf_counter()

    wave = WaveEquation(u="u", c=C, dim=1, time=True)
    net = FullyConnectedArch(
        input_keys=[Key("x"), Key("t")], output_keys=[Key("u")],
        layer_size=LAYER_SIZE, nr_layers=LAYERS_HIDDEN,
        activation_fn=Activation.TANH,
        weight_norm=False,  # match underPINN's plain MLP -- PhysicsNeMo's
        # FullyConnectedArch defaults to weight_norm=True (Salimans & Kingma
        # 2016) on every hidden layer, an undocumented architectural
        # difference from a genuinely plain MLP, not something underPINN's
        # MLP/FourierMLP/GatedMLP do.
    )
    nodes = wave.make_nodes() + [net.make_node(name="wave_net")]

    geo = Line1D(-1, 1)
    domain = Domain()

    # batch sizes match underPINN's per-step minibatch sizes (batch_r=2048,
    # batch_i=256, batch_b=256 in wave/config.yaml); fixed_dataset=False so
    # PhysicsNeMo draws a fresh sample every step (same per-step point count
    # as underPINN's random-index minibatching, not bit-identical resampling
    # semantics -- see pipe_flow3d.py's longer note on this).
    BATCH_IC, BATCH_BC, BATCH_R = 256, 256, 2048

    x_sym, t_sym = Symbol("x"), Symbol("t")
    ic = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo,
        outvar={"u": sin(sym_pi * x_sym), "u__t": 0},
        batch_size=BATCH_IC,
        parameterization={t_sym: 0.0},
        lambda_weighting={"u": IC_W, "u__t": IC_DOT_W},
        fixed_dataset=False,
    )
    domain.add_constraint(ic, "ic")

    bc = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"u": 0}, batch_size=BATCH_BC,
        parameterization={t_sym: (0.0, T_MAX)},
        lambda_weighting={"u": BC_W},
        fixed_dataset=False,
    )
    domain.add_constraint(bc, "bc")

    interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo,
        outvar={"wave_equation": 0}, batch_size=BATCH_R,
        parameterization={t_sym: (0.0, T_MAX)},
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

    Nx_eval, Nt_eval = 101, 41
    x_eval = np.linspace(-1.0, 1.0, Nx_eval)
    t_eval = np.linspace(0.0, T_MAX, Nt_eval)
    XX, TT = np.meshgrid(x_eval, t_eval, indexing="ij")
    u_exact = np.sin(np.pi * XX) * np.cos(C * np.pi * TT)

    net.eval()
    with torch.no_grad():
        x_t = torch.tensor(XX.ravel(), dtype=torch.float32, device=device)[:, None]
        t_t = torch.tensor(TT.ravel(), dtype=torch.float32, device=device)[:, None]
        u_pred = net.forward({"x": x_t, "t": t_t})["u"].detach().cpu().numpy()
    u_pred = u_pred.reshape(Nx_eval, Nt_eval)

    rel_l2 = float(np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-10))
    print(f"\n[physicsnemo wave1d] device={device}  epochs={epochs}  "
         f"setup={setup_s:.2f}s  train={train_s:.2f}s  "
         f"ms/epoch={1e3 * train_s / epochs:.3f}  rel_L2={rel_l2:.4e}")

    out = {"framework": "nvidia-physicsnemo-sym", "problem": "wave_1d",
          "epochs": epochs, "device": str(device),
          "setup_s": setup_s, "train_s": train_s,
          "ms_per_epoch": 1e3 * train_s / epochs, "rel_l2": rel_l2}
    out_path = os.path.join(_HERE, "result_wave1d.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Result saved -> {out_path}")


if __name__ == "__main__":
    run()
