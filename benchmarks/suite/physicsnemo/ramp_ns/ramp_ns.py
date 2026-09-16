"""2-D viscous compression-ramp SBLI (Ramp NS), solved with NVIDIA
PhysicsNeMo (Sym) -- matching
``benchmarks/suite/ablations/ablate_qr_deim_ramp_ns.py`` (the "none" arm)
as closely as PhysicsNeMo's constraint-based API allows:

  * same physics   : 2-D steady compressible Navier-Stokes, conservative
                     flux-divergence form, constant viscosity, Ducros-sensor
                     localised artificial viscosity (shock capturing) --
                     identical to underPINN.pde.compressible_ns_2d
                     .CompressibleNS2DPDE(mu_law="constant",
                     av_sensor="ducros", the un-overridden defaults the
                     ablation itself uses)
  * same problem   : M_inf=3, ramp angle=15 deg, gamma=1.4, Re=1e4, Pr=0.72,
                     art_visc=2e-3, domain L=2, H=1, ramp_start=0.8,
                     slip_end=0.15 (flat no-slip/slip split before the ramp)
  * same network   : 5 hidden layers x 128 units, tanh
  * same batching  : interior minibatched (batch_r=1536, fresh every step,
                     matching the ablation's actual continuous resampling of
                     its uniform pool); inlet/wall/slip/upper use the exact
                     same small *fixed* point counts the ablation draws once
                     and reuses every step (200/200/100/150 -- in the
                     ablation ``min(batch, pool_size) == pool_size`` for all
                     four, i.e. those are already effectively fixed full
                     batches, not real minibatches -- so ``fixed_dataset=
                     True`` at those exact sizes matches, not approximates)
  * same reference : scored exactly as ``RampNSEvaluator.evaluate()`` /
                     the ablation's own scoring block does -- relative L2 of
                     the outer-flow Mach field vs. the analytic oblique-
                     shock solution, using underPINN's own
                     ``CompressibleEulerPDE.oblique_shock`` and
                     ``relative_l2_error`` utilities directly (not a
                     separately re-derived reference).

Not matched (documented, not hidden):
  * PhysicsNeMo's interior draw is a single plain-uniform sample over the
    whole domain. The ablation's "none"-arm pool is a blend of three
    differently-biased sub-pools (40,000 plain-uniform + 5,000
    boundary-layer-clustered, geometrically stretched toward the wall +
    6,000 uniform-but-restricted-to-x>=0.25, frozen at its initial draw) --
    replicating that exact three-way blend inside PhysicsNeMo's geometry API
    was judged not worth the engineering cost for a *uniform-sampling*
    reference point (PhysicsNeMo has no adaptive-resampling analogue to
    compare against here regardless). This gives underPINN's baseline a
    denser near-wall/downstream sample than PhysicsNeMo's, a plausible
    accuracy advantage for underPINN not attributable to the network or
    solver alone.
  * PhysicsNeMo's default LR schedule is exponential decay, not underPINN's
    cosine decay (same caveat as every other problem in this suite).
  * The trapezoidal ramp domain is a 5-vertex ``Polygon``; inlet/upper/
    no-slip-wall/slip-wall regions are isolated from its combined boundary
    via position ``criteria`` (x/y thresholds, including the same
    ``y <= y_wall(x) + eps`` test ``RampGeometry`` itself uses to tell wall
    points from interior/farfield ones) rather than four separate curve
    objects -- exact set selection, not an approximation, just a different
    mechanism from underPINN's per-edge NumPy samplers.

Run (from this directory, using the physicsnemo venv):
    ../.venv/bin/python ramp_ns.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
from sympy import Function, Max, Number, Symbol, exp, tanh

import physicsnemo.sym
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.geometry.primitives_2d import Polygon
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Activation
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.solver import Solver

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))    # -> benchmarks/suite/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", ".."))  # -> repo root

M_INF, THETA_DEG, GAMMA, RE, PR = 3.0, 15.0, 1.4, 1.0e4, 0.72
L, H, RAMP_START, SLIP_END = 2.0, 1.0, 0.8, 0.15
AV_S = 0.05   # Ducros-sensor width; art_visc itself now comes from cfg.art_visc
LAYERS_HIDDEN, LAYER_SIZE = 5, 128
BATCH_R = 1536
N_INLET, N_WALL, N_SLIP, N_UPPER = 200, 200, 100, 150
W_PDE, W_INLET, W_WALL, W_SLIP, W_UPPER = 1.0, 100.0, 100.0, 80.0, 20.0

_theta = math.radians(THETA_DEG)
_tan_theta = math.tan(_theta)
T0 = 1.0 + 0.5 * (GAMMA - 1.0) * M_INF ** 2       # stagnation temperature
_EPS = 1e-4


class CompressibleNS2D(PDE):
    """2-D steady compressible NS, conservative flux-divergence form, with
    Ducros-sensor-localised artificial viscosity -- identical formulation to
    underPINN.pde.compressible_ns_2d.CompressibleNS2DPDE with
    mu_law="constant" (mu*=1) and av_sensor="ducros" (both un-overridden
    defaults the ablation itself uses). The network predicts raw
    (f_rho, u, v, f_T); rho=exp(f_rho) and T=exp(f_T) give the
    positivity-transformed physical state.
    """

    name = "CompressibleNS2D"

    def __init__(self, gamma: float, M_inf: float, Re: float, Pr: float,
                art_visc: float, av_s: float):
        x, y = Symbol("x"), Symbol("y")
        f_rho = Function("f_rho")(x, y)
        u = Function("u")(x, y)
        v = Function("v")(x, y)
        f_T = Function("f_T")(x, y)

        rho = exp(f_rho)
        T = exp(f_T)
        M2 = M_inf ** 2
        p = rho * T / (gamma * M2)
        E = p / (gamma - 1) + Number(0.5) * rho * (u ** 2 + v ** 2)

        mu = Number(1.0)   # mu_law="constant"
        u_x, u_y = u.diff(x), u.diff(y)
        v_x, v_y = v.diff(x), v.diff(y)
        T_x, T_y = T.diff(x), T.diff(y)

        txx = (Number(2.0) * mu / 3) * (2 * u_x - v_y)
        tyy = (Number(2.0) * mu / 3) * (2 * v_y - u_x)
        txy = mu * (u_y + v_x)
        kap = mu / ((gamma - 1) * Pr * M2)

        F = [rho * u, rho * u ** 2 + p, rho * u * v, (E + p) * u]
        G = [rho * v, rho * u * v, rho * v ** 2 + p, (E + p) * v]
        Fv = [Number(0), txx, txy, u * txx + v * txy + kap * T_x]
        Gv = [Number(0), txy, tyy, u * txy + v * tyy + kap * T_y]
        F_tot = [F[i] - Fv[i] / Re for i in range(4)]
        G_tot = [G[i] - Gv[i] / Re for i in range(4)]
        U = [rho, rho * u, rho * v, E]

        theta = u_x + v_y                      # dilatation
        omega = v_x - u_y                      # vorticity
        phi = theta ** 2 / (theta ** 2 + omega ** 2 + Number(1e-8))
        comp = Number(0.5) * (1 - tanh(theta / av_s))
        eps_local = art_visc * phi * comp

        self.equations = {}
        for name, Fi, Gi, Ui in zip(
                ("mass", "momentum_x", "momentum_y", "energy"), F_tot, G_tot, U):
            eq = Fi.diff(x) + Gi.diff(y)
            if art_visc > 0.0:
                eq = eq - eps_local * (Ui.diff(x, 2) + Ui.diff(y, 2))
            self.equations[name] = eq


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg) -> None:
    t0_setup = time.perf_counter()

    # overridable: ART_VISC=2e-4 env var (PhysicsNeMo's hydra config is a
    # strict structured schema -- DefaultPhysicsNeMoConfig -- that rejects
    # arbitrary new top-level keys like `art_visc=X` on the CLI, even when
    # declared in config.yaml; an env var sidesteps that entirely).
    #
    # IMPORTANT when sweeping this: PhysicsNeMo's checkpoint/restart
    # directory (`outputs/<hydra overrides>/ramp_ns`) is named from the
    # *hydra* CLI overrides only (e.g. `training.max_steps=30000`) -- the
    # ART_VISC env var isn't part of that, so two runs at different
    # ART_VISC values with the same hydra overrides silently collide on the
    # same directory. Hit this directly: an `ART_VISC=2e-4` run restored the
    # `ART_VISC=2e-3` run's already-`step=30000` checkpoint, did *zero*
    # additional training (train=3.82s instead of ~840s, ms/epoch=0.13
    # instead of ~28 -- the giveaway), and silently reported that stale
    # model evaluated against the *new* PDE's exact reference as if it were
    # a real 2e-4 result. Always pass a distinguishing `+av_tag=avX` (a
    # `+`-prefixed key, since it's not in the schema either, purely to
    # appear in the auto-generated directory name) alongside any
    # non-default ART_VISC, e.g.:
    #   ART_VISC=2e-4 python ramp_ns.py training.max_steps=30000 \
    #     +av_tag=av2e-4 jit=false
    art_visc = float(os.environ.get("ART_VISC", "2e-3"))
    ns = CompressibleNS2D(gamma=GAMMA, M_inf=M_INF, Re=RE, Pr=PR,
                          art_visc=art_visc, av_s=AV_S)
    net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("f_rho"), Key("u"), Key("v"), Key("f_T")],
        layer_size=LAYER_SIZE,
        nr_layers=LAYERS_HIDDEN,
        activation_fn=Activation.TANH,
        weight_norm=False,  # see burgers1d.py -- PhysicsNeMo's
        # FullyConnectedArch defaults to weight_norm=True; explicitly
        # disabled everywhere in this suite to match underPINN's plain MLP.
    )
    nodes = ns.make_nodes() + [net.make_node(name="ns_net")]

    y_wall_x = Max(0.0, (Symbol("x") - RAMP_START) * _tan_theta)
    verts = [(0.0, 0.0), (RAMP_START, 0.0),
             (L, float(y_wall_x.subs(Symbol("x"), L))),
             (L, H), (0.0, H)]
    geo = Polygon(verts)
    domain = Domain()
    x_sym, y_sym = Symbol("x"), Symbol("y")

    interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo,
        outvar={"mass": 0, "momentum_x": 0, "momentum_y": 0, "energy": 0},
        batch_size=BATCH_R,
        lambda_weighting={"mass": W_PDE, "momentum_x": W_PDE,
                         "momentum_y": W_PDE, "energy": W_PDE},
        fixed_dataset=False,
    )
    domain.add_constraint(interior, "interior")

    inlet = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"f_rho": 0.0, "u": 1.0, "v": 0.0, "f_T": 0.0},   # freestream
        batch_size=N_INLET,
        criteria=(x_sym < _EPS),
        lambda_weighting={"f_rho": W_INLET, "u": W_INLET,
                         "v": W_INLET, "f_T": W_INLET},
        fixed_dataset=True,
    )
    domain.add_constraint(inlet, "inlet")

    upper = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"f_rho": 0.0, "u": 1.0, "v": 0.0, "f_T": 0.0},   # freestream
        batch_size=N_UPPER,
        criteria=(y_sym > H - _EPS),
        lambda_weighting={"f_rho": W_UPPER, "u": W_UPPER,
                         "v": W_UPPER, "f_T": W_UPPER},
        fixed_dataset=True,
    )
    domain.add_constraint(upper, "upper")

    # "on the lower wall" = the same y <= y_wall(x)+eps test RampGeometry's
    # own sampler uses to tell wall points from interior/farfield ones --
    # robust to the x-range overlap between the flat and inclined wall
    # segments and the domain's other (upper/inlet/outlet) edges.
    on_wall = y_sym <= y_wall_x + _EPS
    slip = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"v": 0.0},
        batch_size=N_SLIP,
        criteria=(on_wall & (x_sym < SLIP_END)),
        lambda_weighting={"v": W_SLIP},
        fixed_dataset=True,
    )
    domain.add_constraint(slip, "slip_wall")

    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"u": 0.0, "v": 0.0, "f_T": float(np.log(T0))},
        batch_size=N_WALL,
        criteria=(on_wall & (x_sym >= SLIP_END)),
        lambda_weighting={"u": W_WALL, "v": W_WALL, "f_T": W_WALL},
        fixed_dataset=True,
    )
    domain.add_constraint(no_slip, "no_slip_wall")

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

    # ── Score exactly as RampNSEvaluator.evaluate() / the ablation's own
    #    scoring block does: relative L2 of the outer-flow Mach field vs.
    #    the analytic oblique-shock solution. ──
    from underPINN.geometry.ramp import RampGeometry
    from underPINN.pde.compressible_euler import CompressibleEulerPDE
    from underPINN.utils.metrics import relative_l2_error

    geom = RampGeometry(THETA_DEG, L=L, H=H, ramp_start=RAMP_START,
                        slip_end=SLIP_END)
    XX, YY, mask = geom.make_grid(Nx=140, Ny=110)
    band = 0.12 * H
    outer = mask & (YY > geom.y_wall(XX) + band)

    net.eval()
    with torch.no_grad():
        x_t = torch.tensor(XX.ravel(), dtype=torch.float32, device=device)[:, None]
        y_t = torch.tensor(YY.ravel(), dtype=torch.float32, device=device)[:, None]
        out = net.forward({"x": x_t, "y": y_t})
        u_p = out["u"].detach().cpu().numpy()[:, 0]
        v_p = out["v"].detach().cpu().numpy()[:, 0]
        T_p = torch.exp(out["f_T"]).detach().cpu().numpy()[:, 0]
    a_p = np.sqrt(T_p) / M_INF
    mach_pred = (np.sqrt(u_p ** 2 + v_p ** 2) / a_p).reshape(XX.shape)

    euler = CompressibleEulerPDE(None, gamma=GAMMA)
    shock = euler.oblique_shock(M_INF, THETA_DEG)
    beta = math.radians(shock["beta_deg"])
    dx = np.maximum(XX - RAMP_START, 0.0)
    below = (YY <= dx * math.tan(beta)) & (XX >= RAMP_START)
    mach_exact = np.where(below, shock["M2"], M_INF)

    rel_l2 = float(relative_l2_error(mach_pred[outer], mach_exact[outer]))

    print(f"\n[physicsnemo ramp_ns] device={device}  epochs={epochs}  "
         f"setup={setup_s:.2f}s  train={train_s:.2f}s  "
         f"ms/epoch={1e3 * train_s / epochs:.3f}  rel_L2(outer Mach)={rel_l2:.4e}")

    out_json = {
        "framework": "nvidia-physicsnemo-sym", "physicsnemo_sym_version":
            __import__("physicsnemo.sym").sym.__version__,
        "problem": "ramp_ns_sbli", "gamma": GAMMA, "M_inf": M_INF,
        "theta_deg": THETA_DEG, "Re": RE, "Pr": PR, "art_visc": art_visc,
        "epochs": epochs, "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "setup_s": setup_s, "train_s": train_s,
        "ms_per_epoch": 1e3 * train_s / epochs,
        "rel_l2": rel_l2,
        "layers_hidden": LAYERS_HIDDEN, "layer_size": LAYER_SIZE,
        "batch_r": BATCH_R, "n_inlet": N_INLET, "n_wall": N_WALL,
        "n_slip": N_SLIP, "n_upper": N_UPPER,
    }
    # tag the filename by art_visc so a sweep doesn't overwrite the
    # baseline (2e-3, the original comparison) result.json
    fname = ("result.json" if art_visc == 2e-3
            else f"result_av{art_visc:g}.json")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    with open(out_path, "w") as fh:
        json.dump(out_json, fh, indent=2)
    print(f"Result saved -> {out_path}")


if __name__ == "__main__":
    run()
