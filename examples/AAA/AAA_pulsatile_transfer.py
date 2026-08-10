"""3-D Unsteady Pulsatile Flow through an Axisymmetric AAA — Time-Marching TL.

Run directly or via the CLI:

    python examples/AAA/AAA_pulsatile_transfer.py
    python examples/AAA/AAA_pulsatile_transfer.py myconfig.yaml
    python -m underPINN run examples/AAA/AAA_pulsatile_transfer.yaml

Same time-marching transfer-learning framework as ``pipe_flow_pulsatile_transfer``
but in the **AAA (bulge)** geometry, Newtonian rheology.  The inlet imposes a
pulsatile parabolic profile

    u_inlet(r, t) = (V_max + V_amp · sin(2π t / T_period)) · (1 − r²/R_vessel²)

with no-slip on the curved wall and p = 0 at the outlet.  The steady
Poiseuille profile is used as the initial condition at t = 0.  Overlapping
windows are supported via ``time_marching.stride < dT``.

Network: (x, y, z, t) → (u, v, w, p)
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from underPINN.config.loader import cfg_get
from underPINN.pde.navier_stokes_3d import UnsteadyNS3DPDE
from underPINN.geometry.aaa import BulgeGeometry
from underPINN.utils.pulsatile_time_march import (
    cosine_squared_inlet_factory,
    make_model_from_cfg,
    parabolic_steady_uvw_factory,
    run_pulsatile_time_march,
)


def run_AAA_pulsatile_transfer(cfg) -> dict:
    ph = cfg.physics
    Re       = float(cfg_get(ph, "Re",       default=40.0))
    R_vessel = float(cfg_get(ph, "R_vessel", default=0.5))
    R_AAA    = float(cfg_get(ph, "R_AAA",    default=1.0))
    L        = float(cfg_get(ph, "L",        default=7.0))
    x_lo     = float(cfg_get(ph, "x_lo",     default=-3.5))
    x0       = float(cfg_get(ph, "x0",       default=-2.0))
    L_AAA    = float(cfg_get(ph, "L_AAA",    default=1.5))
    V_max    = float(cfg_get(ph, "V_max",    default=2.0))
    V_amp    = float(cfg_get(ph, "V_amp",    default=1.0))
    T_period = float(cfg_get(ph, "T_period", default=1.0))
    x_hi     = x_lo + L
    x_mid    = 0.5 * (x_lo + x_hi)

    print(f"AAA pulsatile (3-D unsteady):  Re={Re},  R_vessel={R_vessel},  "
          f"R_AAA={R_AAA},  x∈[{x_lo}, {x_hi}],  x0={x0},  L_AAA={L_AAA}")
    print(f"  Inlet peak(t) = {V_max} + {V_amp}·sin(2π t/{T_period})  "
          f"(parabolic profile)")

    geom = BulgeGeometry(R_vessel=R_vessel, R_AAA=R_AAA, L=L,
                         x_lo=x_lo, x0=x0, L_AAA=L_AAA)
    model, _ = make_model_from_cfg(cfg)
    pde = UnsteadyNS3DPDE(model, Re=Re)

    return run_pulsatile_time_march(
        cfg,
        problem_spec=dict(
            problem="AAA_pulsatile_transfer",
            label="AAA pulsatile flow (Newtonian)",
            geom=geom, pde=pde,
            inlet_target_fn=cosine_squared_inlet_factory(
                R_vessel, V_max, V_amp, T_period),
            steady_uvw_fn=parabolic_steady_uvw_factory(
                R_vessel, V_max, radius_fn=geom.radius_at),
            physics_dict={"Re": Re, "R_vessel": R_vessel, "R_AAA": R_AAA,
                          "L": L, "x_lo": x_lo, "x0": x0, "L_AAA": L_AAA,
                          "V_max": V_max, "V_amp": V_amp, "T_period": T_period},
            plot_extent=(x_lo, x_hi, x_mid),
        ),
        out_dir_default="outputs/AAA_pulsatile_transfer",
    )


if __name__ == "__main__":
    import sys
    import pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(
        pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
        else _HERE / "AAA_pulsatile_transfer.yaml"
    )
    from underPINN.config.loader import load_config
    run_AAA_pulsatile_transfer(load_config(cfg_path))
