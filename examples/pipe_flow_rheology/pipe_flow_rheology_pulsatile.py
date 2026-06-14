"""3-D Unsteady Pulsatile Pipe Flow with Carreau (shear-thinning) Rheology.

Time-marching transfer learning, same as ``pipe_flow_pulsatile_transfer`` but
with the **Carreau** constitutive law (Nagargoje, Mishra & Gupta 2021):

    μ*(γ̇*) = 1 + (β − 1)[1 + (Cu γ̇*)²]^((n−1)/2)
    blood: β = μ0/μ∞ = 16,  n = 0.3568.  β = 1 → Newtonian.

Domain and Reynolds number are identical to the steady Carreau pipe case
(``examples/pipe_flow_rheology/config.yaml``).  Inlet imposes the developed
Carreau radial profile, modulated in time:

    u_inlet(r, t) = (V_max + V_amp · sin(2π t / T_period)) · u*_carr(r/R)

where u*_carr is the developed Carreau profile with centreline = 1.

Network: (x, y, z, t) → (u, v, w, p)
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from underPINN.config.loader import cfg_get
from underPINN.pde.carreau_ns_3d import UnsteadyCarreauNS3DPDE
from underPINN.geometry.pipe import Pipe
from underPINN.utils.pulsatile_time_march import (
    carreau_inlet_factory,
    carreau_steady_uvw_factory,
    make_model_from_cfg,
    run_pulsatile_time_march,
)


def run_pipe_flow_rheology_pulsatile(cfg) -> dict:
    ph = cfg.physics
    Re       = float(cfg_get(ph, "Re",       default=40.0))
    R        = float(cfg_get(ph, "R",        default=0.5))
    L        = float(cfg_get(ph, "L",        default=7.0))
    x_lo     = float(cfg_get(ph, "x_lo",     default=-3.5))
    V_max    = float(cfg_get(ph, "V_max",    default=2.0))
    V_amp    = float(cfg_get(ph, "V_amp",    default=1.0))
    T_period = float(cfg_get(ph, "T_period", default=1.0))
    beta     = float(cfg_get(ph, "beta",     default=16.0))
    Cu       = float(cfg_get(ph, "Cu",       default=10.0))
    n        = float(cfg_get(ph, "n",        default=0.3568))
    x_hi     = x_lo + L
    x_mid    = 0.5 * (x_lo + x_hi)

    print(f"Carreau pulsatile pipe (3-D unsteady):  Re={Re},  R={R}, "
          f"x∈[{x_lo}, {x_hi}]")
    print(f"  Carreau:  β={beta},  Cu={Cu},  n={n}")
    print(f"  Inlet peak(t) = {V_max} + {V_amp}·sin(2π t/{T_period})  "
          f"(Carreau-developed profile)")

    geom = Pipe(R=R, L=L, x_lo=x_lo)
    model, _ = make_model_from_cfg(cfg)
    pde = UnsteadyCarreauNS3DPDE(model, Re=Re, beta=beta, Cu=Cu, n=n)

    return run_pulsatile_time_march(
        cfg,
        problem_spec=dict(
            problem="pipe_flow_rheology_pulsatile",
            label="Carreau pulsatile pipe",
            geom=geom, pde=pde,
            inlet_target_fn=carreau_inlet_factory(
                R, V_max, V_amp, T_period, beta, Cu, n),
            steady_uvw_fn=carreau_steady_uvw_factory(R, V_max, beta, Cu, n),
            physics_dict={"Re": Re, "R": R, "L": L, "x_lo": x_lo,
                          "V_max": V_max, "V_amp": V_amp, "T_period": T_period,
                          "beta": beta, "Cu": Cu, "n": n},
            plot_extent=(x_lo, x_hi, x_mid),
        ),
        out_dir_default="outputs/pipe_flow_rheology_pulsatile",
    )


if __name__ == "__main__":
    import sys
    import pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(
        pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
        else _HERE / "pipe_flow_rheology_pulsatile.yaml"
    )
    from underPINN.config.loader import load_config
    run_pipe_flow_rheology_pulsatile(load_config(cfg_path))
