"""3-D Unsteady Pulsatile Flow through an Axisymmetric AAA with Carreau Blood.

Time-marching transfer learning combining the AAA bulge geometry with the
Carreau (shear-thinning blood) constitutive law:

    μ*(γ̇*) = 1 + (β − 1)[1 + (Cu γ̇*)²]^((n−1)/2)
    blood: β = 16,  n = 0.3568.

Domain and Reynolds number are identical to the steady AAA-rheology case
(``examples/AAA_rheology/config.yaml``).  Inlet imposes the developed Carreau
radial profile modulated in time:

    u_inlet(r, t) = (V_max + V_amp · sin(2π t / T_period)) · u*_carr(r/R_vessel)

Network: (x, y, z, t) → (u, v, w, p)
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from underPINN.config.loader import cfg_get
from underPINN.pde.carreau_ns_3d import UnsteadyCarreauNS3DPDE
from underPINN.geometry.aaa import BulgeGeometry
from underPINN.utils.pulsatile_time_march import (
    carreau_inlet_factory,
    carreau_steady_uvw_factory,
    make_model_from_cfg,
    run_pulsatile_time_march,
)


def run_AAA_rheology_pulsatile(cfg) -> dict:
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
    beta     = float(cfg_get(ph, "beta",     default=16.0))
    Cu       = float(cfg_get(ph, "Cu",       default=10.0))
    n        = float(cfg_get(ph, "n",        default=0.3568))
    x_hi     = x_lo + L
    x_mid    = 0.5 * (x_lo + x_hi)

    print(f"Carreau pulsatile AAA (3-D unsteady):  Re={Re},  R_vessel={R_vessel}, "
          f"R_AAA={R_AAA},  x∈[{x_lo}, {x_hi}],  x0={x0},  L_AAA={L_AAA}")
    print(f"  Carreau:  β={beta},  Cu={Cu},  n={n}")
    print(f"  Inlet peak(t) = {V_max} + {V_amp}·sin(2π t/{T_period})  "
          f"(Carreau-developed profile)")

    geom = BulgeGeometry(R_vessel=R_vessel, R_AAA=R_AAA, L=L,
                         x_lo=x_lo, x0=x0, L_AAA=L_AAA)
    model, _ = make_model_from_cfg(cfg)
    pde = UnsteadyCarreauNS3DPDE(model, Re=Re, beta=beta, Cu=Cu, n=n)

    return run_pulsatile_time_march(
        cfg,
        problem_spec=dict(
            problem="AAA_rheology_pulsatile",
            label="Carreau pulsatile AAA",
            geom=geom, pde=pde,
            inlet_target_fn=carreau_inlet_factory(
                R_vessel, V_max, V_amp, T_period, beta, Cu, n),
            steady_uvw_fn=carreau_steady_uvw_factory(
                R_vessel, V_max, beta, Cu, n, radius_fn=geom.radius_at),
            physics_dict={"Re": Re, "R_vessel": R_vessel, "R_AAA": R_AAA,
                          "L": L, "x_lo": x_lo, "x0": x0, "L_AAA": L_AAA,
                          "V_max": V_max, "V_amp": V_amp, "T_period": T_period,
                          "beta": beta, "Cu": Cu, "n": n},
            plot_extent=(x_lo, x_hi, x_mid),
        ),
        out_dir_default="outputs/AAA_rheology_pulsatile",
    )


if __name__ == "__main__":
    import sys
    import pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(
        pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
        else _HERE / "AAA_rheology_pulsatile.yaml"
    )
    from underPINN.config.loader import load_config
    run_AAA_rheology_pulsatile(load_config(cfg_path))
