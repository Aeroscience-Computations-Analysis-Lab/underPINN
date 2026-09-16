"""Isolated worker process for the JAX-accelerated cylinder-flow solver.

``jax.config.update("jax_enable_x64", True)`` is a GLOBAL, process-wide JAX
setting. This flow's Hopf-bifurcation limit cycle amplifies small
perturbations by construction, so silently running the same algorithm in
JAX's float32 default (rather than matching the NumPy solver's float64)
could change which trajectory comes out, not just lose precision. Running
in a dedicated subprocess (with x64 enabled ONLY here, via env var, before
jax is ever imported) keeps that entirely separate from the main process
that later trains the FNO at float32 -- this whole example's accuracy work
has been validated at float32 throughout, and must not silently change.

Usage (called via subprocess.run from datagen.solve_cylinder_flow_jax_isolated,
not normally invoked directly):

    python3 run_datagen_jax_subprocess.py <params.json> <out.npz>

``params.json``: {"Re":.., "T":.., "Lx":.., "Ly":.., "Nx":.., "Ny":.., "Nt":..,
"cx":.., "cy":.., "r":.., "U_in":.., "seed":.., "poisson_iters":..}
Writes ``out.npz`` with keys U, V, P, mask (matching solve_cylinder_flow's
return format).
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"   # MUST be set before jax is imported

import json
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from datagen import solve_cylinder_flow_jax  # noqa: E402


def main():
    params_path, out_path = sys.argv[1], sys.argv[2]
    with open(params_path) as f:
        p = json.load(f)

    import jax
    assert jax.config.jax_enable_x64, "x64 must be enabled for this worker"

    U, V, P, mask = solve_cylinder_flow_jax(
        Re=p["Re"], T=p["T"], Lx=p["Lx"], Ly=p["Ly"],
        Nx=p["Nx"], Ny=p["Ny"], Nt=p["Nt"], cx=p["cx"], cy=p["cy"],
        r=p["r"], U_in=p["U_in"], seed=p["seed"],
        poisson_iters=p.get("poisson_iters", 80))

    np.savez(out_path, U=U, V=V, P=P, mask=mask)


if __name__ == "__main__":
    main()
