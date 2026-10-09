"""Every example must train identically with ``training.data_parallel: true``.

For each example this runs the real ``run_*`` entry point twice -- flag off,
then flag on -- on a multi-device host, and requires the two loss trajectories
to agree to float32 reduction-order noise. Turning data parallelism on must
change where work runs, never what the optimiser does.

Identical trajectories prove nothing on their own: an example that silently
ignored the flag would also produce them. So every run also *counts* the
sharding it performs, and the test fails unless the flag-on run actually split
its batches across devices (and the flag-off run did not).

Needs >1 device -- run in its own process, as CI does::

    UNDERPINN_REQUIRE_MULTI_DEVICE=1 \\
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        python -m pytest tests/test_parallel_examples.py

Sizes in each case are chosen so every *effective* per-step batch divides by 8
(several loops draw ``min(batch_bc, N_inlet)``, so the pool size matters too).
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Optional

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax                                                        # noqa: E402
import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402

from underPINN.config.loader import load_config, merge_config      # noqa: E402
from underPINN.training import parallel as _par                    # noqa: E402

_N_DEV = jax.device_count()
if os.environ.get("UNDERPINN_REQUIRE_MULTI_DEVICE") == "1" and _N_DEV < 2:
    raise RuntimeError(
        f"UNDERPINN_REQUIRE_MULTI_DEVICE=1 but JAX sees {_N_DEV} device(s); "
        "XLA_FLAGS must be set in the environment of a fresh process.")

pytestmark = pytest.mark.skipif(
    _N_DEV < 2, reason=f"needs >=2 JAX devices, saw {_N_DEV}")

_EXAMPLES = pathlib.Path(__file__).parent.parent / "examples"


@dataclass
class Case:
    name: str
    script: str             # relative to examples/
    entry: str              # run_* function
    config: str             # relative to examples/
    overrides: dict = field(default_factory=dict)
    epochs: int = 4
    loss_key: str = "loss_hist"
    rtol: float = 2e-3
    atol: float = 1e-5
    # Mini-batch loops shard every epoch (default: >= epochs arrays). Full-batch
    # loops with a fixed point set correctly shard it *once* (plus once per
    # resample) -- re-sharding an unchanged array every step would copy it.
    min_shards: int | None = None
    # For examples that return summary losses instead of a loss history.
    extract: Optional[Callable[[dict], list]] = None
    # Called as setup(tmp_path, monkeypatch) before both runs (e.g. fixtures).
    setup: Optional[Callable] = None


# Pipe / AAA family: interior, wall, inlet, outlet pools; inlet/outlet draws are
# min(batch_bc, N), so pools are sized >= batch_bc to keep every draw at 16.
_PIPE_SIZES = {
    "data.n_interior": 128, "data.n_wall": 32,
    "data.n_inlet": 32, "data.n_outlet": 32,
    "training.batch_r": 64, "training.batch_bc": 16,
}

CASES = [
    Case("pipe_flow", "PINNs/pipe_flow/pipe_flow.py", "run_pipe_flow",
         "PINNs/pipe_flow/pipe_flow.yaml", dict(_PIPE_SIZES)),
    Case("AAA_flow", "PINNs/AAA/AAA_flow.py", "run_AAA_flow",
         "PINNs/AAA/config.yaml", dict(_PIPE_SIZES)),
    Case("pipe_flow_rheology",
         "PINNs/pipe_flow_rheology/pipe_flow_rheology.py",
         "run_pipe_flow_rheology",
         "PINNs/pipe_flow_rheology/config.yaml", dict(_PIPE_SIZES)),
    Case("AAA_rheology", "PINNs/AAA_rheology/AAA_rheology.py",
         "run_AAA_rheology", "PINNs/AAA_rheology/config.yaml",
         dict(_PIPE_SIZES)),
]

# Shock / Aneurysm family. rar_period=2 forces RAR/RAD resampling to run inside
# the 4-epoch test, so the residual-driven pool refresh is exercised with
# *replicated* parameters -- not just the plain training step.
_RAR = {"training.rar_period": 2, "training.rar_candidates": 2}

CASES += [
    Case("sod_shock", "PINNs/sod_shock/sod_shock.py", "run_sod_shock",
         "PINNs/sod_shock/config.yaml",
         {**_RAR, "data.n_interior": 128, "data.n_ic": 32, "data.n_bc": 32,
          "training.batch_r": 64, "training.batch_bc": 16}),
    Case("toro3", "PINNs/toro3/toro3.py", "run_toro3",
         "PINNs/toro3/config.yaml",
         {**_RAR, "data.n_interior": 128, "data.n_ic": 32, "data.n_bc": 32,
          "training.batch_r": 64, "training.batch_bc": 16}),
    Case("ramp", "PINNs/ramp/ramp.py", "run_ramp", "PINNs/ramp/config.yaml",
         {**_RAR, "training.early_stopping_patience": 9999,
          "data.n_interior": 128, "data.n_inlet": 32, "data.n_wall": 32,
          "data.n_upper": 32, "training.batch_r": 64,
          "training.batch_bc": 16}),
    # batch_r=80 with the shipped adapt_frac=0.4 / bl_frac=0.3 gives interior
    # sub-batches of 32 / 24 / 24 -- each divisible by 8 on its own.
    Case("ramp_ns", "PINNs/ramp_ns/ramp_ns.py", "run_ramp_ns",
         "PINNs/ramp_ns/config.yaml",
         {**_RAR, "data.n_uniform": 128, "data.n_bl": 128, "data.n_adapt": 128,
          "data.n_inlet": 32, "data.n_wall": 32, "data.n_slip": 32,
          "data.n_upper": 32, "training.batch_r": 80,
          "training.batch_bc": 16}),
    Case("Aneurysm", "PINNs/Aneurysm/Aneurysm.py", "run_Aneurysm",
         "PINNs/Aneurysm/config.yaml",
         {**_RAR, "data.n_interior": 128, "data.n_wall": 32,
          "data.n_inlet": 32, "data.n_outlet": 32,
          "integral_plane.n_points": 50,       # closed over, not batched
          "training.batch_r": 64, "training.batch_bc": 16}),
]

# Remaining hand-written loops.
_FULL_BATCH_FLOW = {
    "training.resample_period": 2,          # exercise re-shard after rebuild
    "data.n_interior": 80, "data.n_wake": 16,   # 96 collocation points
    "data.n_inlet": 20, "data.n_outlet": 20, "data.n_body": 20,
    "data.n_wall": 20,                     # closed-over BC sets: any size
}
_HEAT_INV = {"data.n_collocation": 128, "data.n_ic": 20, "data.n_bc": 20,
             "data.n_observations": 10}

CASES += [
    Case("wave", "PINNs/wave/wave.py", "run_wave", "PINNs/wave/config.yaml",
         {"training.early_stopping_patience": 9999,
          "data.n_collocation": 128, "data.n_ic": 32, "data.n_bc": 32,
          "training.batch_r": 64, "training.batch_i": 16,
          "training.batch_b": 16}),
    Case("airfoil", "PINNs/airfoil/airfoil_flow.py", "run_airfoil",
         "PINNs/airfoil/config.yaml", dict(_FULL_BATCH_FLOW), min_shards=2),
    Case("cylinder", "PINNs/cylinder/cylinder_flow.py", "run_cylinder",
         "PINNs/cylinder/config.yaml", dict(_FULL_BATCH_FLOW), min_shards=2),
    Case("fbpinn_ode", "PINNs/fbpinn_ode/fbpinn_ode.py", "run_fbpinn_ode",
         "PINNs/fbpinn_ode/config.yaml",
         {"data.n_collocation": 64, "subdomains.n": 4}, min_shards=1),
    Case("heat_inverse", "PINNs/heat/inverse.py", "run_heat_inverse",
         "PINNs/heat/heat_inverse.yaml", dict(_HEAT_INV), min_shards=2),
    Case("inverse_diffusion", "PINNs/inverse/inverse_diffusion.py",
         "run_inverse_diffusion", "PINNs/inverse/config.yaml",
         dict(_HEAT_INV), min_shards=2),
]

# Pulsatile time-marching examples sharing underPINN.utils.pulsatile_time_march.
# Two windows (the second warm-starts from the first's *replicated* params) and
# the cold-start baseline chain, so every _train_window call path runs under
# data parallelism -- including per-window save_checkpoint of replicated params.
_PULS = {
    "time_marching.T_total": 1.0, "time_marching.dT": 0.5,
    "time_marching.n_first_epochs": 3, "time_marching.n_warm_epochs": 3,
    "time_marching.n_cold_epochs": 3,
    "time_marching.compare_no_transfer": True,
    "data.n_interior": 80, "data.n_wall": 20, "data.n_inlet": 20,
    "data.n_outlet": 20, "data.n_ic": 20,
    "data.batch_r": 32, "data.batch_bc": 8,
}

CASES += [
    Case("AAA_pulsatile", "PINNs/AAA/AAA_pulsatile_transfer.py",
         "run_AAA_pulsatile_transfer",
         "PINNs/AAA/AAA_pulsatile_transfer.yaml", dict(_PULS)),
    # The cold-start baseline chain is the same library code for all three
    # pulsatile examples; AAA_pulsatile above covers it, so these two skip it.
    Case("pipe_flow_rheology_pulsatile",
         "PINNs/pipe_flow_rheology/pipe_flow_rheology_pulsatile.py",
         "run_pipe_flow_rheology_pulsatile",
         "PINNs/pipe_flow_rheology/pipe_flow_rheology_pulsatile.yaml",
         {**_PULS, "time_marching.compare_no_transfer": False}),
    Case("AAA_rheology_pulsatile",
         "PINNs/AAA_rheology/AAA_rheology_pulsatile.py",
         "run_AAA_rheology_pulsatile",
         "PINNs/AAA_rheology/AAA_rheology_pulsatile.yaml",
         {**_PULS, "time_marching.compare_no_transfer": False}),
]


# Own-trainer transfer examples (not the shared pulsatile library).
def _unsteady_finals(r):
    """Final loss of every compared run: source->transfer and scratch, for
    both the Reynolds-number and temporal strategies."""
    return [r["re_transfer"]["transfer_final"], r["re_transfer"]["scratch_final"],
            r["temporal_transfer"]["transfer_final"],
            r["temporal_transfer"]["scratch_final"]]


CASES += [
    Case("pipe_flow_pulsatile_transfer",
         "PINNs/pipe_flow/pipe_flow_pulsatile_transfer.py",
         "run_pipe_flow_pulsatile_transfer",
         "PINNs/pipe_flow/pipe_flow_pulsatile_transfer.yaml",
         {**_PULS, "time_marching.stride": 0.5,
          "data.n_interior": 100, "data.batch_r": 64, "data.batch_bc": 16}),
    Case("pipe_flow_unsteady_transfer",
         "PINNs/pipe_flow/pipe_flow_unsteady_transfer.py",
         "run_pipe_flow_unsteady_transfer",
         "PINNs/pipe_flow/pipe_flow_unsteady_transfer.yaml",
         {"re_transfer.n_source_epochs": 3, "re_transfer.n_transfer_epochs": 3,
          "re_transfer.n_scratch_epochs": 3,
          "temporal_transfer.n_transfer_epochs": 3,
          "temporal_transfer.n_scratch_epochs": 3,
          "data.n_collocation": 128, "data.n_ic": 32, "data.n_bc": 32,
          "data.batch_r": 64, "data.batch_ic": 16, "data.batch_bc": 16},
         extract=_unsteady_finals),
]



# Solver-driven examples (FBPINN / Steady / LDC / ODE solvers).
def _ldc_reference_csv(tmp_path, monkeypatch):
    """run_ldc plots against re100.csv in the cwd; give it a dummy one (as
    tests/test_examples.py does) so the comparison is about training only."""
    pd = pytest.importorskip("pandas")
    n = 201
    pd.DataFrame({
        "x-coordinate": np.tile(np.linspace(0, 1, n), n).astype(np.float32),
        "y-coordinate": np.repeat(np.linspace(0, 1, n), n).astype(np.float32),
        "pressure": np.zeros(n * n, np.float32),
        "x-velocity": np.zeros(n * n, np.float32),
        "y-velocity": np.zeros(n * n, np.float32),
    }).to_csv(tmp_path / "re100.csv", index=False)
    monkeypatch.chdir(tmp_path)


def _transfer_finals(key_a, key_b):
    def f(r):
        return [r[key_a]["transfer_final"], r[key_a]["scratch_final"],
                r[key_b]["transfer_final"], r[key_b]["scratch_final"]]
    return f


_STEADY = {"data.n_collocation": 128, "data.n_bc": 32,
           "training.batch_r": 64, "training.batch_b": 16}

CASES += [
    Case("burgers", "PINNs/burgers/burgers.py", "run_burgers",
         "PINNs/burgers/config.yaml",
         {"data.n_collocation": 128, "data.n_ic": 32, "data.n_bc": 32,
          "training.batch_r": 64, "training.batch_i": 16,
          "training.batch_b": 16}),
    Case("burgers_transfer", "PINNs/transfer/burgers_transfer.py",
         "run_burgers_transfer", "PINNs/transfer/burgers_transfer.yaml",
         {"parameter_transfer.n_source_epochs": 3,
          "parameter_transfer.n_transfer_epochs": 3,
          "parameter_transfer.n_scratch_epochs": 3,
          "temporal_transfer.n_transfer_epochs": 3,
          "temporal_transfer.n_scratch_epochs": 3,
          "data.n_collocation": 128, "data.n_ic": 32, "data.n_bc": 32},
         extract=_transfer_finals("param_transfer", "temporal_transfer")),
    Case("heat_forward", "PINNs/heat/forward.py", "run_heat_forward",
         "PINNs/heat/heat_forward.yaml", {**_STEADY, "data.n_ic": 32}),
    Case("helmholtz", "PINNs/helmholtz/helmholtz.py", "run_helmholtz",
         "PINNs/helmholtz/config.yaml", dict(_STEADY)),
    # LDC walks n_collocation in batch_r steps: 128 / 64 -> 2 steps per epoch.
    Case("LDC", "PINNs/LDC/run_ldc.py", "run_ldc", "PINNs/LDC/config.yaml",
         {"data.n_collocation": 128, "data.n_bc": 40,
          "training.batch_r": 64},
         setup=_ldc_reference_csv),
    # Two ODE problems per run, each full-batch: one shard of t_r per problem.
    Case("ode", "PINNs/ode/ode_test.py", "run_ode", "PINNs/ode/config.yaml",
         {"training.early_stopping_patience": 9999,
          "data.n_collocation": 64}, min_shards=2,
         extract=lambda r: [r["exp_decay_rel_l2"], r["harmonic_rel_l2"]]),
]



# Neural-operator examples (OperatorSolver / DeepONetSolver). Each step's batch
# is of *samples*; batch_size is a multiple of 8 even when fewer training pairs
# exist (safe_choice then samples with replacement). Nt values are CFL-safe for
# the data-generating solvers, which refuse unstable time steps.
_OP = {"loss.pde_warmup_epochs": 1, "training.batch_size": 8}

CASES += [
    Case("deeponet1d_burgers",
         "operators/deeponet1d_burgers/deeponet1d_burgers.py",
         "run_deeponet1d_burgers",
         "operators/deeponet1d_burgers/config.yaml",
         {"training.batch_size": 16, "data.n_ic_pool": 64,
          "loss.n_r": 64, "loss.n_bc": 16}),
    Case("cvit2d_burgers", "operators/cvit2d_burgers/cvit2d_burgers.py",
         "run_cvit2d_burgers", "operators/cvit2d_burgers/config.yaml",
         {**_OP, "data.Nt": 40, "data.n_train": 8, "data.n_test": 4,
          "data.pairs_per_traj": 2}),
    Case("fno1d_dirichlet", "operators/fno1d_dirichlet/fno1d_dirichlet.py",
         "run_fno1d_dirichlet", "operators/fno1d_dirichlet/config.yaml",
         {**_OP, "data.Nt": 170, "data.n_train": 16, "data.n_test": 8,
          "data.pairs_per_traj": 2}),
    Case("fno1d_periodic", "operators/fno1d_periodic/fno1d_periodic.py",
         "run_fno1d_periodic", "operators/fno1d_periodic/config.yaml",
         {**_OP, "data.Nt": 60, "data.n_train": 16, "data.n_test": 8,
          "data.pairs_per_traj": 2}),
    # Same example through OperatorSolver's lax.scan path: the stacked
    # (n_scan, batch, ...) samples must be split on axis 1.
    Case("fno1d_periodic_scan", "operators/fno1d_periodic/fno1d_periodic.py",
         "run_fno1d_periodic", "operators/fno1d_periodic/config.yaml",
         {**_OP, "training.n_scan_steps": 2, "data.Nt": 60,
          "data.n_train": 16, "data.n_test": 8, "data.pairs_per_traj": 2},
         min_shards=2),
    Case("fno2d_burgers", "operators/fno2d_burgers/fno2d_burgers.py",
         "run_fno2d_burgers", "operators/fno2d_burgers/config.yaml",
         {**_OP, "data.Nt": 30, "data.n_train": 8, "data.n_test": 4,
          "data.pairs_per_traj": 2}),
    # The shipped config is T=180 on a 128^2 grid with Nt=18000. Shrink T and
    # the grid *together* so the step stays CFL-safe (U*dt/dx = 0.27 < 0.4);
    # 64^2 keeps the shipped modes1=modes2=24 below the grid's Nyquist limit.
    Case("fno2d_cylinder", "operators/fno2d_cylinder/fno2d_cylinder.py",
         "run_fno2d_cylinder", "operators/fno2d_cylinder/config.yaml",
         {**_OP, "data.T": 2.0, "data.Nx": 64, "data.Ny": 64, "data.Nt": 60,
          "data.n_train_re": 2, "data.n_test_re": 1,
          "data.pairs_per_traj": 2, "data.poisson_iters": 15}),
]

# NOTE: the test machinery below must stay *after* every CASES definition --
# parametrize snapshots the ids list at decoration time.

# ---------------------------------------------------------------------------
# Sharding spy: counts real (enabled) sharding, ignores inert no-op calls
# ---------------------------------------------------------------------------

class _Spy:
    def __init__(self):
        self.prepared = 0
        self.sharded_arrays = 0
        self.axes = set()


@pytest.fixture
def spy(monkeypatch):
    s = _Spy()
    orig_prepare = _par.DataParallel.prepare
    orig_shard = _par.DataParallel.shard

    def prepare(self, params, state, sizes):
        if self.enabled:
            s.prepared += 1
        return orig_prepare(self, params, state, sizes)

    def shard(self, array, axis=0):
        if self.enabled:
            s.sharded_arrays += 1
            s.axes.add(axis)
        return orig_shard(self, array, axis)

    monkeypatch.setattr(_par.DataParallel, "prepare", prepare)
    monkeypatch.setattr(_par.DataParallel, "shard", shard)
    return s


def _run(case: Case, data_parallel: bool, out_dir: pathlib.Path):
    path = _EXAMPLES / case.script
    spec = importlib.util.spec_from_file_location(
        f"_dp_{case.name}_{int(data_parallel)}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = load_config(str(_EXAMPLES / case.config))
    cfg = merge_config(cfg, {
        **case.overrides,
        "training.epochs": case.epochs,
        "training.save_restart_every": 0,   # never resume the other run
        "training.data_parallel": data_parallel,
        "output.dir": str(out_dir),
    })
    result = getattr(mod, case.entry)(cfg)
    vals = case.extract(result) if case.extract else result[case.loss_key]
    return np.asarray(vals, dtype=np.float64)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_example_trains_identically_with_data_parallel(case, tmp_path, spy,
                                                        monkeypatch):
    if case.setup is not None:
        case.setup(tmp_path, monkeypatch)
    ref = _run(case, False, tmp_path / "ref")
    assert spy.prepared == 0 and spy.sharded_arrays == 0, (
        "flag-off run sharded something; the inert path is not inert")

    par = _run(case, True, tmp_path / "par")
    assert spy.prepared >= 1, (
        f"{case.name}: data_parallel=True never called dp.prepare() -- the "
        f"flag is being ignored, so params were never replicated")
    need = case.epochs if case.min_shards is None else case.min_shards
    assert spy.sharded_arrays >= need, (
        f"{case.name}: only {spy.sharded_arrays} arrays sharded (expected >= "
        f"{need}) -- batches are not being split")

    assert ref.shape == par.shape and ref.size >= (
        1 if case.extract else case.epochs)
    assert np.all(np.isfinite(par)), f"{case.name}: non-finite loss with DP"
    np.testing.assert_allclose(
        par, ref, rtol=case.rtol, atol=case.atol,
        err_msg=f"{case.name}: loss trajectory changed when data parallelism "
                f"was turned on")


# ---------------------------------------------------------------------------
# heat2d_transfer: a standalone script (hard-coded 2.5k-4.5k epochs, no YAML),
# so its run_training() -- where the wiring lives -- is tested directly.
# ---------------------------------------------------------------------------

def _heat2d():
    path = _EXAMPLES / "PINNs/transfer/heat2d_transfer.py"
    spec = importlib.util.spec_from_file_location("_dp_heat2d", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_heat2d_transfer_run_training_identical_with_data_parallel(spy):
    mod = _heat2d()
    from underPINN.nn.mlp import MLP
    from underPINN.pde.heat2d_unsteady import UnsteadyHeat2DPDE

    def train(on):
        model = MLP(layers=[3, 16, 16, 1])
        pde = UnsteadyHeat2DPDE(model, alpha=0.1)
        data = mod.make_data(0.5, N_r=128, N_ic=32, N_per_edge=8, seed=0)
        dp = _par.make_data_parallel(active=on)
        _p, loss, _e = mod.run_training(model, pde, data, 4, 1e-3, 0.1,
                                        seed=0, dp=dp)
        return np.asarray(loss)

    ref = train(False)
    assert spy.sharded_arrays == 0
    par = train(True)
    assert spy.prepared >= 1 and spy.sharded_arrays >= 4
    np.testing.assert_allclose(par, ref, rtol=2e-3, atol=1e-5)


def test_heat2d_transfer_orchestrators_route_through_dp():
    """Both strategies run 3 trainings each; all six must receive `dp`."""
    import inspect
    mod = _heat2d()
    for fn in (mod.run_parameter_transfer, mod.run_temporal_transfer):
        src = inspect.getsource(fn)
        assert "make_data_parallel(active=data_parallel)" in src
        assert src.count("= _train(") == 3, fn.__name__
        assert "= run_training(" not in src, (
            f"{fn.__name__} calls run_training directly, bypassing dp")
