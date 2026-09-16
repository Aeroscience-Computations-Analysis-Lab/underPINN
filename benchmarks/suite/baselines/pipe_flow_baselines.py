"""3-D Pipe Flow: underPINN (JAX) vs. eager PyTorch, matched to
``PipeFlowEvaluator`` (``underPINN/benchmark_utils/evaluators.py``).

Same network ([3,128,128,128,128,128,4], tanh), same collocation counts
(40,000 interior / 600 wall / 200 inlet / 200 outlet, minibatched at the
evaluator's own batch sizes), same loss weighting (1/100/50/20), same
steady incompressible-NS residual (continuity + 3 momentum equations,
Re=10, full per-point Jacobian + Hessian via vmap), same Adam(1e-3) with
cosine decay. Geometry (``underPINN.geometry.pipe.Pipe``) is plain NumPy,
so both frameworks draw from literally the same sampler -- no risk of a
geometry mismatch between the two implementations.

Three variants, matching ``burgers_baselines.py``'s eager/jit/scan split:
  torch_eager   eager PyTorch, torch.func (jacrev/hessian/vmap) formulation
                -- PyTorch's structural analogue of JAX's jacfwd/hessian/vmap,
                run with no torch.compile (this is what makes it "eager").
  jax_jit       underPINN's default: one jax.jit step per epoch.
  jax_scan      the fully-fused jax.lax.scan path.

Run:
    python benchmarks/suite/baselines/pipe_flow_baselines.py --epochs 5000
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

import numpy as np                                                # noqa: E402

from common import (base_parser, save_result, timed,               # noqa: E402
                    torch_device_info, torch_sync, jax_device_info,
                    warn_if_cpu)

R, L, U_MAX, RE = 0.5, 2.0, 1.0, 10.0
NU = 1.0 / RE
LAYERS = [3, 128, 128, 128, 128, 128, 4]
LR = 1e-3
N_INT, N_WALL, N_IN, N_OUT = 40000, 600, 200, 200
B_INT, B_WALL, B_IN, B_OUT = 256, 128, 64, 64
W_PDE, W_WALL, W_IN, W_OUT = 1.0, 100.0, 50.0, 20.0


def make_data(seed: int):
    from underPINN.geometry.pipe import Pipe
    pipe = Pipe(R=R, L=L)
    xyz_int  = pipe.sample_interior(N_INT, seed=seed).astype("f4")
    xyz_wall = pipe.sample_wall(N_WALL,   seed=seed).astype("f4")
    xyz_in   = pipe.sample_inlet(N_IN,    seed=seed).astype("f4")
    xyz_out  = pipe.sample_outlet(N_OUT,  seed=seed).astype("f4")
    return xyz_int, xyz_wall, xyz_in, xyz_out


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch  (eager, torch.func jacrev/hessian/vmap)
# ══════════════════════════════════════════════════════════════════════════════

def _torch_net(device, seed: int):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lins = nn.ModuleList(
                [nn.Linear(LAYERS[i], LAYERS[i + 1])
                 for i in range(len(LAYERS) - 1)])

        def forward(self, xyz: "torch.Tensor") -> "torch.Tensor":
            h = xyz
            for i, lin in enumerate(self.lins):
                h = lin(h)
                if i < len(self.lins) - 1:
                    h = torch.tanh(h)
            return h

    return Net().to(device)


def run_torch_func(epochs: int, seed: int, device, data):
    import torch
    from torch.func import functional_call, hessian, jacrev, vmap

    xyz_int, xyz_wall, xyz_in, xyz_out = data
    net = _torch_net(device, seed)

    def uvwp_single(p, xyz_i):
        return functional_call(net, p, (xyz_i.unsqueeze(0),))[0]      # (4,)

    jac = vmap(jacrev(uvwp_single, argnums=1), (None, 0))              # (N,4,3)
    hess = vmap(hessian(uvwp_single, argnums=1), (None, 0))            # (N,4,3,3)
    uvwp_batch = vmap(uvwp_single, (None, 0))

    def residual(p, XYZ):
        uvwp = uvwp_batch(p, XYZ)                          # (N, 4)
        J = jac(p, XYZ)                                    # (N, 4, 3)
        H = hess(p, XYZ)                                   # (N, 4, 3, 3)
        u, v, w = uvwp[:, 0], uvwp[:, 1], uvwp[:, 2]
        u_x, u_y, u_z = J[:, 0, 0], J[:, 0, 1], J[:, 0, 2]
        v_x, v_y, v_z = J[:, 1, 0], J[:, 1, 1], J[:, 1, 2]
        w_x, w_y, w_z = J[:, 2, 0], J[:, 2, 1], J[:, 2, 2]
        p_x, p_y, p_z = J[:, 3, 0], J[:, 3, 1], J[:, 3, 2]
        lap_u = H[:, 0, 0, 0] + H[:, 0, 1, 1] + H[:, 0, 2, 2]
        lap_v = H[:, 1, 0, 0] + H[:, 1, 1, 1] + H[:, 1, 2, 2]
        lap_w = H[:, 2, 0, 0] + H[:, 2, 1, 1] + H[:, 2, 2, 2]
        cont  = u_x + v_y + w_z
        mom_x = u * u_x + v * u_y + w * u_z + p_x - NU * lap_u
        mom_y = u * v_x + v * v_y + w * v_z + p_y - NU * lap_v
        mom_z = u * w_x + v * w_y + w * w_z + p_z - NU * lap_w
        return cont, mom_x, mom_y, mom_z

    def total_loss(p, XR, XW, XI, XO):
        cont, mx, my, mz = residual(p, XR)
        pde_l = torch.mean(cont**2 + mx**2 + my**2 + mz**2)
        uvw_w = uvwp_batch(p, XW)
        wall_l = torch.mean(uvw_w[:, 0]**2 + uvw_w[:, 1]**2 + uvw_w[:, 2]**2)
        r_in = torch.sqrt(XI[:, 1]**2 + XI[:, 2]**2)
        u_ex = U_MAX * (1 - r_in**2 / R**2)
        uvw_i = uvwp_batch(p, XI)
        in_l = torch.mean((uvw_i[:, 0] - u_ex)**2 + uvw_i[:, 1]**2 + uvw_i[:, 2]**2)
        uvw_o = uvwp_batch(p, XO)
        out_l = torch.mean(uvw_o[:, 1]**2 + uvw_o[:, 2]**2)
        return W_PDE*pde_l + W_WALL*wall_l + W_IN*in_l + W_OUT*out_l

    tt = lambda a: torch.tensor(a, device=device)          # noqa: E731
    XR_full, XW_full = tt(xyz_int), tt(xyz_wall)
    XI_full, XO_full = tt(xyz_in), tt(xyz_out)

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=LR * 1e-2)

    g = torch.Generator(device="cpu").manual_seed(seed + 11)

    def sample_batch():
        ir = torch.randint(0, N_INT, (B_INT,), generator=g)
        iw = torch.randint(0, N_WALL, (B_WALL,), generator=g)
        ii = torch.randint(0, N_IN, (B_IN,), generator=g)
        io = torch.randint(0, N_OUT, (B_OUT,), generator=g)
        return (XR_full[ir.to(device)], XW_full[iw.to(device)],
                XI_full[ii.to(device)], XO_full[io.to(device)])

    def one_epoch():
        XR, XW, XI, XO = sample_batch()
        opt.zero_grad(set_to_none=True)
        loss = total_loss(dict(net.named_parameters()), XR, XW, XI, XO)
        loss.backward()
        opt.step()
        sched.step()
        return loss

    one_epoch()                       # warm up, untimed
    torch_sync(device)
    losses: list = []
    _, wall = timed(lambda: [losses.append(one_epoch()) for _ in range(epochs)])
    torch_sync(device)
    return {"wall_s": wall, "ms_per_epoch": 1e3 * wall / epochs,
            "final_loss": float(losses[-1].detach()),
            "formulation": "torch.func jacrev/hessian/vmap"}


# ══════════════════════════════════════════════════════════════════════════════
# JAX  (jax.jit per epoch / jax.lax.scan fused) -- reuses SteadyNS3DPDE directly
# ══════════════════════════════════════════════════════════════════════════════

def _jax_pieces(epochs: int, seed: int, data):
    import jax
    import jax.numpy as jnp
    import optax
    from underPINN.nn.mlp import MLP
    from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE

    xyz_int, xyz_wall, xyz_in, xyz_out = data
    model = MLP(layers=LAYERS)
    pde = SteadyNS3DPDE(model, Re=RE)
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 3)))

    XR_full = jnp.array(xyz_int)
    XW_full = jnp.array(xyz_wall)
    XI_full = jnp.array(xyz_in)
    XO_full = jnp.array(xyz_out)

    def loss_fn(p, XR, XW, XI, XO):
        res = pde.residual(p, XR)
        pde_l = jnp.mean(jnp.sum(res**2, axis=-1))
        u_w, v_w, w_w, _ = pde.uvwp(p, XW)
        wall_l = jnp.mean(u_w**2 + v_w**2 + w_w**2)
        r_in = jnp.sqrt(XI[:, 1]**2 + XI[:, 2]**2)
        u_ex = U_MAX * (1 - r_in**2 / R**2)
        u_in, v_in, w_in, _ = pde.uvwp(p, XI)
        in_l = jnp.mean((u_in - u_ex)**2 + v_in**2 + w_in**2)
        u_out, v_out, w_out, _ = pde.uvwp(p, XO)
        out_l = jnp.mean(v_out**2 + w_out**2)
        return W_PDE*pde_l + W_WALL*wall_l + W_IN*in_l + W_OUT*out_l

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=1e-2)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))
    return (jax, jnp, optax, loss_fn, params0, opt,
            XR_full, XW_full, XI_full, XO_full)


def run_jax_jit(epochs: int, seed: int, data):
    (jax, jnp, optax, loss_fn, params0, opt,
     XR_full, XW_full, XI_full, XO_full) = _jax_pieces(epochs, seed, data)

    @jax.jit
    def step(p, s, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        ir = jax.random.randint(k1, (B_INT,), 0, N_INT)
        iw = jax.random.randint(k2, (B_WALL,), 0, N_WALL)
        ii = jax.random.randint(k3, (B_IN,), 0, N_IN)
        io = jax.random.randint(k4, (B_OUT,), 0, N_OUT)
        loss, g = jax.value_and_grad(loss_fn)(
            p, XR_full[ir], XW_full[iw], XI_full[ii], XO_full[io])
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, loss

    def run():
        p, s = params0, opt.init(params0)
        key = jax.random.PRNGKey(seed + 11)
        loss = None
        for _ in range(epochs):
            key, k = jax.random.split(key)
            p, s, loss = step(p, s, k)
        loss.block_until_ready()
        return float(loss)

    key0 = jax.random.PRNGKey(seed + 11)
    _p, _s, warm = step(params0, opt.init(params0), key0)
    warm.block_until_ready()
    final, wall = timed(run)
    return {"wall_s": wall, "ms_per_epoch": 1e3 * wall / epochs,
            "final_loss": final}


def run_jax_scan(epochs: int, seed: int, data):
    (jax, jnp, optax, loss_fn, params0, opt,
     XR_full, XW_full, XI_full, XO_full) = _jax_pieces(epochs, seed, data)

    def body(carry, key):
        p, s = carry
        k1, k2, k3, k4 = jax.random.split(key, 4)
        ir = jax.random.randint(k1, (B_INT,), 0, N_INT)
        iw = jax.random.randint(k2, (B_WALL,), 0, N_WALL)
        ii = jax.random.randint(k3, (B_IN,), 0, N_IN)
        io = jax.random.randint(k4, (B_OUT,), 0, N_OUT)
        loss, g = jax.value_and_grad(loss_fn)(
            p, XR_full[ir], XW_full[iw], XI_full[ii], XO_full[io])
        upd, s = opt.update(g, s)
        return (optax.apply_updates(p, upd), s), loss

    @jax.jit
    def run(p, s, keys):
        (p, s), losses = jax.lax.scan(body, (p, s), keys)
        return p, s, losses

    keys0 = jax.random.split(jax.random.PRNGKey(seed + 11), epochs)

    def once():
        p, s, losses = run(params0, opt.init(params0), keys0)
        losses.block_until_ready()
        return float(losses[-1])

    final1, t_first = timed(once)
    final2, t_second = timed(once)
    return {"wall_s_first_call_incl_compile": t_first,
            "ms_per_epoch_first_call_incl_compile": 1e3 * t_first / epochs,
            "wall_s_second_call_compiled_only": t_second,
            "ms_per_epoch_second_call_compiled_only": 1e3 * t_second / epochs,
            "compile_overhead_s_estimate": max(t_first - t_second, 0.0),
            "final_loss_run1": final1, "final_loss_run2": final2}


# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = base_parser("3-D Pipe Flow: JAX vs eager PyTorch (torch.func)")
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()

    data = make_data(args.seed)
    require_gpu = not args.allow_cpu
    results, devices = {}, {}

    if "torch_eager" not in args.skip:
        device, tinfo = torch_device_info(require_gpu)
        warn_if_cpu(tinfo)
        devices["torch"] = tinfo
        print(f"PyTorch {tinfo['torch_version']} on {tinfo['platform']} "
              f"({tinfo['device_name']})")
        print("--- torch_eager")
        r = run_torch_func(args.epochs, args.seed, device, data)
        results["torch_eager"] = r
        print(f"    {r['wall_s']:8.2f}s  {r['ms_per_epoch']:7.3f} ms/ep  "
              f"final_loss={r['final_loss']:.4e}")

    jax_variants = [("jax_jit", run_jax_jit), ("jax_scan", run_jax_scan)]
    if any(n not in args.skip for n, _ in jax_variants):
        jinfo = jax_device_info(require_gpu)
        warn_if_cpu(jinfo)
        devices["jax"] = jinfo
        print(f"JAX on {jinfo['platform']} ({jinfo['device_name']})")
        for name, fn in jax_variants:
            if name in args.skip:
                continue
            print(f"--- {name}")
            r = fn(args.epochs, args.seed, data)
            results[name] = r
            ms = r.get("ms_per_epoch", r.get("ms_per_epoch_second_call_compiled_only"))
            print(f"    {ms:7.3f} ms/ep (steady state)")

    save_result(f"baselines_pipe_flow_seed{args.seed}", {
        "problem": "pipe_flow_3d", "epochs": args.epochs, "seed": args.seed,
        "devices": devices, "variants": results,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
