"""2-D Ramp NS (viscous SBLI): underPINN (JAX) vs. eager PyTorch, matched to
``RampNSEvaluator`` (``underPINN/benchmark_utils/evaluators.py``).

Same network ([2,128,128,128,128,128,4], tanh), same fixed interior pool size
(40,000 uniform + 5,000 boundary-layer-clustered = 45,000, minibatched at
1536/step) and boundary counts (200 inlet / 200 no-slip wall / 100 slip wall /
150 upper), same loss weighting (1/100/100/80/20), same conservative viscous
NS residual with Ducros-sensor artificial viscosity (art_visc=2e-3), same
Adam(1e-3) with cosine decay. Geometry (``underPINN.geometry.ramp.RampGeometry``)
is plain NumPy, so both frameworks draw from literally the same sampler.

Deliberate simplification, disclosed rather than hidden: the evaluator's
RAR-D-migrated *adaptive* sub-pool is a NumPy-only, non-jittable operation
(the paper's own Appendix~\\ref{app:framework-detail} already documents this
as the reason ``jax.lax.scan`` cannot fuse it in), so a fair three-way
eager/jit/scan comparison uses this evaluator's STATIC collocation only
(uniform + boundary-layer pools, no mid-training resampling) -- identical
per-step computational structure to the production run, just without the
periodic-resampling side channel that none of the three variants could
include on equal footing.

Three variants, matching ``pipe_flow_baselines.py``'s split:
  torch_eager   eager PyTorch, torch.func (jacrev/hessian/vmap) formulation.
  jax_jit       underPINN's default: one jax.jit step per epoch.
  jax_scan      the fully-fused jax.lax.scan path.

Run:
    python benchmarks/suite/baselines/ramp_ns_baselines.py --epochs 5000
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

M_INF, THETA_DEG, GAMMA, RE, PR = 3.0, 15.0, 1.4, 1.0e4, 0.72
ART_VISC, AV_S = 2e-3, 0.05
L, H, RAMP_START, SLIP_END = 2.0, 1.0, 0.8, 0.15
LAYERS = [2, 128, 128, 128, 128, 128, 4]
LR = 1e-3
N_UNI, N_BL = 40000, 5000
N_R = N_UNI + N_BL
N_IN, N_W, N_SLIP, N_UP = 200, 200, 100, 150
B_R, B_IN, B_W, B_SLIP, B_UP = 1536, 200, 200, 100, 150
W_PDE, W_INLET, W_WALL, W_SLIP, W_UPPER = 1.0, 100.0, 100.0, 80.0, 20.0
EPS = 1e-6

T0 = 1.0 + 0.5 * (GAMMA - 1.0) * M_INF ** 2
RHO_INF, U_INF, V_INF, T_INF = 1.0, 1.0, 0.0, 1.0


def make_data(seed: int):
    from underPINN.geometry.ramp import RampGeometry
    geom = RampGeometry(THETA_DEG, L=L, H=H,
                        ramp_start=RAMP_START, slip_end=SLIP_END)
    xy_uniform = geom.sample_interior(N_UNI, seed=seed)
    xy_bl      = geom.sample_boundary_layer(N_BL, beta=4.0, seed=seed + 7)
    xy_r    = np.concatenate([xy_uniform, xy_bl], axis=0).astype("f4")
    xy_in   = geom.sample_inlet(N_IN).astype("f4")
    xy_w    = geom.sample_noslip_wall(N_W).astype("f4")
    xy_slip = geom.sample_slip_wall(N_SLIP).astype("f4")
    xy_up   = geom.sample_upper(N_UP).astype("f4")
    return xy_r, xy_in, xy_w, xy_slip, xy_up


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch  (eager, torch.func jacrev/vmap, matching the nested-jacfwd structure
# of CompressibleNS2DPDE.residual: one Jacobian for the flux divergence, a
# second, independent Jacobian-of-Jacobian for the artificial-viscosity
# Laplacian of the conserved variables.)
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

        def forward(self, xy: "torch.Tensor") -> "torch.Tensor":
            h = xy
            for i, lin in enumerate(self.lins):
                h = lin(h)
                if i < len(self.lins) - 1:
                    h = torch.tanh(h)
            return h

    return Net().to(device)


def run_torch_func(epochs: int, seed: int, device, data):
    import torch
    import torch.nn.functional as F
    from torch.func import functional_call, jacrev, vmap

    xy_r, xy_in, xy_w, xy_slip, xy_up = data
    net = _torch_net(device, seed)

    def prim_single(p, xy_i):
        raw = functional_call(net, p, (xy_i.unsqueeze(0),))[0]     # (4,)
        rho = F.softplus(raw[0]) + EPS
        u, v = raw[1], raw[2]
        T = F.softplus(raw[3]) + EPS
        return torch.stack([rho, u, v, T])

    def cons_single(p, xy_i):
        rho, u, v, T = prim_single(p, xy_i)
        pr = rho * T / (GAMMA * M_INF ** 2)
        E = pr / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
        return torch.stack([rho, rho * u, rho * v, E])

    def flux_pair_single(p, xy_i):
        rho, u, v, T = prim_single(p, xy_i)
        pr = rho * T / (GAMMA * M_INF ** 2)
        E = pr / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
        Fc = torch.stack([rho * u, rho * u * u + pr, rho * u * v, (E + pr) * u])
        Gc = torch.stack([rho * v, rho * u * v, rho * v * v + pr, (E + pr) * v])

        Jp = jacrev(prim_single, argnums=1)(p, xy_i)               # (4, 2)
        u_x, u_y = Jp[1, 0], Jp[1, 1]
        v_x, v_y = Jp[2, 0], Jp[2, 1]
        T_x, T_y = Jp[3, 0], Jp[3, 1]
        mu = torch.ones_like(rho)                                   # mu_law="constant"
        txx = (2.0 * mu / 3.0) * (2.0 * u_x - v_y)
        tyy = (2.0 * mu / 3.0) * (2.0 * v_y - u_x)
        txy = mu * (u_y + v_x)
        kap = mu / ((GAMMA - 1.0) * PR * M_INF ** 2)
        Fv = torch.stack([torch.zeros_like(rho), txx, txy,
                          u * txx + v * txy + kap * T_x])
        Gv = torch.stack([torch.zeros_like(rho), txy, tyy,
                          u * txy + v * tyy + kap * T_y])
        F_tot = Fc - Fv / RE
        G_tot = Gc - Gv / RE
        return torch.stack([F_tot, G_tot], dim=1)                   # (4, 2)

    def point_residual(p, xy_i):
        Hf = jacrev(flux_pair_single, argnums=1)(p, xy_i)          # (4, 2, 2)
        r = Hf[:, 0, 0] + Hf[:, 1, 1]                               # (4,)
        Hc = jacrev(jacrev(cons_single, argnums=1), argnums=1)(p, xy_i)  # (4,2,2)
        lap = Hc[:, 0, 0] + Hc[:, 1, 1]
        Jp = jacrev(prim_single, argnums=1)(p, xy_i)
        theta = Jp[1, 0] + Jp[2, 1]
        omega = Jp[2, 0] - Jp[1, 1]
        phi = theta ** 2 / (theta ** 2 + omega ** 2 + 1e-8)
        comp = 0.5 * (1.0 - torch.tanh(theta / AV_S))
        eps_local = ART_VISC * phi * comp
        return r - eps_local * lap

    residual_batch = vmap(point_residual, (None, 0))
    prim_batch = vmap(prim_single, (None, 0))

    def total_loss(p, XR, XI, XW, XS, XU):
        res = residual_batch(p, XR)                                 # (N, 4)
        pde_l = torch.mean(torch.sum(res ** 2, dim=-1))

        pv_in = prim_batch(p, XI)
        in_l = (torch.mean((pv_in[:, 0] - RHO_INF) ** 2)
                + torch.mean((pv_in[:, 1] - U_INF) ** 2)
                + torch.mean((pv_in[:, 2] - V_INF) ** 2)
                + torch.mean((pv_in[:, 3] - T_INF) ** 2))

        pv_w = prim_batch(p, XW)
        wall_l = (torch.mean(pv_w[:, 1] ** 2) + torch.mean(pv_w[:, 2] ** 2)
                  + torch.mean((pv_w[:, 3] - T0) ** 2))

        pv_s = prim_batch(p, XS)
        slip_l = torch.mean(pv_s[:, 2] ** 2)

        pv_u = prim_batch(p, XU)
        up_l = (torch.mean((pv_u[:, 0] - RHO_INF) ** 2)
                + torch.mean((pv_u[:, 1] - U_INF) ** 2)
                + torch.mean((pv_u[:, 2] - V_INF) ** 2)
                + torch.mean((pv_u[:, 3] - T_INF) ** 2))

        return (W_PDE * pde_l + W_INLET * in_l + W_WALL * wall_l
                + W_SLIP * slip_l + W_UPPER * up_l)

    tt = lambda a: torch.tensor(a, device=device)          # noqa: E731
    XR_full, XI_full = tt(xy_r), tt(xy_in)
    XW_full, XS_full, XU_full = tt(xy_w), tt(xy_slip), tt(xy_up)

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=LR * 1e-2)

    g = torch.Generator(device="cpu").manual_seed(seed + 11)

    def sample_batch():
        ir = torch.randint(0, N_R, (B_R,), generator=g)
        ii = torch.randint(0, N_IN, (B_IN,), generator=g)
        iw = torch.randint(0, N_W, (B_W,), generator=g)
        isl = torch.randint(0, N_SLIP, (B_SLIP,), generator=g)
        iu = torch.randint(0, N_UP, (B_UP,), generator=g)
        return (XR_full[ir.to(device)], XI_full[ii.to(device)],
                XW_full[iw.to(device)], XS_full[isl.to(device)],
                XU_full[iu.to(device)])

    def one_epoch():
        XR, XI, XW, XS, XU = sample_batch()
        opt.zero_grad(set_to_none=True)
        loss = total_loss(dict(net.named_parameters()), XR, XI, XW, XS, XU)
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
            "formulation": "torch.func jacrev/vmap (nested, matching "
                           "CompressibleNS2DPDE.residual's structure)"}


# ══════════════════════════════════════════════════════════════════════════════
# JAX  (jax.jit per epoch / jax.lax.scan fused) -- reuses CompressibleNS2DPDE
# directly, so the JAX side is bit-for-bit the production physics code.
# ══════════════════════════════════════════════════════════════════════════════

def _jax_pieces(epochs: int, seed: int, data):
    import jax
    import jax.numpy as jnp
    import optax
    from underPINN.nn.mlp import MLP
    from underPINN.pde.compressible_ns_2d import CompressibleNS2DPDE

    xy_r, xy_in, xy_w, xy_slip, xy_up = data
    model = MLP(layers=LAYERS)
    pde = CompressibleNS2DPDE(model, gamma=GAMMA, M_inf=M_INF, Re=RE, Pr=PR,
                              art_visc=ART_VISC)
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 2)))

    XR_full, XI_full = jnp.array(xy_r), jnp.array(xy_in)
    XW_full, XS_full, XU_full = jnp.array(xy_w), jnp.array(xy_slip), jnp.array(xy_up)

    def loss_fn(p, XR, XI, XW, XS, XU):
        res = pde.residual(p, XR)
        pde_l = jnp.mean(jnp.sum(res ** 2, axis=-1))
        pv_in = pde.apply(p, XI)
        in_l = (jnp.mean((pv_in[:, 0] - RHO_INF) ** 2)
                + jnp.mean((pv_in[:, 1] - U_INF) ** 2)
                + jnp.mean((pv_in[:, 2] - V_INF) ** 2)
                + jnp.mean((pv_in[:, 3] - T_INF) ** 2))
        pv_w = pde.apply(p, XW)
        wall_l = (jnp.mean(pv_w[:, 1] ** 2) + jnp.mean(pv_w[:, 2] ** 2)
                  + jnp.mean((pv_w[:, 3] - T0) ** 2))
        pv_s = pde.apply(p, XS)
        slip_l = jnp.mean(pv_s[:, 2] ** 2)
        pv_u = pde.apply(p, XU)
        up_l = (jnp.mean((pv_u[:, 0] - RHO_INF) ** 2)
                + jnp.mean((pv_u[:, 1] - U_INF) ** 2)
                + jnp.mean((pv_u[:, 2] - V_INF) ** 2)
                + jnp.mean((pv_u[:, 3] - T_INF) ** 2))
        return (W_PDE * pde_l + W_INLET * in_l + W_WALL * wall_l
                + W_SLIP * slip_l + W_UPPER * up_l)

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=1e-2)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))
    return (jax, jnp, optax, loss_fn, params0, opt,
            XR_full, XI_full, XW_full, XS_full, XU_full)


def run_jax_jit(epochs: int, seed: int, data):
    (jax, jnp, optax, loss_fn, params0, opt,
     XR_full, XI_full, XW_full, XS_full, XU_full) = _jax_pieces(epochs, seed, data)

    @jax.jit
    def step(p, s, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        ir = jax.random.randint(k1, (B_R,), 0, N_R)
        ii = jax.random.randint(k2, (B_IN,), 0, N_IN)
        iw = jax.random.randint(k3, (B_W,), 0, N_W)
        isl = jax.random.randint(k4, (B_SLIP,), 0, N_SLIP)
        iu = jax.random.randint(k5, (B_UP,), 0, N_UP)
        loss, g = jax.value_and_grad(loss_fn)(
            p, XR_full[ir], XI_full[ii], XW_full[iw], XS_full[isl], XU_full[iu])
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
     XR_full, XI_full, XW_full, XS_full, XU_full) = _jax_pieces(epochs, seed, data)

    def body(carry, key):
        p, s = carry
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        ir = jax.random.randint(k1, (B_R,), 0, N_R)
        ii = jax.random.randint(k2, (B_IN,), 0, N_IN)
        iw = jax.random.randint(k3, (B_W,), 0, N_W)
        isl = jax.random.randint(k4, (B_SLIP,), 0, N_SLIP)
        iu = jax.random.randint(k5, (B_UP,), 0, N_UP)
        loss, g = jax.value_and_grad(loss_fn)(
            p, XR_full[ir], XI_full[ii], XW_full[iw], XS_full[isl], XU_full[iu])
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
    ap = base_parser("2-D Ramp NS: JAX vs eager PyTorch (torch.func), static collocation")
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

    save_result(f"baselines_ramp_ns_seed{args.seed}", {
        "problem": "ramp_ns_sbli", "epochs": args.epochs, "seed": args.seed,
        "devices": devices, "variants": results,
        "note": "static collocation (uniform+BL pools only, no RAR-D resampling) "
                "for a fair eager/jit/scan comparison -- see module docstring.",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
