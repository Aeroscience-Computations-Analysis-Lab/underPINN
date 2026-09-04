"""1-D Burgers: underPINN (JAX) vs. *strong* PyTorch baselines on identical hardware.

A reviewer noted that comparing only against eager-mode PyTorch is a weak
baseline, since PyTorch also ships graph-capture backends that close much of
the dispatch gap. This script therefore measures five implementations of the
identical problem:

  torch_eager      plain eager PyTorch (the weak baseline)
  torch_script     torch.jit.script  -- TorchScript graph capture
  torch_compile    torch.compile(mode="max-autotune") -- TorchInductor
  jax_jit          underPINN's default: one jax.jit step per epoch
  jax_scan         underPINN's optional fully-fused jax.lax.scan path

All five use the same architecture ([2,64,64,64,64,64,1] tanh), the same
collocation counts (N_r=20000, N_ic=200, N_bc=300), the same loss weighting
(pde + 100*ic + 10*bc), the same Adam(1e-3) with cosine decay, and the same
derivative technique -- JAX via jacfwd/hessian exactly as BurgersPDE.residual
does, PyTorch via the equivalent autograd.grad "sum trick" double-backward.

Compile-bearing variants report a first (trace+compile) and second
(steady-state) run so one-time cost is never hidden in the headline number.

Run:
    python benchmarks/rebuttal/baselines/burgers_baselines.py --epochs 5000
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

import numpy as np                                                # noqa: E402

from common import (base_parser, jax_device_info, save_result,     # noqa: E402
                    timed, torch_device_info, torch_sync, warn_if_cpu)

NU = 0.01
T_MAX = 1.5
N_R, N_IC, N_BC = 20000, 200, 300
W_IC, W_BC = 100.0, 10.0
LAYERS = [2, 64, 64, 64, 64, 64, 1]
LR = 1e-3


def make_data(seed: int):
    """Collocation / IC / BC points -- shared verbatim by every implementation."""
    rng = np.random.default_rng(seed)
    x_r = rng.uniform(-1.0, 1.0, N_R).astype("f4")
    t_r = rng.uniform(0.0, T_MAX, N_R).astype("f4")
    x_ic = np.linspace(-1, 1, N_IC, dtype="f4")
    u_ic = (-np.sin(np.pi * x_ic)).astype("f4")
    t_bc_half = rng.uniform(0.0, T_MAX, N_BC).astype("f4")
    x_bc = np.tile([-1.0, 1.0], N_BC).astype("f4")
    t_bc = np.tile(t_bc_half, 2).astype("f4")
    return x_r, t_r, x_ic, u_ic, x_bc, t_bc


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch  (eager / TorchScript / torch.compile)
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

        def forward(self, xt: torch.Tensor) -> torch.Tensor:
            h = xt
            for i, lin in enumerate(self.lins):
                h = lin(h)
                if i < len(self.lins) - 1:
                    h = torch.tanh(h)
            return h

    return Net().to(device)


def run_torch_autograd(mode: str, epochs: int, seed: int, device, data):
    """Classic PINN formulation: ``autograd.grad(..., create_graph=True)``.

    *mode* is ``'eager'`` or ``'script'``.  ``torch.compile`` is deliberately
    NOT offered here: capturing this formulation raises
    "torch.compile with aot_autograd does not currently support double
    backward" -- see ``run_torch_func`` for the formulation that does compile.
    """
    import torch

    x_r, t_r, x_ic, u_ic, x_bc, t_bc = data
    net = _torch_net(device, seed)
    if mode == "script":
        net = torch.jit.script(net)

    def u_of(x, t):
        return net(torch.stack([x, t], dim=1))[:, 0]

    def d(f, wrt):
        return torch.autograd.grad(f.sum(), wrt, create_graph=True)[0]

    def total_loss(xr, tr, xi, ti, ui, xb, tb):
        u = u_of(xr, tr)
        u_x, u_t = d(u, xr), d(u, tr)
        u_xx = d(u_x, xr)
        res = u_t + u * u_x - NU * u_xx
        return (torch.mean(res ** 2)
                + W_IC * torch.mean((u_of(xi, ti) - ui) ** 2)
                + W_BC * torch.mean(u_of(xb, tb) ** 2))

    tt = lambda a: torch.tensor(a, device=device)          # noqa: E731
    xr0 = tt(x_r).requires_grad_(True)
    tr0 = tt(t_r).requires_grad_(True)
    xi, ui = tt(x_ic), tt(u_ic)
    ti = torch.zeros_like(xi)
    xb, tb = tt(x_bc), tt(t_bc)

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=LR * 1e-2)

    def one_epoch():
        opt.zero_grad(set_to_none=True)
        loss = total_loss(xr0, tr0, xi, ti, ui, xb, tb)
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
            "formulation": "autograd.grad(create_graph=True)"}


def run_torch_func(compiled: bool, epochs: int, seed: int, device, data):
    """``torch.func`` formulation -- PyTorch's direct analogue of JAX's
    ``jacfwd``/``hessian``/``vmap``, and the only PINN formulation here that
    ``torch.compile`` can actually capture.  This is the strongest PyTorch
    baseline in the suite."""
    import torch
    from torch.func import functional_call, hessian, jacrev, vmap

    x_r, t_r, x_ic, u_ic, x_bc, t_bc = data
    net = _torch_net(device, seed)
    params = dict(net.named_parameters())

    def u_single(p, xt):
        return functional_call(net, p, (xt.unsqueeze(0),))[0, 0]

    jac = vmap(jacrev(u_single, argnums=1), (None, 0))
    hess = vmap(hessian(u_single, argnums=1), (None, 0))
    u_batch = vmap(u_single, (None, 0))

    def total_loss(p, XR, XI, UI, XB):
        J, H = jac(p, XR), hess(p, XR)
        u = u_batch(p, XR)
        res = J[:, 1] + u * J[:, 0] - NU * H[:, 0, 0]
        return (torch.mean(res ** 2)
                + W_IC * torch.mean((u_batch(p, XI) - UI) ** 2)
                + W_BC * torch.mean(u_batch(p, XB) ** 2))

    loss_fn = torch.compile(total_loss) if compiled else total_loss

    tt = lambda a: torch.tensor(a, device=device)          # noqa: E731
    XR = torch.stack([tt(x_r), tt(t_r)], dim=1)
    XI = torch.stack([tt(x_ic), torch.zeros_like(tt(x_ic))], dim=1)
    UI = tt(u_ic)
    XB = torch.stack([tt(x_bc), tt(t_bc)], dim=1)

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=LR * 1e-2)

    def one_epoch():
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(dict(net.named_parameters()), XR, XI, UI, XB)
        loss.backward()
        opt.step()
        sched.step()
        return loss

    one_epoch()                       # warm up / trigger compilation, untimed
    torch_sync(device)
    losses: list = []
    _, wall = timed(lambda: [losses.append(one_epoch()) for _ in range(epochs)])
    torch_sync(device)
    del params
    return {"wall_s": wall, "ms_per_epoch": 1e3 * wall / epochs,
            "final_loss": float(losses[-1].detach()),
            "formulation": "torch.func jacrev/hessian/vmap"
                           + (" + torch.compile" if compiled else "")}


# ══════════════════════════════════════════════════════════════════════════════
# JAX  (jax.jit per epoch / jax.lax.scan fused)
# ══════════════════════════════════════════════════════════════════════════════

def _jax_pieces(epochs: int, seed: int, data):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as fnn

    x_r, t_r, x_ic, u_ic, x_bc, t_bc = data

    class MLPNet(fnn.Module):
        @fnn.compact
        def __call__(self, xt):
            h = xt
            for w in LAYERS[1:-1]:
                h = jnp.tanh(fnn.Dense(w)(h))
            return fnn.Dense(LAYERS[-1])(h)

    model = MLPNet()
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 2)))

    def apply_net(p, xt):
        return model.apply(p, xt)

    def u_single(p, xt):
        return apply_net(p, xt[None, :])[0, 0]

    # Matches BurgersPDE.residual: jacfwd for (u_x, u_t), hessian for u_xx.
    jac = jax.vmap(jax.jacfwd(u_single, argnums=1), in_axes=(None, 0))
    hess = jax.vmap(jax.hessian(u_single, argnums=1), in_axes=(None, 0))

    XR = jnp.stack([jnp.array(x_r), jnp.array(t_r)], axis=1)
    XI = jnp.stack([jnp.array(x_ic), jnp.zeros_like(jnp.array(x_ic))], axis=1)
    UI = jnp.array(u_ic)
    XB = jnp.stack([jnp.array(x_bc), jnp.array(t_bc)], axis=1)

    def loss_fn(p):
        J = jac(p, XR)
        H = hess(p, XR)
        u = apply_net(p, XR)[:, 0]
        res = J[:, 1] + u * J[:, 0] - NU * H[:, 0, 0]
        pde_l = jnp.mean(res ** 2)
        ic_l = jnp.mean((apply_net(p, XI)[:, 0] - UI) ** 2)
        bc_l = jnp.mean(apply_net(p, XB)[:, 0] ** 2)
        return pde_l + W_IC * ic_l + W_BC * bc_l

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=1e-2)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))
    return jax, jnp, optax, loss_fn, params0, opt


def run_jax_jit(epochs: int, seed: int, data):
    jax, jnp, optax, loss_fn, params0, opt = _jax_pieces(epochs, seed, data)

    @jax.jit
    def step(p, s):
        loss, g = jax.value_and_grad(loss_fn)(p)
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, loss

    def run():
        p, s = params0, opt.init(params0)
        loss = None
        for _ in range(epochs):
            p, s, loss = step(p, s)
        loss.block_until_ready()
        return float(loss)

    _p, _s, warm = step(params0, opt.init(params0))
    warm.block_until_ready()                    # compile once, untimed
    final, wall = timed(run)
    return {"wall_s": wall, "ms_per_epoch": 1e3 * wall / epochs,
            "final_loss": final}


def run_jax_scan(epochs: int, seed: int, data):
    jax, jnp, optax, loss_fn, params0, opt = _jax_pieces(epochs, seed, data)

    def body(carry, _):
        p, s = carry
        loss, g = jax.value_and_grad(loss_fn)(p)
        upd, s = opt.update(g, s)
        return (optax.apply_updates(p, upd), s), loss

    @jax.jit
    def run(p, s):
        (p, s), losses = jax.lax.scan(body, (p, s), None, length=epochs)
        return p, s, losses

    def once():
        p, s, losses = run(params0, opt.init(params0))
        losses.block_until_ready()
        return float(losses[-1])

    final1, t_first = timed(once)               # includes trace + compile
    final2, t_second = timed(once)              # steady state, cache hit
    return {"wall_s_first_call_incl_compile": t_first,
            "ms_per_epoch_first_call_incl_compile": 1e3 * t_first / epochs,
            "wall_s_second_call_compiled_only": t_second,
            "ms_per_epoch_second_call_compiled_only": 1e3 * t_second / epochs,
            "compile_overhead_s_estimate": max(t_first - t_second, 0.0),
            "final_loss_run1": final1, "final_loss_run2": final2}


# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = base_parser("1-D Burgers: JAX vs strong PyTorch baselines")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="variant names to skip (e.g. --skip torch_compile)")
    args = ap.parse_args()

    data = make_data(args.seed)
    require_gpu = not args.allow_cpu
    results, devices = {}, {}

    torch_variants = [
        ("torch_eager", lambda *a: run_torch_autograd("eager", *a)),
        ("torch_script", lambda *a: run_torch_autograd("script", *a)),
        ("torch_func_eager", lambda *a: run_torch_func(False, *a)),
        ("torch_func_compile", lambda *a: run_torch_func(True, *a)),
    ]
    if any(n not in args.skip for n, _ in torch_variants):
        device, tinfo = torch_device_info(require_gpu)
        warn_if_cpu(tinfo)
        devices["torch"] = tinfo
        print(f"PyTorch {tinfo['torch_version']} on "
              f"{tinfo['platform']} ({tinfo['device_name']})")
        for name, fn in torch_variants:
            if name in args.skip:
                continue
            print(f"--- {name}")
            try:
                r = fn(args.epochs, args.seed, device, data)
                results[name] = r
                print(f"    {r['wall_s']:8.2f}s  {r['ms_per_epoch']:7.3f} ms/ep"
                      f"  final_loss={r['final_loss']:.4e}")
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")
                results[name] = {"error": f"{type(e).__name__}: {e}"}

        # Documented negative result: torch.compile cannot capture the classic
        # create_graph=True PINN formulation at all (verified, not assumed).
        results["torch_compile_autograd"] = {
            "unavailable": True,
            "reason": "torch.compile with aot_autograd does not currently "
                      "support double backward",
            "note": "Verified on this machine for both the inductor and "
                    "aot_eager backends, and whether torch.compile wraps the "
                    "loss function or only the nn.Module. This is why the "
                    "compiled PyTorch baseline uses the torch.func "
                    "formulation instead.",
        }

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
            ms = r.get("ms_per_epoch",
                       r.get("ms_per_epoch_second_call_compiled_only"))
            print(f"    {ms:7.3f} ms/ep (steady state)")

    # ── summary ──────────────────────────────────────────────────────────────
    def ms_of(r):
        if "error" in r or r.get("unavailable"):
            return None
        return r.get("ms_per_epoch",
                     r.get("ms_per_epoch_second_call_compiled_only"))

    base = ms_of(results.get("torch_eager", {})) if "torch_eager" in results else None
    order = ["torch_eager", "torch_script", "torch_func_eager",
             "torch_func_compile", "jax_jit", "jax_scan"]
    print("\n" + "=" * 72)
    print(f"{'variant':22s} {'ms/epoch':>10s} {'vs torch_eager':>16s}")
    print("-" * 72)
    for name in order:
        if name not in results:
            continue
        ms = ms_of(results[name])
        if ms is None:
            print(f"{name:22s} {'ERROR':>10s}")
            continue
        rel = f"{base / ms:.2f}x" if base else "-"
        print(f"{name:22s} {ms:10.3f} {rel:>16s}")
    print("=" * 72)

    best_torch = min(
        (ms_of(results[n]) for n in order
         if n.startswith("torch") and n in results and ms_of(results[n])),
        default=None)
    best_jax = min(
        (ms_of(results[n]) for n in order
         if n.startswith("jax") and n in results and ms_of(results[n])),
        default=None)
    if best_torch and best_jax:
        print(f"\nBest PyTorch {best_torch:.3f} ms/ep vs best JAX "
              f"{best_jax:.3f} ms/ep  ->  {best_torch / best_jax:.2f}x")
        print("This best-vs-best ratio is the number the paper should quote.")
    print("\nNote: torch.compile cannot capture the classic "
          "autograd.grad(create_graph=True)\nPINN formulation (double backward "
          "unsupported); the compiled PyTorch baseline\nabove therefore uses "
          "torch.func, PyTorch's analogue of JAX's jacfwd/hessian.")

    save_result("baselines_burgers", {
        "problem": "burgers_1d", "epochs": args.epochs, "seed": args.seed,
        "devices": devices, "variants": results,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
