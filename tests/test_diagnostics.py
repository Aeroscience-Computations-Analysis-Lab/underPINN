"""Tests of underPINN.diagnostics against exact identities and the numbers quoted in the
companion notes: "Representation and the Geometry of the PINN Loss" (spring oscillator,
pinn_math_notes), "Representation, Operator and the PINN Loss Landscape" (multiscale Poisson)
and "Post-processing a Two-Layer PINN".

Checked against the notes: feature conditioning kappa(G_Phi) (spring and Poisson), the
operator-mapped kappa(G_DPhi), the exact-solution constants, the Poisson mode weights.
Checked against exact identities: Lanczos/SLQ vs the full Hessian, kappa_10 and Jacobi scaling,
Gauss-Newton (exact and Lanczos paths), NTK vs the Jacobian, the output-layer refit and
refit-pruning vs brute-force least squares, the update ratio, spectra of known signals,
sine coefficients and the modal kernel of a model that is linear in its parameters."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.flatten_util import ravel_pytree

import flax.linen as nn

from underPINN import diagnostics as dg

M_, C_, K_, F0, OM, U0, V0, T = 1.0, 1.2, 40.0, 20.0, 4.0, 1.0, 0.0, 3.0
W0 = np.sqrt(K_ / M_)
ZETA = C_ / (2 * np.sqrt(K_ * M_))
WD = W0 * np.sqrt(1 - ZETA ** 2)


@pytest.fixture(scope="module", autouse=True)
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def features(kind, t, xp=np):
    """phi_A = (tau, ..., tau^5), phi_B = (tau, sin wd t, cos wd t, sin Om t, cos Om t), tau = 2t/T - 1."""
    tau = 2 * t / T - 1
    if kind == "A":
        return xp.stack([tau ** p for p in range(1, 6)], -1)
    return xp.stack([tau, xp.sin(WD * t), xp.cos(WD * t), xp.sin(OM * t), xp.cos(OM * t)], -1)


def exact_constants():
    """X, delta, C1, C2 of the exact solution (pinn_math_notes eqs. 3-4)."""
    X = (F0 / M_) / np.sqrt((W0 ** 2 - OM ** 2) ** 2 + (2 * ZETA * W0 * OM) ** 2)
    delta = np.arctan2(2 * ZETA * W0 * OM, W0 ** 2 - OM ** 2)
    C1 = U0 - X * np.cos(delta)
    C2 = (V0 - X * OM * np.sin(delta) + ZETA * W0 * C1) / WD
    return X, delta, C1, C2


def u_exact(t, xp=np):
    X, delta, C1, C2 = exact_constants()
    return xp.exp(-ZETA * W0 * t) * (C1 * xp.cos(WD * t) + C2 * xp.sin(WD * t)) + X * xp.cos(OM * t - delta)


class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        h = jnp.tanh(nn.Dense(32, param_dtype=jnp.float64)(x))
        return nn.Dense(1, param_dtype=jnp.float64)(h)


def make_spring(kind="B", n=256, seed=0):
    """The spring PINN of the notes: 5 -> 32 tanh -> 1 (P = 225) on standardised features."""
    t_grid = np.linspace(0, T, n)
    P = features(kind, t_grid)
    mu, sd = P.mean(0), P.std(0)
    model = Net()
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 5), jnp.float64))

    def std_features(t):
        return (features(kind, t, jnp) - mu) / sd

    def u(p, t):
        return model.apply(p, std_features(t)[None])[0, 0]
    u1 = jax.grad(u, 1)
    u2 = jax.grad(u1, 1)

    def resid(p, t):
        return (M_ * u2(p, t) + C_ * u1(p, t) + K_ * u(p, t) - F0 * jnp.cos(OM * t)) / K_

    def res_fn(p, x):
        return jax.vmap(lambda t: resid(p, t))(x[:, 0])[:, None]

    def ic(p):
        return u(p, 0.0) - U0, (u1(p, 0.0) - V0) / W0

    def terms(p, batch):
        a, b = ic(p)
        return {"residual": jnp.mean(res_fn(p, batch)[:, 0] ** 2), "ic": a ** 2 + b ** 2}

    def rho(p, batch):
        a, b = ic(p)
        return jnp.concatenate([res_fn(p, batch)[:, 0] / jnp.sqrt(batch.shape[0]), jnp.stack([a, b])])

    def apply_fn(p, x):
        return jax.vmap(lambda t: u(p, t))(x[:, 0])

    x = jnp.asarray(t_grid[:, None])
    hidden = dg.flax_hidden_fn(model, input_transform=lambda xx: std_features(xx[:, 0]))
    prob = dg.Problem(terms, x, residual_fn=res_fn, x_res=x, rho_fn=rho, apply_fn=apply_fn,
                      hidden_fn=hidden, name=f"spring_{kind}")
    return prob, params, model, (mu, sd)


def train(prob, params, steps=1500, lr=5e-3):
    opt = optax.adam(lr)
    st = opt.init(params)

    @jax.jit
    def step(p, s):
        loss, g = jax.value_and_grad(prob.loss)(p)
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, loss
    for _ in range(steps):
        params, st, _ = step(params, st)
    return params


@pytest.fixture(scope="module")
def spring_a(x64):
    prob, p0, model, ms = make_spring("A")
    return prob, p0, train(prob, p0, 300), model, ms


@pytest.fixture(scope="module")
def spring_b(x64):
    prob, p0, model, ms = make_spring("B")
    return prob, p0, train(prob, p0, 1500), model, ms


# --------------------------------------------------------------------------- the notes' numbers
def test_exact_solution_matches_notes():
    X, delta, C1, C2 = exact_constants()
    np.testing.assert_allclose([W0, ZETA, WD, ZETA * W0], [6.3246, 0.09487, 6.2960, 0.60], atol=1e-4)
    np.testing.assert_allclose([X, delta, C1, C2, np.hypot(C1, C2)], [0.8172, 0.1974, 0.1987, -0.0829, 0.2153],
                               atol=1e-4)
    u = lambda t: u_exact(t, jnp)                                            # noqa: E731
    u1, u2 = jax.grad(u), jax.grad(jax.grad(u))
    for t in (0.0, 0.7, 2.9):
        assert abs(M_ * u2(t) + C_ * u1(t) + K_ * u(t) - F0 * np.cos(OM * t)) < 1e-10
    assert abs(u(0.0) - U0) < 1e-12 and abs(u1(0.0) - V0) < 1e-12


def test_feature_gram_matches_notes():
    t = np.linspace(0, T, 256)
    _, evA, kA = dg.feature_gram(features("A", t))
    _, evB, kB = dg.feature_gram(features("B", t))
    np.testing.assert_allclose(evA, [0.00495, 0.0417, 0.185, 1.96, 2.81], rtol=0.01)
    np.testing.assert_allclose(evB, [0.566, 0.906, 0.921, 1.07, 1.53], rtol=0.01)
    assert abs(kA - 567.7) / 567.7 < 0.01 and abs(kB - 2.7) < 0.1


def _operator_mapped_features(kind, t):
    """D phi~ with D = 1 + (c/k) d/dt + (m/k) d^2/dt^2 applied to the standardised features."""
    phi = features(kind, t)
    mu, sd = phi.mean(0), phi.std(0)
    d1 = np.stack([jax.vmap(jax.jacfwd(lambda s: features(kind, s, jnp)))(t)[:, j] for j in range(5)], -1)
    d2 = np.stack([jax.vmap(jax.jacfwd(jax.jacfwd(lambda s: features(kind, s, jnp))))(t)[:, j] for j in range(5)], -1)
    return (phi - mu) / sd + (C_ / K_) * d1 / sd + (M_ / K_) * d2 / sd


def test_operator_mapped_gram_matches_notes():
    """kappa(G_DPhi) = 990.0 (A), 33.0 (B): the operator-mapped features, NOT re-standardised."""
    t = jnp.linspace(0, T, 256)
    _, _, kA = dg.feature_gram(_operator_mapped_features("A", t), standardise=False)
    _, _, kB = dg.feature_gram(_operator_mapped_features("B", t), standardise=False)
    assert abs(kA - 990.0) < 0.1 and abs(kB - 33.0) < 0.05
    # re-standardising would undo the operator's scaling and lose most of the gap for B
    assert dg.feature_gram(_operator_mapped_features("B", t))[2] < 3.0


def test_poisson_feature_gram_matches_notes():
    x = np.linspace(0, 1, 512)
    tau = 2 * x - 1
    phiA = np.stack([tau ** p for p in range(1, 6)], -1)
    phiB = np.stack([tau, np.sin(2 * np.pi * x), np.cos(2 * np.pi * x), np.sin(50 * np.pi * x),
                     np.cos(50 * np.pi * x)], -1)
    assert abs(dg.feature_gram(phiA)[2] - 568) < 1.0 and abs(dg.feature_gram(phiB)[2] - 8.0) < 0.05


# --------------------------------------------------------------------------- curvature
def test_lanczos_and_slq_vs_exact_hessian(spring_a):
    prob, _, params, *_ = spring_a
    ev = dg.full_hessian_spectrum(prob, params)
    assert abs(dg.sharpness(prob, params, 60) - ev[0]) / ev[0] < 1e-6
    n, nodes, w = dg.spectrum_slq(prob, params, k=80, n_probes=6)
    ne, nstd, lm = dg.effective_dimension(n, nodes, w, 1e-3)
    exact = int(np.sum(ev > 1e-3 * ev[0]))
    assert abs(lm - ev[0]) / ev[0] < 1e-3
    assert abs(ne - exact) <= max(3, 0.25 * exact), (ne, nstd, exact)


def test_top_ten_condition_number(spring_a):
    prob, _, params, *_ = spring_a
    ev = dg.full_hessian_spectrum(prob, params)
    top = dg.top_eigenvalues(prob, params, 10, k=60)
    np.testing.assert_allclose(top, ev[:10], rtol=1e-8)
    assert dg.kappa_top(top) == pytest.approx(ev[0] / ev[9], rel=1e-8)
    assert np.isnan(dg.kappa_top(ev[:5])) and np.isnan(dg.kappa_top(np.array([3.0, 2.0, -1.0]), 3))


def test_jacobi_scaling():
    rng = np.random.default_rng(1)
    B = rng.standard_normal((12, 30))
    H = B @ B.T
    s = np.exp(rng.uniform(-3, 3, 12))                                  # wild per-parameter scales
    ref = dg.jacobi_scaled_spectrum(H)
    np.testing.assert_allclose(dg.jacobi_scaled_spectrum(H * s[:, None] * s[None, :]), ref, rtol=1e-9)
    np.testing.assert_allclose(dg.jacobi_scaled_spectrum(np.diag(s)), np.ones(12), rtol=1e-12)
    Hz = np.zeros((13, 13))
    Hz[:12, :12] = H                                                     # a parameter with no curvature
    np.testing.assert_allclose(dg.jacobi_scaled_spectrum(Hz), ref, rtol=1e-9)


def test_layer_blocks_bounded_by_full_sharpness(spring_b):
    prob, _, params, *_ = spring_b
    lay = dg.layer_sharpness(prob, params, 30)
    assert set(lay) == {"Dense_0", "Dense_1"}
    top = dg.full_hessian_spectrum(prob, params)[0]
    assert all(0 < v <= top * (1 + 1e-9) for v in lay.values())          # Cauchy interlacing


def test_gauss_newton_identity_on_linear_model():
    rng = np.random.default_rng(0)
    A, b = jnp.asarray(rng.standard_normal((40, 9))), jnp.asarray(rng.standard_normal(40))
    prob = dg.Problem(lambda p, bt: {"ls": jnp.sum((A @ p["w"] - b) ** 2)}, None,
                      rho_fn=lambda p, bt: A @ p["w"] - b)
    w = {"w": jnp.asarray(rng.standard_normal(9))}
    for exact in (True, False):
        out = dg.gauss_newton_fidelity(prob, w, exact=exact)
        assert out["gn_rel_err"] < 1e-12 and out["method"] == ("exact" if exact else "lanczos")
        assert out["norm_H"] == pytest.approx(2 * np.linalg.norm(np.asarray(A), 2) ** 2, rel=1e-10)


def test_gauss_newton_fidelity_exact_vs_lanczos_and_training(spring_b):
    prob, p0, params, *_ = spring_b
    e0 = dg.gauss_newton_fidelity(prob, p0)
    e1 = dg.gauss_newton_fidelity(prob, params)
    e1l = dg.gauss_newton_fidelity(prob, params, exact=False, k=60)
    assert e1["method"] == "exact" and e1l["method"] == "lanczos"
    assert e1l["gn_rel_err"] == pytest.approx(e1["gn_rel_err"], rel=1e-6)
    assert e1["gn_rel_err"] < 1e-3 < e0["gn_rel_err"]                    # H -> 2 J^T J as residuals shrink


def test_inconsistent_rho_is_rejected(spring_b):
    prob, _, params, *_ = spring_b
    bad = dg.Problem(prob.loss_terms_fn, prob.batch, rho_fn=lambda p, b: prob.rho_fn(p, b)[:-2])  # drops the IC rows
    with pytest.raises(ValueError, match="does not match the loss"):
        dg.gauss_newton_fidelity(bad, params)
    with pytest.raises(ValueError, match="does not match the loss"):
        dg.output_layer_refit(bad, params, "Dense_1")
    assert prob.check_rho(params) < 1e-12


# --------------------------------------------------------------------------- kernel
def test_ntk_matches_dense_jacobian_and_gram_spectrum(spring_b):
    prob, _, params, *_ = spring_b
    flat, unravel = ravel_pytree(params)
    x = prob.x_res[:64]
    J = np.asarray(jax.jacrev(lambda v: prob.residual_fn(unravel(v), x)[:, 0])(flat))
    Jd = dg.residual_jacobian(prob, params, n_max=64)
    np.testing.assert_allclose(Jd, J, rtol=1e-9, atol=1e-12)
    K, ev = dg.ntk(prob, params, n_max=64)
    np.testing.assert_allclose(K, J @ J.T / 64, rtol=1e-9, atol=1e-14)
    gram = np.sort(np.linalg.eigvalsh(J.T @ J / 64))[::-1][:10]
    np.testing.assert_allclose(ev[:10], gram, rtol=1e-8)
    assert dg.ntk_drift(K, K) == 0.0


# --------------------------------------------------------------------------- landscape
def test_count_extrema():
    a = np.linspace(-1, 1, 201)
    assert dg.count_extrema(a ** 2) == 1
    assert dg.count_extrema(a ** 4 - a ** 2) == 3
    assert dg.count_extrema(a ** 2 + 1e-9 * np.sin(300 * a), rtol=1e-6) == 1       # noise below rtol ignored


def test_slice_passes_through_loss_and_direction_is_filter_normalised(spring_b):
    prob, _, params, *_ = spring_b
    a, f = dg.slice_1d(prob, params, n_dir=3, n_alpha=11)
    assert np.allclose(f[:, 5], float(prob.loss(params)), rtol=1e-10)
    d = dg.filter_normalised_direction(params, jax.random.PRNGKey(3))
    W, D = params["params"]["Dense_0"]["kernel"], d["params"]["Dense_0"]["kernel"]
    np.testing.assert_allclose(np.linalg.norm(D, axis=0), np.linalg.norm(W, axis=0), rtol=1e-10)
    b, Db = params["params"]["Dense_0"]["bias"], d["params"]["Dense_0"]["bias"]
    assert np.linalg.norm(Db) == pytest.approx(np.linalg.norm(b), rel=1e-10)        # bias: one block
    g, F = dg.slice_2d(prob, params, n=5)
    assert np.isclose(F[2, 2], float(prob.loss(params)), rtol=1e-10)


# --------------------------------------------------------------------------- design
def test_output_refit_is_exact_for_linear_problem(spring_a):
    prob, _, params, *_ = spring_a
    new, info = dg.output_layer_refit(prob, params, "Dense_1")
    assert info["loss_after"] <= info["loss_before"] and info["step"] == 1.0
    np.testing.assert_allclose(info["loss_after"], info["loss_predicted_full_step"], rtol=1e-8)
    # minimum: gradient wrt the output layer vanishes
    g = jax.grad(prob.loss)(new)["params"]["Dense_1"]
    gn = max(float(jnp.abs(v).max()) for v in g.values())
    g0 = jax.grad(prob.loss)(params)["params"]["Dense_1"]
    assert gn < 1e-6 * max(float(jnp.abs(v).max()) for v in g0.values()) + 1e-12


def _linear_layer_problem(n_rows=80, width=7, seed=0):
    """loss = ||H w + b - y||^2: an output layer on fixed features H, distinct unit importances."""
    rng = np.random.default_rng(seed)
    H = jnp.asarray(rng.standard_normal((n_rows, width)))
    y = H @ jnp.asarray(2.0 ** -np.arange(width) * (1 + 0.1 * rng.standard_normal(width))) + 0.3
    y = y + 1e-3 * jnp.asarray(rng.standard_normal(n_rows))

    def rho(p, _):
        return H @ p["params"]["out"]["kernel"][:, 0] + p["params"]["out"]["bias"][0] - y
    prob = dg.Problem(lambda p, b: {"ls": jnp.sum(rho(p, b) ** 2)}, None, rho_fn=rho)
    params = {"params": {"out": {"kernel": jnp.asarray(rng.standard_normal((width, 1))),
                                 "bias": jnp.zeros(1)}}}
    return prob, params, np.asarray(H), np.asarray(y)


def test_refit_pruning_matches_brute_force():
    prob, params, H, y = _linear_layer_problem()

    def ls_loss(units):
        A = np.concatenate([H[:, units], np.ones((len(y), 1))], 1)
        r = A @ np.linalg.lstsq(A, y, rcond=None)[0] - y
        return float(r @ r)
    active, order, losses = list(range(H.shape[1])), [], [ls_loss(list(range(H.shape[1])))]
    while len(active) > 1:
        trial = [ls_loss([u for u in active if u != v]) for v in active]
        drop = active[int(np.argmin(trial))]
        active.remove(drop)
        order.append(drop)
        losses.append(min(trial))
    out = dg.refit_pruning(prob, params, "out")
    assert out["removed"] == order and out["units"] == list(range(H.shape[1], 0, -1))
    np.testing.assert_allclose(out["loss"], losses, rtol=1e-9)
    np.testing.assert_allclose(out["loss_true"], out["loss"], rtol=1e-9)


def test_refit_pruning_on_network(spring_b):
    prob, _, params, *_ = spring_b
    out = dg.refit_pruning(prob, params, "Dense_1")
    _, info = dg.output_layer_refit(prob, params, "Dense_1")
    assert out["units"][0] == 32 and out["units"][-1] == 1 and sorted(out["removed"]) == sorted(set(out["removed"]))
    assert out["loss"][0] == pytest.approx(info["loss_after"], rel=1e-6)
    assert out["loss"][0] <= out["loss_trained"] * (1 + 1e-9)
    L = np.asarray(out["loss"])
    assert np.all(np.diff(L) >= -1e-9 * L[:-1])                         # fewer units never fit better
    np.testing.assert_allclose(out["loss_true"], L, rtol=1e-6)            # residual linear in the output layer


def test_offgrid_ratio_one_on_same_points(spring_b):
    prob, p0, *_ = spring_b
    r = dg.offgrid_residual_ratio(prob, p0, prob.x_res)
    assert abs(r["ratio"] - 1) < 1e-12


# --------------------------------------------------------------------------- representation
def test_erank_cka_probe_saturation():
    rng = np.random.default_rng(0)
    u = rng.standard_normal((500, 1))
    assert abs(dg.erank(u @ rng.standard_normal((1, 8))) - 1) < 1e-6
    Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    X = rng.standard_normal((500, 8))
    assert abs(dg.linear_cka(X, X) - 1) < 1e-12 and abs(dg.linear_cka(X, X @ Q) - 1) < 1e-12
    assert dg.linear_cka(X, rng.standard_normal((500, 8))) < 0.2
    y = X[:, :1] * 3.0
    pr = dg.linear_probe(X, y, ks=(1, 8))
    assert pr[8] < 1e-20
    assert dg.saturation(np.array([0.5, 2.5, -3.0, 1.0])) == 0.5
    # degenerate inputs give defined answers instead of nan/inf warnings
    assert dg.erank(np.ones((10, 4))) == 0.0 and not np.any(dg.singular_values(np.ones((10, 4))))
    assert np.isnan(dg.linear_probe(X, np.ones(500), ks=(1,))[1])


def test_hidden_fn_and_ablation(spring_b):
    prob, params, _, model, (mu, sd) = spring_b
    t = jnp.linspace(0, T, 256)[:, None]
    z, h = prob.hidden_fn(params, t)["Dense_0"]
    phi = (features("B", np.asarray(t[:, 0])) - mu) / sd
    d0 = params["params"]["Dense_0"]
    np.testing.assert_allclose(z, phi @ d0["kernel"] + d0["bias"], rtol=1e-12, atol=1e-12)
    assert z.shape == (256, 32) and np.allclose(h, np.tanh(z))
    w = params["params"]["Dense_1"]["kernel"]
    params = {"params": {**params["params"], "Dense_1": {**params["params"]["Dense_1"], "kernel": w.at[3].set(0.0)}}}
    imp = dg.ablation_importance(prob, params, t, "Dense_0", "Dense_1")
    assert imp.shape == (32,) and abs(imp[3] - 1) < 1e-9


# --------------------------------------------------------------------------- monitor
def test_monitor_tracks_terms_updates_and_best(spring_b):
    prob, params, *_ = spring_b
    mon = dg.TrainingMonitor(prob, params, every=10)
    opt = optax.adam(5e-3)
    st = opt.init(params)

    @jax.jit
    def step(p, s):
        upd, s = opt.update(jax.grad(prob.loss)(p), s)
        return optax.apply_updates(p, upd), s
    for k in range(101):
        mon.update(k, params)
        params, st = step(params, st)
    assert mon.rec["step"] == list(range(0, 101, 10))
    terms = np.array(mon.rec["terms"])
    assert terms.shape == (11, 2) and terms[-1].sum() < terms[0].sum()
    assert np.array(mon.rec["update"]).shape == (11, 2) and np.array(mon.rec["dist0"])[0].max() == 0.0
    assert np.array(mon.rec["grad_term"]).shape == (11, 2)
    assert mon.best_loss == min(t.sum() for t in terms)
    assert np.isclose(float(prob.loss(mon.best_params())), mon.best_loss)
    np.testing.assert_allclose(mon.total(), terms.sum(1))


def test_monitor_update_ratio_is_relative_to_previous_iterate(spring_b):
    """rho_l(k) = ||theta_l(k+1) - theta_l(k)|| / ||theta_l(k)|| (twolayer notes Sec. 3.2)."""
    prob, p0, *_ = spring_b
    mon = dg.TrainingMonitor(prob, p0, every=1, grad_terms=False)
    grads = jax.jit(jax.grad(prob.loss))
    traj, p = [], p0
    for k in range(4):
        mon.update(k, p)
        traj.append(p)
        p = jax.tree_util.tree_map(lambda a, g: a - 0.05 * g, p, grads(p))
    for k in range(1, 4):
        for j, lay in enumerate(["Dense_0", "Dense_1"]):
            a, b = traj[k]["params"][lay], traj[k - 1]["params"][lay]
            num = np.sqrt(sum(float(jnp.sum((a[n] - b[n]) ** 2)) for n in a))
            den = np.sqrt(sum(float(jnp.sum(b[n] ** 2)) for n in b))
            assert mon.rec["update"][k][j] == pytest.approx(num / den, rel=1e-10)


def test_best_so_far_and_iterations_to():
    L = [1.0, 0.5, 2.0, 0.2, 0.3, 0.05]
    np.testing.assert_array_equal(dg.best_so_far(L), [1.0, 0.5, 0.5, 0.2, 0.2, 0.05])
    assert dg.iterations_to(L, 0.25) == 3 and dg.iterations_to(L, 1e-3) is None
    assert dg.iterations_to(L, 0.25, steps=[0, 10, 20, 30, 40, 50]) == 30


# --------------------------------------------------------------------------- spectra
def test_amplitude_spectrum_and_energy_fraction():
    n, dt = 1024, 3.0 / 1024
    t = np.arange(n) * dt
    om = 2 * np.pi * np.fft.rfftfreq(n, dt)
    w_lo, w_hi = om[20], om[300]                                         # on-bin frequencies
    s = 0.7 * np.sin(w_lo * t) + 0.2 * np.cos(w_hi * t)
    omega, amp = dg.amplitude_spectrum(s, dt)
    np.testing.assert_allclose(omega, om)
    assert omega[np.argmax(amp)] == w_lo
    assert amp[20] == pytest.approx(0.7, rel=2e-3) and amp[300] == pytest.approx(0.2, rel=2e-3)
    assert dg.energy_fraction_below(s, dt, 0.5 * (w_lo + w_hi)) == pytest.approx(0.49 / 0.53, rel=1e-3)
    _, flat = dg.amplitude_spectrum(np.full(n, 2.0), dt, window=None)
    assert flat[0] == pytest.approx(2.0) and np.allclose(flat[1:], 0, atol=1e-12)   # DC not doubled
    assert dg.grid_step(t) == pytest.approx(dt)
    with pytest.raises(ValueError):
        dg.grid_step(t ** 2)


# --------------------------------------------------------------------------- modal (Poisson notes)
def test_sine_coefficients_and_mode_weights_match_poisson_notes():
    x = np.linspace(0, 1, 2001)
    u_star = np.sin(2 * np.pi * x) + 0.1 * np.sin(50 * np.pi * x)
    c = dg.sine_coefficients(u_star, x, 80)
    expect = np.zeros(80)
    expect[1], expect[49] = 1.0, 0.1                                      # a*_2 = 1, a*_50 = 0.1
    np.testing.assert_allclose(c, expect, atol=1e-10)
    S = 0.1 * (50 * np.pi) ** 2
    assert S == pytest.approx(2467.4, abs=0.05)
    w = dg.laplacian_mode_weights(80, scale=S)
    assert w[1] / w[49] == pytest.approx((2 / 50) ** 4) and w[1] / w[49] == pytest.approx(2.6e-6, rel=0.02)
    # Parseval (poisson notes eq. 6): the mean squared residual of an error e is 1/2 sum_n w_n da_n^2
    da = np.zeros(80)
    da[1], da[49] = 0.3, 0.01
    xc = np.linspace(0, 1, 512)
    e2 = -(0.3 * (2 * np.pi) ** 2 * np.sin(2 * np.pi * xc) + 0.01 * (50 * np.pi) ** 2 * np.sin(50 * np.pi * xc))
    assert np.mean((e2 / S) ** 2) == pytest.approx(0.5 * np.sum(w * da ** 2), rel=1e-2)
    np.testing.assert_allclose(dg.modal_error(u_star + 0.3 * np.sin(2 * np.pi * x), u_star, x, 3), [0, 0.3, 0],
                               atol=1e-10)


def test_modal_kernel_of_model_linear_in_parameters():
    """u_theta = sum_n theta_n sin(n pi x): c_n = theta_n, so mu_n = 1 for n <= 6, 0 beyond, Theta = I."""
    def apply_fn(p, x):
        return jnp.sin(jnp.pi * x[:, :1] * jnp.arange(1, 7)) @ p["a"]
    params = {"a": jnp.linspace(0.1, 0.6, 6)}
    x = np.linspace(0, 1, 801)
    mu, Th = dg.modal_kernel(apply_fn, params, x, 9, full=True)
    np.testing.assert_allclose(mu, [1] * 6 + [0] * 3, atol=1e-10)
    np.testing.assert_allclose(Th[:6, :6], np.eye(6), atol=1e-10)
    np.testing.assert_allclose(dg.sine_coefficients(apply_fn(params, x[:, None]), x, 6), params["a"], atol=1e-10)
    with pytest.raises(ValueError):
        dg.modal_kernel(apply_fn, params, x[::-1], 3)


# --------------------------------------------------------------------------- report
def test_run_checkpoint_end_to_end_and_skips_gracefully(tmp_path, spring_b):
    prob, p0, params, *_ = spring_b
    xd = np.linspace(0, T, 500)[:, None]
    cfg = dg.Config(n_ntk=32, n_dir=2, n_alpha=21, n2d=5, refit_layer="Dense_1", prune=True, x_dense=xd,
                    reference_fn=lambda x: u_exact(x[:, 0], jnp))
    res = dg.run_checkpoint(prob, params, params0=p0, cfg=cfg, out_dir=str(tmp_path), step=7)
    for k in ("curvature", "layer_sharpness", "gauss_newton", "dist0", "ntk", "slices", "hidden",
              "output_refit", "pruning", "offgrid", "error"):
        assert k in res, (k, res.get("skipped"))
    assert not res.get("skipped")
    cur = res["curvature"]
    assert cur["method"] == "exact" and cur["kappa_top"] == pytest.approx(cur["eigs"][0] / cur["eigs"][9])
    assert 1 < cur["kappa_top_jacobi"] < cur["kappa_top"]
    e = np.asarray(prob.apply_fn(params, jnp.asarray(xd))) - u_exact(xd[:, 0])
    assert res["error"]["rel_l2"] == pytest.approx(np.linalg.norm(e) / np.linalg.norm(u_exact(xd[:, 0])), rel=1e-10)
    assert res["error"]["error_amp"].shape[0] == 251
    assert set(res["dist0"]) == {"Dense_0", "Dense_1"} and all(v > 0 for v in res["dist0"].values())
    assert (tmp_path / "ckpt_0000007.json").exists() and (tmp_path / "ckpt_0000007.npz").exists()
    # a Problem without rho_fn/hidden_fn: those diagnostics are skipped with a reason, the rest still
    # run; full_hessian_max=0 forces the Lanczos/SLQ path
    bare = dg.Problem(prob.loss_terms_fn, prob.batch, residual_fn=prob.residual_fn, x_res=prob.x_res)
    res2 = dg.run_checkpoint(bare, params, cfg=dg.Config(n_ntk=16, n_dir=1, n_alpha=11, full_hessian_max=0))
    assert {"gauss_newton", "hidden"} <= set(res2["skipped"]) and "ntk" in res2
    assert res2["curvature"]["method"] == "slq"
    assert res2["curvature"]["lambda_max"] == pytest.approx(cur["lambda_max"], rel=1e-6)
