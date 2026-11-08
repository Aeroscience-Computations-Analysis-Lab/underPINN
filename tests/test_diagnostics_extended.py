"""Tests for the extended diagnostics: loss blocks, balance, profiles, N-D spectra,
generic hidden activations, interpretation and their run_checkpoint integration."""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from underPINN import diagnostics as dg
from underPINN.losses.loss import PINNLoss
from underPINN.nn.factory import build_model
from underPINN.pde.failure_modes import ConvectionPDE
from underPINN.pde.wave import WavePDE


def _mlp(d_in=2, width=8):
    m = build_model({"type": "mlp", "layers": [d_in, width, width, 1]})
    return m, m.init(jax.random.PRNGKey(0), jnp.zeros((1, d_in)))


def _pinn_data(n=24, seed=0):
    r = np.random.default_rng(seed)
    f = lambda a: jnp.asarray(a, jnp.float32)                                   # noqa: E731
    return (f(r.uniform(0, 1, n)), f(r.uniform(0, 1, n)), f(np.linspace(0, 1, n)),
            f(np.sin(np.pi * np.linspace(0, 1, n))), f(r.uniform(0, 1, n)), f(r.uniform(0, 1, n)),
            f(r.uniform(-1, 1, n)))


# ── problem construction ─────────────────────────────────────────────────────

def test_blocks_loss_rho_and_residual_agree():
    m, p = _mlp()
    X = jax.random.uniform(jax.random.PRNGKey(1), (30, 2))
    blocks = {"pde": dg.LossBlock(lambda q, x: m.apply(q, x)[:, 0] - x[:, 0], X, 1.0),
              "data": dg.LossBlock(lambda q, x: jnp.stack([m.apply(q, x)[:, 0], 2 * x[:, 1]], 1), X[:10], 3.0)}
    prob = dg.problem_from_blocks(blocks)
    u = m.apply(p, X)[:, 0]
    manual = jnp.mean((u - X[:, 0]) ** 2) + 3.0 * jnp.mean(
        jnp.stack([u[:10], 2 * X[:10, 1]], 1) ** 2)
    np.testing.assert_allclose(float(prob.loss(p)), float(manual), rtol=1e-6)
    assert prob.check_rho(p) < 1e-5
    assert prob.residual_fn(p, X).shape == (30, 1)
    with pytest.raises(ValueError):
        dg.problem_from_blocks(blocks, residual_block="nope")


@pytest.mark.parametrize("kind", ["dirichlet", "periodic", "velocity"])
def test_problem_from_pinnloss_matches_pinnloss(kind):
    m, p = _mlp()
    if kind == "velocity":
        loss = PINNLoss(m, WavePDE(m, c=2.0), ic_weight=7.0, bc_weight=3.0, ic_velocity_weight=0.5)
    else:
        loss = PINNLoss(m, ConvectionPDE(m, 5.0), ic_weight=7.0, bc_weight=3.0,
                        periodic_length=1.0 if kind == "periodic" else None)
    data = _pinn_data()
    total, _ = loss(p, *data)
    prob = dg.problem_from_pinnloss(loss, *data)
    np.testing.assert_allclose(float(prob.loss(p)), float(total), rtol=1e-5)
    assert prob.check_rho(p) < 1e-5
    assert set(prob.blocks) >= {"pde", "ic", "bc"}


def test_problem_from_pinnloss_warns_on_rba():
    m, _ = _mlp()
    with pytest.warns(UserWarning, match="rba"):
        dg.problem_from_pinnloss(PINNLoss(m, ConvectionPDE(m), rba=True), *_pinn_data())


# ── loss balance ─────────────────────────────────────────────────────────────

def test_gradient_conflict_exact_for_quadratics():
    a, b = np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    theta = {"w": jnp.array([0.5, 0.2, 0.1])}
    prob = dg.Problem(lambda q, _: {"A": jnp.dot(a, q["w"]) ** 2, "B": 4.0 * jnp.dot(b, q["w"]) ** 2}, None)
    gc = dg.gradient_conflict(prob, theta)
    np.testing.assert_allclose(gc["cos"][0, 1], a @ b, rtol=1e-6)            # both projections positive
    np.testing.assert_allclose(gc["grad_norm"]["A"], 2 * 0.5, rtol=1e-6)
    np.testing.assert_allclose(sum(gc["share"].values()), 1.0, rtol=1e-6)
    gc2 = dg.gradient_conflict(dg.Problem(lambda q, _: {"A": q["w"][0], "B": -q["w"][0]}, None), theta)
    assert gc2["min_cos"] == pytest.approx(-1.0)


def test_term_ntk_matches_hand_jacobian():
    # u(x) = w . phi(x) is linear in w, so the block Jacobian is phi(x) exactly
    class Lin(nn.Module):
        @nn.compact
        def __call__(self, x):
            phi = jnp.concatenate([x, jnp.sin(x)], 1)
            return phi @ self.param("w", nn.initializers.ones, (4, 1))
    m = Lin()
    p = m.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
    X = jax.random.uniform(jax.random.PRNGKey(2), (12, 2))
    phi = np.concatenate([X, np.sin(X)], 1)
    blocks = {"pde": dg.LossBlock(lambda q, x: m.apply(q, x)[:, 0], X, 1.0),
              "bc": dg.LossBlock(lambda q, x: 3.0 * m.apply(q, x)[:, 0], X[:6], 10.0)}
    tn = dg.term_ntk(dg.problem_from_blocks(blocks), p, n_max=12)
    md_pde = np.mean(np.sum(phi ** 2, 1))
    md_bc = np.mean(np.sum((3 * phi[:6]) ** 2, 1))
    np.testing.assert_allclose(tn["mean_diag"]["pde"], md_pde, rtol=1e-5)
    np.testing.assert_allclose(tn["mean_diag"]["bc"], md_bc, rtol=1e-5)
    np.testing.assert_allclose(tn["effective_rate"]["bc"], 10 * md_bc, rtol=1e-5)
    np.testing.assert_allclose(tn["ntk_weights"]["pde"], (md_pde + md_bc) / md_pde, rtol=1e-5)
    assert tn["slowest"] == "pde"
    with pytest.raises(ValueError, match="blocks"):
        dg.term_ntk(dg.Problem(lambda q, _: {"a": 0.0}, None), p)


# ── profiles ────────────────────────────────────────────────────────────────

def _synthetic(res_fn, apply_fn=None):
    return dg.Problem(lambda q, _: {"a": jnp.zeros(())}, None, residual_fn=res_fn, apply_fn=apply_fn)


def test_binned_mean_and_causal_weights():
    c, m, n = dg.binned_mean([1, 2, 3, 4], [0.0, 0.1, 0.9, 1.0], n_bins=2)
    np.testing.assert_allclose(m, [1.5, 3.5])
    np.testing.assert_array_equal(n, [2, 2])
    np.testing.assert_allclose(dg.causal_weights([1.0, 2.0, 3.0], eps=1.0), np.exp([0.0, -1.0, -3.0]))


def test_residual_profile_detects_growth_along_time():
    x = jnp.asarray(np.stack([np.zeros(2000), np.linspace(0, 1, 2000)], 1))
    prob = _synthetic(lambda q, z: (z[:, 1:2] ** 2) * 10.0)                     # residual grows with t
    prof = dg.residual_profile(prob, None, x, axis=-1, n_bins=10, eps=10.0)
    assert prof["growth"] > 100
    assert np.all(np.diff(prof["mean_sq_residual"]) > 0)
    assert prof["min_causal_weight"] < 1e-3
    flat = dg.residual_profile(_synthetic(lambda q, z: jnp.ones((z.shape[0], 1))), None, x)
    assert flat["growth"] == pytest.approx(1.0)


def test_error_profile_and_correlation_and_concentration():
    x = jnp.asarray(np.stack([np.linspace(-1, 1, 500), np.linspace(0, 1, 500)], 1))
    prob = _synthetic(lambda q, z: (z[:, :1] ** 2), apply_fn=lambda q, z: z[:, 0] ** 2)
    ref = lambda z: jnp.zeros(z.shape[0])                                      # noqa: E731
    assert dg.residual_error_correlation(prob, None, x, ref)["spearman"] == pytest.approx(1.0)
    ep = dg.error_profile(prob, None, x, lambda z: 1.0 + 0 * z[:, 0], axis=-1, n_bins=5)
    assert ep["rel_sq_error"].shape == (5,)
    spike = _synthetic(lambda q, z: jnp.where(jnp.arange(z.shape[0])[:, None] < 5, 100.0, 0.01))
    conc = dg.residual_concentration(spike, None, x)
    assert conc["top_share"] > 0.99 and conc["gini"] > 0.9


# ── spectra ─────────────────────────────────────────────────────────────────

def test_radial_spectrum_peak_and_spectral_error_bandwidth():
    n = 64
    g = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(g, g, indexing="ij")
    k, A = dg.radial_spectrum(np.sin(3 * X), g[1] - g[0], n_shells=32, window=None)
    assert abs(k[np.argmax(A)] - 3.0) < 1.0
    ref = np.sin(2 * X) + 0.5 * np.sin(12 * Y)
    se = dg.spectral_error(np.sin(2 * X), ref, g[1] - g[0], n_shells=32)       # misses the k=12 mode
    assert 2.0 <= se["resolved_k"] < 12.0
    assert dg.spectral_error(ref, ref, g[1] - g[0], n_shells=32)["resolved_k"] >= 12.0


# ── representation for any architecture ─────────────────────────────────────

@pytest.mark.parametrize("cfg", [
    {"type": "mlp", "layers": [2, 8, 8, 1]},
    {"type": "piratenet", "width": 8, "n_blocks": 1},
    {"type": "pinnsformer", "d_model": 8, "heads": 2, "d_ff": 8, "d_hidden": 8},
    {"type": "chebykan", "layers": [2, 6, 6, 1], "degree": 3},
])
def test_generic_hidden_fn_any_architecture(cfg):
    m = build_model(cfg)
    p = m.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
    x = jax.random.uniform(jax.random.PRNGKey(1), (40, 2), minval=-1, maxval=1)
    hf = dg.generic_hidden_fn(m)
    acts = hf(p, x)
    assert acts and all(h.shape[0] == 40 and h.ndim == 2 for _, h in acts.values())
    prob = dg.Problem(lambda q, _: {"a": jnp.zeros(())}, None, hidden_fn=hf)
    st = dg.hidden_stats(prob, p, x)
    assert all(np.isfinite(v["erank"]) for v in st.values())


@pytest.mark.parametrize("cfg,last", [({"type": "mlp", "layers": [2, 8, 8, 1]}, "Dense_2"),
                                      ({"type": "chebykan", "layers": [2, 6, 6, 1], "degree": 3}, "layer_2"),
                                      ({"type": "piratenet", "width": 8, "n_blocks": 1}, "head")])
def test_generic_hidden_fn_excludes_output_layer(cfg, last):
    """The submodule that produces the network output is not a hidden layer."""
    m = build_model(cfg)
    p = m.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
    acts = dg.generic_hidden_fn(m)(p, jax.random.uniform(jax.random.PRNGKey(1), (20, 2)))
    assert last not in acts and len(acts) >= 2


def test_generic_hidden_fn_reproduces_mlp_activations():
    m, p = _mlp(width=8)
    x = jax.random.uniform(jax.random.PRNGKey(1), (10, 2))
    acts = dg.generic_hidden_fn(m)(p, x)
    W0, b0 = p["params"]["Dense_0"]["kernel"], p["params"]["Dense_0"]["bias"]
    np.testing.assert_allclose(np.asarray(acts["Dense_0"][1]), np.asarray(x @ W0 + b0), rtol=1e-5)


# ── interpretation ──────────────────────────────────────────────────────────

def test_interpret_rules_fire_and_stay_quiet():
    res = {"loss_terms": {"pde": 0.01, "bc": 0.99},
           "gradient_conflict": {"grad_norm": {"pde": 1.0, "bc": 500.0}, "norm_ratio": 500.0, "min_cos": -0.6,
                                 "share": {"pde": 0.02, "bc": 0.98}, "lra_factor": {"pde": 1.0, "bc": 0.01}},
           "residual_profile": {"growth": 50.0, "min_causal_weight": 1e-6, "eps": 1.0},
           "curvature": {"lambda_max": 1e5, "kappa_top": 1e4},
           "offgrid": {"ratio": 5.0}}
    f = dg.interpret(res, lr=1e-3)
    areas = {x.area for x in f}
    assert {"loss balance", "causality", "landscape", "sampling"} <= areas
    assert f[0].severity == "high"
    assert "lr" in dg.to_markdown(f) or "λ_max" in dg.to_markdown(f)
    assert dg.interpret({"loss_terms": {"pde": 0.5, "bc": 0.5}}) == []
    assert "No issues" in dg.to_markdown([])


# ── integration ─────────────────────────────────────────────────────────────

def test_run_checkpoint_with_new_sections(tmp_path):
    m, p = _mlp()
    loss = PINNLoss(m, ConvectionPDE(m, 5.0), ic_weight=7.0, bc_weight=3.0, periodic_length=1.0)
    prob = dg.problem_from_pinnloss(loss, *_pinn_data())
    g = np.linspace(0, 1, 12)
    X, T = np.meshgrid(g, g, indexing="ij")
    xd = jnp.asarray(np.stack([X.ravel(), T.ravel()], 1), jnp.float32)
    ref = lambda z: jnp.sin(z[:, 0] - 5.0 * z[:, 1])                           # noqa: E731
    cfg = dg.Config(lanczos_k=6, slq_k=6, slq_probes=1, n_ntk=8, n_dir=2, n_alpha=11, layer_k=4,
                    x_dense=xd, reference_fn=ref, term_ntk=8, profile_axis=-1, causal_eps=1.0,
                    grid_shape=(12, 12), grid_spacings=(g[1] - g[0], g[1] - g[0]), lr=1e-3)
    res = dg.run_checkpoint(prob, p, params0=p, cfg=cfg, out_dir=str(tmp_path), step=7)
    for k in ("gradient_conflict", "term_ntk", "residual_profile", "error_profile",
              "residual_concentration", "residual_error_correlation", "spectral_error", "findings"):
        assert k in res, (k, res.get("skipped"))
    assert isinstance(res["findings_md"], str)
    assert (tmp_path / "findings_0000007.md").exists()


def test_monitor_records_gradient_conflict():
    m, p = _mlp()
    prob = dg.problem_from_pinnloss(PINNLoss(m, ConvectionPDE(m, 5.0)), *_pinn_data())
    mon = dg.TrainingMonitor(prob, p, every=1)
    mon.update(0, p)
    assert len(mon.rec["grad_cos_min"]) == 1 and -1.0 <= mon.rec["grad_cos_min"][0] <= 1.0


def test_interpret_detects_trivial_solution_signature():
    """Error accumulating along time while the residual falls -> collapsed solution
    (the convection failure mode); error growth alone gives the milder finding."""
    t = np.linspace(0, 1, 16)
    err = np.clip(2 * t, 0.01, 1.0) ** 2                       # relative error 0.1 -> 1
    res_fall = {"mean_sq_residual": 0.05 * np.exp(-6 * t), "growth": 0.01}
    ep = {"rel_sq_error": err, "early": float(np.sqrt(err[:4].mean())), "late": 1.0, "growth": 9.0}
    f = dg.interpret({"residual_profile": res_fall, "error_profile": ep})
    assert any("residual falls" in x.finding for x in f)
    g = dg.interpret({"error_profile": ep})
    assert any(x.area == "causality" and "residual falls" not in x.finding for x in g)
    assert dg.interpret({"error_profile": {**ep, "late": 0.05}}) == []


def test_interpret_is_context_aware():
    """Opposing gradients / a dominant PDE term are expected at convergence (info),
    but actionable when the model is failing or an IC/BC term dominates."""
    base = {"gradient_conflict": {"grad_norm": {"pde": 1.0, "bc": 1.2}, "norm_ratio": 1.2, "min_cos": -0.4,
                                  "share": {"pde": 0.5, "bc": 0.5}, "lra_factor": {"pde": 1.0, "bc": 1.0},
                                  "cancellation": 0.05},
            "loss_terms": {"pde": 0.9, "bc": 0.1}}
    good = dg.interpret({**base, "rel_l2": 1e-4})
    assert good and all(f.severity == "info" for f in good)
    bad = dg.interpret({**base, "rel_l2": 0.5})
    assert any(f.severity == "high" and "opposing" in f.finding for f in bad)
    bc_dom = dg.interpret({**base, "loss_terms": {"pde": 0.1, "bc": 0.9}})
    assert any(f.severity == "high" for f in bc_dom)


def test_point_singularity_is_located_and_flagged():
    """A residual spike at one point (e.g. a normalisation layer's 1/sqrt(eps) at a zero
    embedding) must be located and flagged as a singularity, not as under-sampling."""
    x = jnp.asarray(np.stack(np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 41)), -1).reshape(-1, 2))
    spike = _synthetic(lambda q, z: (0.01 + 1e4 * jnp.exp(-((z[:, :1]) ** 2 + (z[:, 1:]) ** 2) / 1e-4)))
    rc = dg.residual_concentration(spike, None, x)
    np.testing.assert_allclose(rc["worst_points"][0], [0.0, 0.0], atol=1e-6)
    assert rc["max_over_median"] > 1e5
    f = dg.interpret({"residual_concentration": rc})
    assert any("concentrated at a point" in g.finding and g.severity == "high" for g in f)


def test_moderate_error_growth_is_flagged():
    t = np.linspace(0, 1, 16)
    rel = (6e-4 + 0.05 * t ** 2) ** 2
    ep = {"rel_sq_error": rel, "early": float(np.sqrt(rel[0])), "late": float(np.sqrt(rel[-4:].mean())), "growth": 50}
    f = dg.interpret({"error_profile": ep})
    assert any(g.area == "causality" and g.severity == "medium" for g in f)
