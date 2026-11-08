"""Turn diagnostic numbers into findings you can act on.

``interpret(res, lr=None)`` reads whatever is present in a diagnostics result dict (as
returned by :func:`~underPINN.diagnostics.report.run_checkpoint`, or assembled by hand from
the individual functions) and returns a list of :class:`Finding` — severity, area, what was
observed, why it matters, and what to try. ``to_markdown(findings)`` formats them.

Rules (thresholds are deliberately conservative; every finding quotes its evidence so the
reader can disagree):

loss balance      an IC/BC/data term ≥ 80 % of the loss                      -> re-weight / hard constraint
                  (the PDE term dominating is normal at convergence: info only)
                  gradient norm ratio ≥ 100                                  -> LRA / NTK weights
                  a cosine ≤ -0.2 while failing (error ≥ 5 % or IC/BC dominant) -> PCGrad / re-weight
                  (near a minimum opposing term gradients are expected: info only)
                  per-term NTK effective-rate imbalance ≥ 100                -> NTK-balanced weights
causality         residual growth along time ≥ 10, or min causal weight ≤ 1e-3 -> causal weighting / time marching
                  late relative error ≥ 0.3 and ≥ 3× early error             -> errors propagate forward
                  late ≥ 0.01 and ≥ 10× early (milder)                       -> accuracy degrades with time
                    ... while the residual profile falls (rank corr ≤ -0.5)  -> trivial / collapsed solution
landscape         λ_max·lr ≥ 2 (with lr)                                     -> step exceeds the stability limit
                  κ₁₀ ≥ 1e3                                                  -> ill-conditioned: preconditioner / 2nd order
kernel            NTK rank ≤ 5 % of rows                                     -> strong spectral bias: Fourier features
                  NTK drift ≤ 0.05 (end vs init)                             -> lazy regime: features never learned
                  NTK trace ratio ≤ 0.05                                     -> kernel collapsed: training stalls
sampling          off-grid residual ratio ≥ 3                                -> more / adaptive collocation points
                  max/median |r| ≥ 1e3 or top-10 % share ≥ 0.95              -> point singularity (network or data)
                  top-10 % residual share ≥ 0.5 and Spearman(|r|,|e|) ≥ 0.5  -> residual-adaptive sampling (RAR-D)
                  Spearman(|r|,|e|) ≤ 0.2 while failing                      -> residual is not an error indicator
output layer      LS refit lowers the loss ≥ 2× (ratio = before/after ≥ 2)   -> last layer unconverged: LS refit
representation    mean saturation ≥ 0.3                                      -> saturated units: rescale inputs/init
                  effective rank ≤ 25 % of width                             -> redundant width
spectrum          resolved bandwidth < 25 % of the solution's band           -> spectral bias limits accuracy
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["Finding", "interpret", "to_markdown"]

_SEV = {"high": 0, "medium": 1, "info": 2}


@dataclass
class Finding:
    severity: str            # "high" | "medium" | "info"
    area: str
    finding: str
    evidence: str
    action: str
    data: Dict[str, Any] = field(default_factory=dict)


def _f(x, fmt=".3g"):
    return format(float(x), fmt)


_PDE_NAMES = ("pde", "residual", "physics", "res")


def _overall_error(res) -> Optional[float]:
    e = res.get("rel_l2")
    if e is None and isinstance(res.get("error"), dict):
        e = res["error"].get("rel_l2")
    return None if e is None else float(e)


def interpret(res: Dict[str, Any], lr: Optional[float] = None,
              pde_terms=_PDE_NAMES, error_threshold: float = 0.05) -> List[Finding]:
    """Findings from a diagnostics dict; keys that are missing are simply not interpreted.

    ``pde_terms``: names of PDE-residual terms (their dominance is normal at convergence).
    ``error_threshold``: relative error above which the model counts as failing; used to
    decide whether conflict / indicator findings are actionable (needs ``res["rel_l2"]`` or
    ``res["error"]["rel_l2"]``; without it those findings are reported as info).
    """
    out: List[Finding] = []
    add = lambda *a, **k: out.append(Finding(*a, **k))            # noqa: E731
    err = _overall_error(res)
    failing = err is not None and err >= error_threshold
    is_pde = lambda n: str(n).lower() in pde_terms                  # noqa: E731

    # ── loss balance ────────────────────────────────────────────────────────
    terms = res.get("loss_terms")
    if terms and len(terms) > 1:
        tot = sum(terms.values())
        name, val = max(terms.items(), key=lambda kv: kv[1])
        if tot > 0 and val / tot >= 0.8:
            if is_pde(name):
                add("info", "loss balance", f"the PDE residual is {val / tot:.0%} of the loss",
                    ", ".join(f"{k}={_f(v)}" for k, v in terms.items()),
                    "IC/BC are satisfied; the residual is the limiting term (normal near convergence). "
                    "To go further: more capacity, more/adaptive points, or a second-order phase.",
                    {"term": name, "share": val / tot})
            else:
                add("high" if failing else "medium", "loss balance", f"'{name}' is {val / tot:.0%} of the loss",
                    ", ".join(f"{k}={_f(v)}" for k, v in terms.items()),
                    f"The unsatisfied constraint is '{name}'. Raise its weight, enforce it exactly "
                    f"(ansatz / hard constraint), or check that it is consistent with the other data.",
                    {"term": name, "share": val / tot})
    gc = res.get("gradient_conflict")
    if gc:
        if gc["norm_ratio"] >= 100:
            big = max(gc["grad_norm"], key=gc["grad_norm"].get)
            small = min(gc["grad_norm"], key=gc["grad_norm"].get)
            add("high", "loss balance", f"gradient of '{big}' is {gc['norm_ratio']:.0f}× that of '{small}'",
                ", ".join(f"|∇{k}|={_f(v)}" for k, v in gc["grad_norm"].items()),
                "Stiff gradient imbalance (Wang et al. 2021): the optimiser serves "
                f"'{big}' and starves '{small}'. Multiply weights by the learning-rate-annealing "
                "factors " + ", ".join(f"{k}×{_f(v, '.2g')}" for k, v in gc["lra_factor"].items()) + ".",
                {"lra_factor": gc["lra_factor"]})
        if gc["min_cos"] <= -0.2:
            non_pde_dominates = bool(terms) and any(
                (not is_pde(k)) and v >= 0.5 * sum(terms.values()) for k, v in terms.items())
            canc = gc.get("cancellation")
            ev = "pairwise cos(∇Lᵢ, ∇Lⱼ) in gradient_conflict['cos']" + (
                f"; cancellation ‖Σ∇‖/Σ‖∇‖ = {canc:.2f}" if canc is not None else "")
            if failing or non_pde_dominates:
                add("high", "loss balance", f"loss terms pull in opposing directions (min cosine {gc['min_cos']:.2f})",
                    ev, "The optimiser is stuck in a compromise between terms that the solution cannot "
                    "satisfy together. Re-weight (LRA / NTK weights), use gradient surgery (PCGrad), or "
                    "impose the IC/BC exactly so the conflict disappears.", {"min_cos": gc["min_cos"]})
            else:
                add("info", "loss balance", f"term gradients oppose each other (min cosine {gc['min_cos']:.2f})",
                    ev, "Expected near a minimum of a weighted sum, where term gradients cancel; "
                    "actionable only if the solution error is large.", {"min_cos": gc["min_cos"]})
        dom = max(gc["share"].items(), key=lambda kv: kv[1])
        if dom[1] >= 0.9 and len(gc["share"]) > 1:
            if dom[1] <= 1.05:
                add("info", "loss balance", f"the update direction is {dom[1]:.0%} '{dom[0]}'",
                    ", ".join(f"{k}: {_f(v, '.2f')}" for k, v in gc["share"].items()),
                    "Only one term is effectively being trained at this point.")
            else:
                add("info", "loss balance", f"'{dom[0]}' pulls {dom[1]:.1f}× harder than the net step; "
                    "the other terms push back", ", ".join(f"{k}: {_f(v, '.2f')}" for k, v in gc["share"].items()),
                    "Terms largely cancel: the step is the small difference of large opposing pulls.")
    tn = res.get("term_ntk")
    if tn and tn.get("imbalance", 0) >= 100:
        add("high", "loss balance", f"per-term NTK rates differ {tn['imbalance']:.0f}×; slowest: '{tn['slowest']}'",
            ", ".join(f"{k}: w·tr/n={_f(v)}" for k, v in tn["effective_rate"].items()),
            f"'{tn['slowest']}' converges orders of magnitude slower (Wang, Yu & Perdikaris 2022). "
            "NTK-balanced weights: " + ", ".join(f"{k}={_f(v, '.2g')}" for k, v in tn["ntk_weights"].items()) + ".",
            {"ntk_weights": tn["ntk_weights"]})

    # ── causality ───────────────────────────────────────────────────────────
    rp, ep = res.get("residual_profile"), res.get("error_profile")
    if rp and rp.get("growth", 0) >= 10:
        add("medium", "causality", f"residual grows {rp['growth']:.0f}× from early to late times",
            "residual_profile['mean_sq_residual'] per time bin",
            "Late times are less resolved than early ones. Train causally: causal weights, "
            "time-marching windows, or curriculum on the time horizon.", {"growth": rp["growth"]})
    if rp and rp.get("min_causal_weight", 1.0) <= 1e-3:
        add("high", "causality", f"causal weights fall to {rp['min_causal_weight']:.1e}",
            f"w_k = exp(-ε Σ_(j<k) L_j), ε={rp.get('eps')}",
            "Early-time residual is large enough that later times are not yet meaningful. Use "
            "causal training (Wang, Sankaran & Perdikaris 2022) or time marching.",
            {"min_causal_weight": rp["min_causal_weight"]})
    if ep and "late" in ep:
        early, late = ep["early"], ep["late"]
        if late >= 0.3 and late >= 3 * max(early, 1e-12):
            anti = None
            if rp is not None:
                a = np.asarray(rp["mean_sq_residual"], float)
                b = np.asarray(ep["rel_sq_error"], float)
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() >= 4:
                    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
                    anti = float(np.corrcoef(ra, rb)[0, 1])
            if anti is not None and anti <= -0.5:
                add("high", "causality", f"error accumulates along time ({early:.2g} → {late:.2g}) while the "
                    f"residual falls (rank corr {anti:.2f})",
                    "error_profile vs residual_profile",
                    "Trivial / collapsed solution: the network satisfies the PDE where it has given up on "
                    "the true solution (typical of transport and wave problems). Train causally (causal "
                    "weights, time marching, curriculum on the horizon) and enforce IC/BC harder or "
                    "exactly; more collocation points will not help.",
                    {"early": early, "late": late, "profile_rank_corr": anti})
            else:
                add("high", "causality", f"error accumulates along time ({early:.2g} → {late:.2g})",
                    "error_profile: relative error of the first vs last quarter of the time bins",
                    "Errors made early propagate forward. Train causally (causal weights or time "
                    "marching) so early times are resolved first.", {"early": early, "late": late})
        elif late >= 0.01 and late >= 10 * max(early, 1e-12):
            add("medium", "causality", f"error grows {late / max(early, 1e-12):.0f}× along time ({early:.2g} → {late:.2g})",
                "error_profile: first bin vs last quarter",
                "Accuracy degrades with time even though the solution is qualitatively right; "
                "causal weighting or time marching usually recovers the late-time accuracy.",
                {"early": early, "late": late})

    # ── landscape ───────────────────────────────────────────────────────────
    cv = res.get("curvature")
    if cv:
        lm = cv.get("lambda_max")
        if lr is not None and lm is not None and lm * lr >= 2:
            add("high", "landscape", f"λ_max·lr = {lm * lr:.3g} ≥ 2",
                f"λ_max={_f(lm)}, lr={lr:g}",
                "Gradient descent is beyond its stability limit along the sharpest direction (Adam "
                "masks this partly). Lower the learning rate, add warm-up, or decay sooner.")
        k10 = cv.get("kappa_top")
        if k10 is not None and np.isfinite(k10) and k10 >= 1e3:
            kj = cv.get("kappa_top_jacobi")
            add("medium", "landscape", f"κ₁₀ = {k10:.3g}: ill-conditioned top of the spectrum",
                f"κ₁₀={_f(k10)}" + (f", Jacobi-scaled {_f(kj)}" if kj is not None else ""),
                "First-order methods crawl along flat directions. A second-order phase (L-BFGS / "
                "natural gradient) or a better-conditioned representation helps"
                + ("; diagonal scaling does not (κ stays large after Jacobi)" if kj is not None and kj >= 1e3 else "") + ".")

    # ── kernel ──────────────────────────────────────────────────────────────
    nk = res.get("ntk")
    if nk and "eigs" in nk:
        n = len(nk["eigs"])
        rk = nk.get("rank_1e-3", None)
        if rk is not None and n >= 20 and rk <= 0.05 * n:
            add("medium", "kernel", f"residual NTK has effective rank {rk} of {n}",
                "eigenvalues above 1e-3·λ₁",
                "Strong spectral bias: only a few residual modes train at a useful rate. Fourier "
                "features / periodic embeddings or a wider first layer raise the rank.")
        if nk.get("drift") is not None and nk["drift"] <= 0.05:
            add("info", "kernel", f"NTK drift {nk['drift']:.3f}: lazy (kernel) regime",
                "‖K−K₀‖/‖K₀‖", "Features are barely learned; accuracy is bounded by the initial "
                "kernel. A smaller init scale or larger learning rate moves into feature learning.")
        if nk.get("trace_ratio") is not None and nk["trace_ratio"] <= 0.05:
            add("medium", "kernel", f"NTK trace fell to {nk['trace_ratio']:.3f} of its initial value",
                "tr(K)/tr(K₀)", "The kernel collapsed: gradients w.r.t. the residual have vanished "
                "(saturation or a degenerate representation), so training stalls.")

    # ── sampling ────────────────────────────────────────────────────────────
    og = res.get("offgrid")
    if og:
        ratio = og.get("ratio") if isinstance(og, dict) else og
        if ratio is not None and ratio >= 3:
            add("high", "sampling", f"residual off the collocation points is {ratio:.1f}× higher",
                "RMS(dense) / RMS(collocation)",
                "Overfitting the collocation set. Use more points, resample every few epochs, or "
                "residual-adaptive sampling.")
    rc, rec = res.get("residual_concentration"), res.get("residual_error_correlation")
    if rc and rec and rc["top_share"] >= 0.5 and rec["spearman"] >= 0.5:
        add("medium", "sampling", f"{rc['top_share']:.0%} of the residual is in 10 % of the points, "
            f"and it tracks the error (Spearman {rec['spearman']:.2f})",
            f"Gini={_f(rc['gini'], '.2f')}",
            "Residual-adaptive sampling (RAR-D, resample_period>0) will put points where they matter.")
    if rc and (rc.get("max_over_median", 0) >= 1e3 or rc["top_share"] >= 0.95):
        wp = rc.get("worst_points")
        where = f" near {np.round(np.asarray(wp[0], float), 3).tolist()}" if wp else ""
        add("high", "sampling", f"the residual is concentrated at a point{where} "
            f"(max/median |r| = {rc.get('max_over_median', float('nan')):.3g}, top-10 % share {rc['top_share']:.2f})",
            "residual_concentration['worst_points']",
            "One location dominates the loss. If the solution is smooth there, suspect a singularity "
            "in the network's derivatives (e.g. a normalisation layer acting on near-zero "
            "activations, a kink, a division) or in the data at that point; adding collocation "
            "points will only make the optimiser chase it.", {"worst_points": wp})
    if rec and rec["spearman"] <= 0.2:
        add("medium" if failing else "info", "sampling",
            f"the residual does not indicate the error (Spearman {rec['spearman']:.2f})",
            "rank correlation of |r| and |e|" + (f"; relative error {err:.2g}" if err is not None else ""),
            "Residual-driven refinement would target the wrong places; the error is set by "
            "boundary/initial data or propagation, not by local residual."
            + ("" if failing else " (Low correlation is common once both are small.)"))

    # ── output layer / representation ───────────────────────────────────────
    orf = res.get("output_refit")
    if orf and orf.get("ratio") is not None and orf["ratio"] >= 2:    # ratio = loss before / after
        add("medium", "output layer", f"least-squares refit of the last layer lowers the loss {orf['ratio']:.2f}×",
            "loss / loss(refit)", "Adam had not converged the output layer: a periodic LS refit "
            "(hybrid LS + Adam) is a cheap accuracy gain.")
    hid = res.get("hidden")
    if hid:
        sats = [v["saturation"] for v in hid.values() if "saturation" in v]
        if sats and np.mean(sats) >= 0.3:
            add("medium", "representation", f"mean unit saturation {np.mean(sats):.2f}",
                "fraction |z|>2", "Units sit in the flat part of tanh: rescale inputs to ~[-1,1] or "
                "reduce the init scale.")
        low = [k for k, v in hid.items() if v.get("width") and v.get("erank", 1e9) <= 0.25 * v["width"]]
        if low:
            add("info", "representation", f"{len(low)} layer(s) use ≤25 % of their width",
                ", ".join(f"{k}: erank {hid[k]['erank']:.1f}/{hid[k]['width']}" for k in low[:4]),
                "Redundant width; refit-pruning can confirm a narrower network suffices.")

    # ── spectrum ────────────────────────────────────────────────────────────
    se = res.get("spectral_error")
    if se and se.get("k") is not None:
        k = np.asarray(se["k"])
        sig = np.isfinite(np.asarray(se["ratio"], np.float64))
        if sig.any():
            kband = k[sig].max()
            if se["resolved_k"] < 0.25 * kband:
                what = (f"no wavenumber band is reproduced within {se['tol']:.0%}" if se["resolved_k"] <= 0
                        else f"only wavenumbers up to {se['resolved_k']:.3g} are resolved")
                add("medium", "spectrum", f"{what} (solution has content to {kband:.3g})",
                    f"|E(k)|/|U(k)| < {se['tol']} below resolved_k",
                    "Spectral bias limits accuracy: Fourier features with σ near the missing band, "
                    "or a multiscale / domain-decomposed (FBPINN) network.")

    out.sort(key=lambda f: _SEV[f.severity])
    return out


def to_markdown(findings: List[Finding], title: str = "Diagnostic findings") -> str:
    if not findings:
        return f"## {title}\n\nNo issues flagged by the rules.\n"
    lines = [f"## {title}", ""]
    for f in findings:
        lines += [f"- **[{f.severity}] {f.area}: {f.finding}**",
                  f"  - evidence: {f.evidence}",
                  f"  - action: {f.action}"]
    return "\n".join(lines) + "\n"
