"""Exact solution of the 1-D compressible-Euler Riemann problem.

Standard exact Riemann solver (Toro, *Riemann Solvers and Numerical Methods
for Fluid Dynamics*): Newton iteration for the star-region pressure, then a
self-similar sampling of the rarefaction / contact / shock structure.  Used by
the Sod and Toro shock-tube examples to score the PINN against ground truth.
"""
from __future__ import annotations

import numpy as np


def exact_riemann_1d(x, t, x0, gamma, left, right):
    """Exact solution of the 1-D Euler Riemann problem at time *t*.

    Parameters
    ----------
    x     : array of positions.
    t     : evaluation time (scalar > 0).
    x0    : diaphragm location.
    gamma : ratio of specific heats.
    left  : (ρ_L, u_L, p_L) ;  right : (ρ_R, u_R, p_R).

    Returns
    -------
    (rho, u, p) arrays the same shape as *x*.
    """
    rhoL, uL, pL = left
    rhoR, uR, pR = right
    g  = gamma
    cL = np.sqrt(g * pL / rhoL)
    cR = np.sqrt(g * pR / rhoR)

    G1 = (g - 1.0) / (2.0 * g)
    G2 = (g + 1.0) / (2.0 * g)
    G3 = 2.0 * g / (g - 1.0)
    G4 = 2.0 / (g - 1.0)
    G5 = 2.0 / (g + 1.0)
    G6 = (g - 1.0) / (g + 1.0)
    G7 = (g - 1.0) / 2.0

    def fk(p, rhoK, pK, cK):
        if p > pK:                                  # shock
            Ak = G5 / rhoK
            Bk = G6 * pK
            return (p - pK) * np.sqrt(Ak / (p + Bk))
        return G4 * cK * ((p / pK) ** G1 - 1.0)     # rarefaction

    def dfk(p, rhoK, pK, cK):
        if p > pK:
            Ak = G5 / rhoK
            Bk = G6 * pK
            return np.sqrt(Ak / (p + Bk)) * (1.0 - 0.5 * (p - pK) / (p + Bk))
        return (1.0 / (rhoK * cK)) * (p / pK) ** (-G2)

    # Solve for star-region pressure by Newton iteration
    p = max(1e-6, 0.5 * (pL + pR))
    for _ in range(100):
        fval = fk(p, rhoL, pL, cL) + fk(p, rhoR, pR, cR) + (uR - uL)
        dval = dfk(p, rhoL, pL, cL) + dfk(p, rhoR, pR, cR)
        p_new = p - fval / dval
        if p_new <= 0.0:
            p_new = 1e-6
        if abs(p_new - p) < 1e-12:
            p = p_new
            break
        p = p_new
    pstar = p
    ustar = 0.5 * (uL + uR) + 0.5 * (fk(pstar, rhoR, pR, cR)
                                     - fk(pstar, rhoL, pL, cL))

    x   = np.asarray(x, dtype=float)
    rho = np.empty_like(x)
    u   = np.empty_like(x)
    pr  = np.empty_like(x)

    for i, xi in enumerate(x):
        S = (xi - x0) / t if t > 0 else 0.0
        if S <= ustar:                              # left of contact
            if pstar > pL:                          # left shock
                SL = uL - cL * np.sqrt(G2 * pstar / pL + G1)
                if S <= SL:
                    r, uu, pp = rhoL, uL, pL
                else:
                    r  = rhoL * ((pstar / pL + G6) / (G6 * pstar / pL + 1.0))
                    uu, pp = ustar, pstar
            else:                                   # left rarefaction
                SHL    = uL - cL
                cLstar = cL * (pstar / pL) ** G1
                STL    = ustar - cLstar
                if S <= SHL:
                    r, uu, pp = rhoL, uL, pL
                elif S >= STL:
                    r  = rhoL * (pstar / pL) ** (1.0 / g)
                    uu, pp = ustar, pstar
                else:
                    uu = G5 * (cL + G7 * uL + S)
                    c  = G5 * (cL + G7 * (uL - S))
                    r  = rhoL * (c / cL) ** G4
                    pp = pL * (c / cL) ** G3
        else:                                       # right of contact
            if pstar > pR:                          # right shock
                SR = uR + cR * np.sqrt(G2 * pstar / pR + G1)
                if S >= SR:
                    r, uu, pp = rhoR, uR, pR
                else:
                    r  = rhoR * ((pstar / pR + G6) / (G6 * pstar / pR + 1.0))
                    uu, pp = ustar, pstar
            else:                                   # right rarefaction
                SHR    = uR + cR
                cRstar = cR * (pstar / pR) ** G1
                STR    = ustar + cRstar
                if S >= SHR:
                    r, uu, pp = rhoR, uR, pR
                elif S <= STR:
                    r  = rhoR * (pstar / pR) ** (1.0 / g)
                    uu, pp = ustar, pstar
                else:
                    uu = G5 * (-cR + G7 * uR + S)
                    c  = G5 * (cR - G7 * (uR - S))
                    r  = rhoR * (c / cR) ** G4
                    pp = pR * (c / cR) ** G3
        rho[i], u[i], pr[i] = r, uu, pp

    return rho, u, pr
