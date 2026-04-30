"""
Paper IV — light deflection by ray-tracing through R(r) of Paper II
=====================================================================

Background profile (from Paper II, single-throttle mobility):
    R(r)/R_0 = sqrt(1 - beta/r),   beta = kappa E_0 / (2 pi alpha R_0).

We treat the photon phenomenologically with a coordinate-speed ansatz
    c_eff(r) = c * (R/R_0)^alpha,
yielding a refractive index n(r) = c/c_eff = (R_0/R)^alpha.

Two ansatze are tested:
  alpha = 1: temporal only (clock factor only)         -> deflection = beta/b
  alpha = 2: temporal + spatial (combined GR analogue) -> deflection = 2*beta/b
                                                       = 4 GM / c^2 b

Eikonal equation:  dk/ds = grad(ln n),  with k unit vector along ray.
Initial condition: photon enters from x=-X with k=(+1, 0) at impact
parameter b (i.e. y = b).

Output: deflection angle in the asymptotic region; comparison to GR.
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

DS = 0.05
X_START = -800.0
X_END = +800.0
CHI_FLOOR = 1e-3  # avoid R/R_0 below this (bandwidth horizon regularisation)


def chi(r, beta):
    """R(r)/R_0 with horizon regularisation."""
    arg = 1.0 - beta / max(r, 1e-9)
    return float(np.sqrt(max(arg, CHI_FLOOR**2)))


def grad_log_n(x, y, beta, alpha):
    """grad(ln n) where n = chi^{-alpha}.
    ln n = -alpha ln chi
    chi = sqrt(1 - beta/r), so ln chi = 0.5 ln(1 - beta/r)
    grad ln chi = 0.5 / (1 - beta/r) * (beta/r^3) * (x,y)
    """
    r = np.sqrt(x*x + y*y)
    if r < 1e-6:
        return np.array([0.0, 0.0])
    arg = max(1.0 - beta / r, CHI_FLOOR**2)
    factor = -alpha * 0.5 / arg * (beta / r**3)
    return np.array([factor * x, factor * y])


def trace_ray(beta, b, alpha):
    """Trace one ray with impact parameter b under ansatz alpha; return deflection."""
    x = X_START
    y = b
    kx = 1.0
    ky = 0.0
    n_steps = 0
    while x < X_END and n_steps < 50000:
        g = grad_log_n(x, y, beta, alpha)
        kx += g[0] * DS
        ky += g[1] * DS
        norm = np.sqrt(kx*kx + ky*ky)
        kx /= norm; ky /= norm
        x += kx * DS
        y += ky * DS
        n_steps += 1
    deflection = float(np.arctan2(ky, kx))
    return abs(deflection), n_steps


def main():
    beta = 1.0
    bs = [2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]

    print(f"beta = {beta}")
    print(f"GR prediction: deflection = 4 G M / c^2 b = 2 beta / b")
    print()
    print(f"{'b':>6} {'b/beta':>8} "
          f"{'alpha=1 (temp)':>16} {'alpha=2 (full)':>16} "
          f"{'GR pred (2b/b)':>16} {'rel_err alpha=2':>16}")

    rows = []
    for b in bs:
        d1, n1 = trace_ray(beta, b, alpha=1)
        d2, n2 = trace_ray(beta, b, alpha=2)
        gr_pred = 2.0 * beta / b
        err2 = (d2 - gr_pred) / gr_pred
        print(f"{b:6.1f} {b/beta:8.2f} {d1:16.6f} {d2:16.6f} "
              f"{gr_pred:16.6f} {err2:16.2%}")
        rows.append({
            "b": b, "b_over_beta": b / beta,
            "deflection_alpha1": d1,
            "deflection_alpha2": d2,
            "deflection_GR_pred": gr_pred,
            "rel_err_alpha2": err2,
            "n_steps_alpha1": n1, "n_steps_alpha2": n2,
        })

    out = {"beta": beta, "DS": DS,
           "X_START": X_START, "X_END": X_END,
           "results": rows}
    with open(DATA / "01_photon_deflection.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved data to {DATA / '01_photon_deflection.json'}")


if __name__ == "__main__":
    main()
