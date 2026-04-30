"""
Paper IV — Shapiro delay through R(r) of Paper II
====================================================

Time of flight along a path from x = -X_E to x = +X_M at impact
parameter b, with refractive index n(r) = (R_0/R)^alpha and
R(r)/R_0 = sqrt(1 - beta/r).

The lab-frame time elapsed is
    T_path = (1/c) integral_path n(r) ds

For a straight-line ray (paraxial) at impact parameter b, ds = dx and
r = sqrt(x^2 + b^2). The flat-space time is T_flat = (X_E + X_M) / c.
The Shapiro delay is  Delta_T = T_path - T_flat = (1/c) integral (n-1) dx.

Under the weak-field expansion, n - 1 = alpha beta / (2 r) + O(beta^2/r^2),
giving the analytic prediction
    c * Delta_T ~ alpha beta * ln[(X_E + r_E)(X_M + r_M) / b^2]
where r_X = sqrt(X^2 + b^2). For X >> b, this is
    c * Delta_T ~ 2 alpha beta * ln(2 X / b)

GR Shapiro: c * Delta_T = 4 GM / c^2 * ln(...) = 2 beta * ln(...).
So:
    alpha = 1  -> half GR
    alpha = 2  -> full GR
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)


def n_of_r(r, beta, alpha, chi_floor=1e-3):
    arg = max(1.0 - beta / max(r, 1e-9), chi_floor**2)
    chi = np.sqrt(arg)
    return chi ** (-alpha)


def shapiro(beta, b, X_E, X_M, alpha, dx=0.05):
    """Compute integral of (n-1) along x in [-X_E, X_M] at impact parameter b."""
    xs = np.arange(-X_E, X_M, dx)
    delta_T = 0.0
    for x in xs:
        r = np.sqrt(x*x + b*b)
        delta_T += (n_of_r(r, beta, alpha) - 1.0) * dx
    return float(delta_T)


def main():
    beta = 1.0
    X_E = X_M = 1000.0
    bs = [3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]

    print(f"beta = {beta}, X_E = X_M = {X_E}")
    print(f"GR prediction: c*Delta_T = beta * ln[(r_E+X_E)(r_M+X_M) / b^2]")
    print()
    print(f"{'b':>6} {'b/beta':>8} "
          f"{'alpha=1':>12} {'alpha=2':>12} "
          f"{'GR pred':>12} {'rel_err alpha=2':>16}")

    rows = []
    for b in bs:
        d1 = shapiro(beta, b, X_E, X_M, alpha=1)
        d2 = shapiro(beta, b, X_E, X_M, alpha=2)
        # GR prediction: full formula valid for finite X
        r_E = np.sqrt(X_E*X_E + b*b)
        r_M = np.sqrt(X_M*X_M + b*b)
        gr_pred = beta * np.log((r_E + X_E) * (r_M + X_M) / b**2)
        err2 = (d2 - gr_pred) / gr_pred
        print(f"{b:6.1f} {b/beta:8.2f} {d1:12.4f} {d2:12.4f} {gr_pred:12.4f} {err2:16.2%}")
        rows.append({
            "b": b, "b_over_beta": b / beta,
            "shapiro_alpha1": d1,
            "shapiro_alpha2": d2,
            "shapiro_GR_pred": float(gr_pred),
            "rel_err_alpha2": float(err2),
        })

    out = {"beta": beta, "X_E": X_E, "X_M": X_M, "results": rows}
    with open(DATA / "02_shapiro_delay.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {DATA / '02_shapiro_delay.json'}")


if __name__ == "__main__":
    main()
