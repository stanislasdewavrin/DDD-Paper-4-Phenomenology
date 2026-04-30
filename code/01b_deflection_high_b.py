"""Extended deflection scan: large b/beta with X = 100 * b adaptive,
to verify real convergence to weak-field GR."""
import json, numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

CHI_FLOOR = 1e-3
DS_FRACTION = 0.005  # ds = DS_FRACTION * b -> finer near the deflector


def grad_log_n(x, y, beta, alpha):
    r = np.sqrt(x*x + y*y)
    if r < 1e-6:
        return np.array([0.0, 0.0])
    arg = max(1.0 - beta / r, CHI_FLOOR**2)
    factor = -alpha * 0.5 / arg * (beta / r**3)
    return np.array([factor * x, factor * y])


def trace_ray(beta, b, alpha, X_factor=100.0, ds_factor=DS_FRACTION):
    X = X_factor * b
    ds = ds_factor * b
    x = -X
    y = b
    kx = 1.0
    ky = 0.0
    n_steps = 0
    max_steps = int(2.0 * X / ds) + 1000
    while x < X and n_steps < max_steps:
        g = grad_log_n(x, y, beta, alpha)
        kx += g[0] * ds
        ky += g[1] * ds
        norm = np.sqrt(kx*kx + ky*ky)
        kx /= norm; ky /= norm
        x += kx * ds
        y += ky * ds
        n_steps += 1
    deflection = float(np.arctan2(ky, kx))
    return abs(deflection), n_steps


def main():
    beta = 1.0
    bs = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

    print(f"beta = {beta}")
    print(f"adaptive X = 100*b, ds = {DS_FRACTION}*b")
    print()
    print(f"{'b/beta':>8} "
          f"{'alpha=2 meas':>14} {'GR pred':>14} {'rel_err alpha2':>16}")

    rows = []
    for b in bs:
        d2, n2 = trace_ray(beta, b, alpha=2)
        gr_pred = 2.0 * beta / b
        err2 = (d2 - gr_pred) / gr_pred
        print(f"{b/beta:8.1f} {d2:14.6e} {gr_pred:14.6e} {err2:15.3%}")
        rows.append({"b": b, "b_over_beta": b/beta,
                     "deflection_alpha2": d2,
                     "deflection_GR_pred": gr_pred,
                     "rel_err_alpha2": err2,
                     "n_steps": n2})

    out = {"beta": beta, "X_factor": 100.0, "ds_factor": DS_FRACTION,
           "results": rows}
    with open(DATA / "01b_deflection_high_b.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
