"""Fix Fig 1: replace the cosmetic schematic with real eikonal ray-traces
through R(r) = sqrt(1 - beta/r) profile.

Uses the same integrator as 01_photon_deflection.py but accumulates the
trajectory points so we can plot the actual curved paths.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

PI = np.pi
CHI_FLOOR = 1e-3


def grad_log_n(x, y, beta, alpha):
    r = np.sqrt(x*x + y*y)
    if r < 1e-6:
        return np.array([0.0, 0.0])
    arg = max(1.0 - beta / r, CHI_FLOOR**2)
    factor = -alpha * 0.5 / arg * (beta / r**3)
    return np.array([factor * x, factor * y])


def trace_full(beta, b, alpha, X=80.0, ds=0.05):
    """Trace and return full trajectory (xs, ys)."""
    x = -X; y = b
    kx = 1.0; ky = 0.0
    xs = [x]; ys = [y]
    n_steps = 0
    max_steps = int(2.5 * X / ds)
    while x < X and n_steps < max_steps:
        g = grad_log_n(x, y, beta, alpha)
        kx += g[0] * ds
        ky += g[1] * ds
        norm = np.sqrt(kx*kx + ky*ky)
        kx /= norm; ky /= norm
        x += kx * ds
        y += ky * ds
        xs.append(x); ys.append(y)
        n_steps += 1
    return np.array(xs), np.array(ys)


def main():
    beta = 5.0  # large beta so visible bending in modest field of view
    # Background: R(r)/R_0 over the plot area
    L = 50
    xs_grid = np.linspace(-L, L, 400)
    ys_grid = np.linspace(-30, 30, 240)
    X_g, Y_g = np.meshgrid(xs_grid, ys_grid)
    R = np.sqrt(X_g**2 + Y_g**2)
    chi = np.where(R > beta,
                   np.sqrt(np.maximum(1 - beta / np.maximum(R, 1e-9), 0)),
                   np.nan)  # mask interior

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='#222222')  # interior shown as dark grey
    im = ax.contourf(X_g, Y_g, chi, levels=24, cmap=cmap, vmin=0, vmax=1)
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r'$R(r)/R_0 = \sqrt{1 - \beta/r}$')

    # Real ray-traces at several impact parameters, alpha=2 (full GR)
    impact_params = [8.0, 12.0, 18.0, 25.0]
    colors = ['#ff5555', '#ffaa44', '#ffdd55', '#ffffaa']
    for b, c in zip(impact_params, colors):
        xs, ys = trace_full(beta, b, alpha=2, X=80.0, ds=0.05)
        # Clip to plot region
        mask = (xs >= -L) & (xs <= L)
        ax.plot(xs[mask], ys[mask], color=c, lw=1.8, alpha=0.95)
        # Annotate at right edge
        idx_right = np.argmax(xs[mask])
        ax.annotate(rf'$b = {b:.0f}$',
                    (xs[mask][idx_right], ys[mask][idx_right]),
                    color=c, fontsize=10, ha='right', va='bottom',
                    xytext=(-3, 1), textcoords='offset points')

    # Bandwidth horizon dashed circle
    theta = np.linspace(0, 2*PI, 200)
    ax.plot(beta * np.cos(theta), beta * np.sin(theta),
            color='red', linestyle='--', lw=1.5,
            label=rf'$r = \beta = {beta:.0f}$ (bandwidth horizon)')

    # Mark the source
    ax.plot(0, 0, 'r*', ms=18, mec='black', mew=0.8, label='source (mass)')

    ax.set_aspect('equal')
    ax.set_xlim(-L, L); ax.set_ylim(-30, 30)
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')
    ax.set_title(r'Light rays at impact parameter $b$ through the bandwidth profile  '
                 r'$R(r)/R_0 = \sqrt{1 - \beta/r}$  ($\alpha = 2$ ansatz)')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_schematic.pdf", bbox_inches='tight')
    fig.savefig(FIG / "fig1_schematic.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fixed Fig 1 with real ray-traces at impact parameters {impact_params}")


if __name__ == "__main__":
    main()
