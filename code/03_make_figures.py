"""Paper IV figures."""
import json, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

# ============================================================
# Figure 1: schematic of light passing through R(r) profile
# ============================================================
print("Fig 1: schematic of ray through R(r)...")
fig, ax = plt.subplots(figsize=(8, 5))
xs = np.linspace(-50, 50, 400)
ys = np.linspace(-30, 30, 240)
X, Y = np.meshgrid(xs, ys)
R = np.sqrt(X**2 + Y**2)
beta = 5.0
chi = np.where(R > beta, np.sqrt(np.maximum(1 - beta/np.maximum(R, 1e-3), 0)), 0)
im = ax.contourf(X, Y, chi, levels=30, cmap='viridis')
plt.colorbar(im, ax=ax, label=r'$R(r)/R_0$')
# Show a few rays at different impact parameters
for b, color in zip([8, 12, 18, 25], ['red', 'orange', 'yellow', 'white']):
    ax.axhline(b, xmin=0.0, xmax=0.45, color=color, lw=1.2,
               linestyle='--', alpha=0.7)
    # Approximate refraction trajectory: small bend at the gravity zone
    x_pre = np.linspace(-50, -10, 50)
    y_pre = np.full_like(x_pre, b)
    x_zone = np.linspace(-10, 10, 50)
    # Smooth bending
    delta = 1.5 / b  # not quantitative; just illustrative
    y_zone = b - delta * np.sin(np.pi * (x_zone + 10) / 20) * 5
    x_post = np.linspace(10, 50, 50)
    y_post = (b - delta * 5) + (x_post - 10) * (-delta * 5 / 40)  # exit slope
    ax.plot(x_pre, y_pre, color=color, lw=1.5)
    ax.plot(x_zone, y_zone, color=color, lw=1.5)
    ax.plot(x_post, y_post, color=color, lw=1.5)
    ax.annotate(rf'$b={b}$', (49, y_post[-1]), color=color, fontsize=9,
                ha='right', va='bottom')

ax.plot(0, 0, 'r*', ms=14, label='source (mass)')
# Mark the bandwidth horizon
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(beta * np.cos(theta), beta * np.sin(theta), 'r--', lw=1,
        label=rf'$r = \beta = {beta}$ (horizon)')
ax.set_aspect('equal')
ax.set_xlim(-50, 50); ax.set_ylim(-30, 30)
ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')
ax.set_title(r'Light rays at impact parameter $b$ through the bandwidth profile $R(r)/R_0 = \sqrt{1-\beta/r}$')
ax.legend(loc='lower right', fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig1_schematic.pdf", bbox_inches='tight')
fig.savefig(FIG / "fig1_schematic.png", dpi=150, bbox_inches='tight')
plt.close(fig)


# ============================================================
# Figure 2: deflection vs b/beta, two ansatze + GR
# ============================================================
print("Fig 2: deflection vs b/beta...")
with open(DATA / "01_photon_deflection.json") as f:
    d = json.load(f)
res = d["results"]
b_over_beta = np.array([r["b_over_beta"] for r in res])
d1 = np.array([r["deflection_alpha1"] for r in res])
d2 = np.array([r["deflection_alpha2"] for r in res])
gr = np.array([r["deflection_GR_pred"] for r in res])

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.loglog(b_over_beta, d2, 'ro-', ms=8, lw=1.5,
          label=r'$\alpha = 2$ (temporal $+$ spatial)')
ax.loglog(b_over_beta, d1, 'bs-', ms=6, lw=1.2,
          label=r'$\alpha = 1$ (temporal only)')
ax.loglog(b_over_beta, gr, 'k--', lw=1.5,
          label=r'GR: $4 G M / c^2 b = 2 \beta / b$')
ax.set_xlabel(r'$b / \beta$')
ax.set_ylabel(r'deflection angle $\Delta \theta$  [rad]')
ax.set_title('Light deflection through the static spherical profile')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(FIG / "fig2_deflection.pdf", bbox_inches='tight')
fig.savefig(FIG / "fig2_deflection.png", dpi=150, bbox_inches='tight')
plt.close(fig)


# ============================================================
# Figure 3: Shapiro delay vs b/beta
# ============================================================
print("Fig 3: Shapiro delay...")
with open(DATA / "02_shapiro_delay.json") as f:
    d = json.load(f)
res = d["results"]
b_over_beta = np.array([r["b_over_beta"] for r in res])
s1 = np.array([r["shapiro_alpha1"] for r in res])
s2 = np.array([r["shapiro_alpha2"] for r in res])
gr = np.array([r["shapiro_GR_pred"] for r in res])

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.semilogx(b_over_beta, s2, 'ro-', ms=8, lw=1.5,
            label=r'$\alpha = 2$ (full GR)')
ax.semilogx(b_over_beta, s1, 'bs-', ms=6, lw=1.2,
            label=r'$\alpha = 1$ (half GR)')
ax.semilogx(b_over_beta, gr, 'k--', lw=1.5,
            label=r'GR Shapiro $\beta \ln[(r_E+X_E)(r_M+X_M)/b^2]$')
ax.set_xlabel(r'$b / \beta$')
ax.set_ylabel(r'$c \cdot \Delta T_{\rm Shapiro}$  [length units, $\beta = 1$]')
ax.set_title(r'Shapiro logarithmic time delay through $R(r)$')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(FIG / "fig3_shapiro.pdf", bbox_inches='tight')
fig.savefig(FIG / "fig3_shapiro.png", dpi=150, bbox_inches='tight')
plt.close(fig)


# ============================================================
# Figure 4: relative deviation alpha=2 from GR for both observables
# ============================================================
print("Fig 4: relative deviations...")
with open(DATA / "01_photon_deflection.json") as f:
    d_def = json.load(f)
with open(DATA / "02_shapiro_delay.json") as f:
    d_sh = json.load(f)

b_def = np.array([r["b_over_beta"] for r in d_def["results"]])
err_def = np.array([100 * r["rel_err_alpha2"] for r in d_def["results"]])

b_sh = np.array([r["b_over_beta"] for r in d_sh["results"]])
err_sh = np.array([100 * r["rel_err_alpha2"] for r in d_sh["results"]])

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.semilogx(b_def, err_def, 'ro-', ms=7, lw=1.2, label='deflection')
ax.semilogx(b_sh, err_sh, 'bs-', ms=7, lw=1.2, label='Shapiro')
ax.axhline(0, color='k', ls=':', alpha=0.5)
ax.fill_between([1, 1000], -1, 1, color='green', alpha=0.1,
                label=r'$|err| < 1\%$')
ax.fill_between([1, 1000], -10, 10, color='gold', alpha=0.1,
                label=r'$|err| < 10\%$')
ax.set_xlabel(r'$b / \beta$')
ax.set_ylabel(r'$(\alpha=2 \text{ measured} - \text{GR}) / \text{GR}$  [%]')
ax.set_title(r'Convergence of $\alpha = 2$ ansatz to weak-field GR')
ax.set_xlim(1, 100)
ax.set_ylim(-50, 200)
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(FIG / "fig4_convergence.pdf", bbox_inches='tight')
fig.savefig(FIG / "fig4_convergence.png", dpi=150, bbox_inches='tight')
plt.close(fig)

print("All figures saved to:", FIG)


# ============================================================
# Figure 5: refractive index n(r) for both ansatze
# ============================================================
print("Fig 5: refractive index n(r)...")
beta = 1.0
r_over_beta = np.linspace(1.05, 10, 400)
chi = np.sqrt(1.0 - 1.0/r_over_beta)
n1 = 1.0 / chi          # alpha = 1 (n = R0/R)
n2 = 1.0 / (chi**2)     # alpha = 2 (n = R0^2/R^2)

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(r_over_beta, n2, 'r-', lw=2, label=r'$\alpha = 2$: $n(r) = (R_0/R)^2 = 1/(1 - \beta/r)$')
ax.plot(r_over_beta, n1, 'b-', lw=2, label=r'$\alpha = 1$: $n(r) = R_0/R = 1/\sqrt{1 - \beta/r}$')
ax.axhline(1.0, color='k', ls=':', alpha=0.5, label=r'flat space ($n = 1$)')
ax.axvline(1.0, color='gray', ls='--', alpha=0.5,
           label=r'horizon $r = \beta$ (eikonal breaks down)')
ax.set_xlabel(r'$r/\beta$')
ax.set_ylabel(r'refractive index $n(r)$')
ax.set_title(r'Photon refractive index for the two coupling ansatze')
ax.set_xlim(0.5, 10)
ax.set_ylim(0.9, 6)
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig5_refractive_index.pdf", bbox_inches='tight')
fig.savefig(FIG / "fig5_refractive_index.png", dpi=150, bbox_inches='tight')
plt.close(fig)


# ============================================================
# Figure 6: effective metric components g_tt and g_rr
# ============================================================
print("Fig 6: effective metric components...")
r_over_beta = np.linspace(1.05, 10, 400)
chi2 = 1.0 - 1.0/r_over_beta  # = (R/R_0)^2

# alpha = 1: g_tt = -(R/R_0)^2 = -chi^2 = -(1 - beta/r), g_rr = +1
gtt_1 = -chi2
grr_1 = np.ones_like(chi2)

# alpha = 2: g_tt = -(1 - beta/r), g_rr = +1/(1 - beta/r) (Schwarzschild)
gtt_2 = -chi2
grr_2 = 1.0 / chi2

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

ax = axes[0]
ax.plot(r_over_beta, gtt_2, 'r-', lw=2, label=r'$\alpha=2$: $g_{tt} = -(1-\beta/r)$')
ax.plot(r_over_beta, gtt_1, 'b--', lw=2, label=r'$\alpha=1$: $g_{tt} = -(1-\beta/r)$')
ax.axhline(-1.0, color='k', ls=':', alpha=0.5, label=r'flat ($g_{tt} = -1$)')
ax.axvline(1.0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel(r'$r/\beta$'); ax.set_ylabel(r'$g_{tt}$')
ax.set_title(r'Time component (same for both ansatze)')
ax.set_xlim(0.5, 10); ax.set_ylim(-1.05, 0.05)
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(r_over_beta, grr_2, 'r-', lw=2, label=r'$\alpha=2$: $g_{rr} = 1/(1-\beta/r)$ (Schwarzschild)')
ax.plot(r_over_beta, grr_1, 'b-', lw=2, label=r'$\alpha=1$: $g_{rr} = 1$ (flat radial)')
ax.axhline(1.0, color='k', ls=':', alpha=0.5)
ax.axvline(1.0, color='gray', ls='--', alpha=0.5,
           label=r'horizon')
ax.set_xlabel(r'$r/\beta$'); ax.set_ylabel(r'$g_{rr}$')
ax.set_title(r'Radial component (where the two ansatze differ)')
ax.set_xlim(0.5, 10); ax.set_ylim(0, 6)
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle(r'Effective metric components implied by the two photon-substrate ansatze',
             fontsize=11, y=1.04)
fig.savefig(FIG / "fig6_metric_components.pdf", bbox_inches='tight')
fig.savefig(FIG / "fig6_metric_components.png", dpi=150, bbox_inches='tight')
plt.close(fig)

print("Figures 5 and 6 saved.")
