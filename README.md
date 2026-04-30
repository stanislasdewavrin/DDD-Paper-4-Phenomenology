# Paper IV — Gravitational phenomenology

**Title:** Discrete Drainage Dynamics IV: Light Deflection, Shapiro Delay,
and the Photon-Substrate Coupling in the Static Spherical Sector

## What this paper does

Tests whether the static spherical bandwidth profile of Paper II,
`R(r)/R_0 = sqrt(1 - β/r)`, reproduces classical gravitational
propagation observables (deflection, Shapiro delay) under a
phenomenological photon-substrate coupling
`c_eff(r) = c·(R/R_0)^α`. Two ansatze:

- α=1 (temporal only): predicts half of GR — inconsistent with data
- α=2 (temporal + spatial): predicts full GR `4GM/c²b` — matches at <1%

α=2 is identified as the calibration consistent with weak-field GR.
Its derivation from the underlying lattice rule is open.

## Code

- `code/01_photon_deflection.py` — eikonal ray-tracing
- `code/02_shapiro_delay.py` — time-of-flight integral
- `code/03_make_figures.py` — figure generation

## Build

```
make
```
