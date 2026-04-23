# Rebuttal Error Decomposition

These results were generated for the response to Reviewer oB6r.

## Setup

- Numerical precision: NumPy complex128 for Fourier/state quantities and float64 for density fields and norms.
- Table 1 reuses the V=0 sanity configuration from `figures/plot_v0_sanity.py`: `N=64`, `sigma=3.0`, `K0=6`, `t=0.30*pi`.
- Table 2 reuses the paper's standard single-cosine family from `experiments/run_sweep.py`: canonical seed `0`, `nx=5`, `q=(1,0)`, `t=0.5`, and `alpha in {0.2, 0.5, 1.0}`.
- Table 3 reuses the structured dense rebuttal family from `experiments/run_rebuttal_stress_tests.py` with `J=8`, fixed `alpha_total=0.5`, `K0=5`, and `R_hops in {1, 3}`.
- For all rows, `rho_full_lp` is reconstructed from exact full-state coefficients truncated to the same task cutoff `K0`; no method definition was changed.

## Auto Summary

In the evaluated regime for Reviewer oB6r, the exact-closure sanity check gives machine-precision shadow error (eps_shadow = 1.844839e-16), so eps_total matches eps_cutoff up to numerical precision. Across the main V != 0 single-cosine decomposition table, the maximum eps_shadow is 0.010831 and the maximum shadow_fraction is 0.19854, while the maximum eps_shadow / eps_cutoff ratio is 0.195505. This supports the rebuttal claim that, in the tested regime, the total density error is primarily driven by the cutoff K0 and the additional shadow contribution remains secondary. In the optional dense structured case, eps_shadow stays small (max 2.979840e-04) and shadow_fraction stays below 0.010003.

Across the main V != 0 rows, the maximum triangle-gap diagnostic `eps_cutoff + eps_shadow - eps_total` is 0.011678. This confirms that the decomposition is a useful diagnostic split, but not an exact additive identity.

## Table 1: Exact-Closure Sanity Check

| Setting | eps_cutoff | eps_shadow | eps_total | Observation |
| --- | --- | --- | --- | --- |
| V=0, N=64, K0=6, t=0.30*pi | 0.027933 | 1.844839e-16 | 0.027933 | Machine-precision shadow error; total error matches cutoff error. |

## Table 2: Main V != 0 Decomposition

| alpha | K0 | eps_cutoff | eps_shadow | eps_total | shadow_fraction | DeltaZ_F | leakage_l_rms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 4 | 0.052175 | 2.784490e-03 | 0.051352 | 0.054223 | 3.663143e-03 | 0.101686 |
| 0.2 | 6 | 0.020282 | 3.503918e-04 | 0.020257 | 0.017298 | 3.083589e-03 | 0.094697 |
| 0.2 | 8 | 6.038098e-03 | 7.901769e-05 | 6.038023e-03 | 0.013087 | 3.076745e-03 | 0.091611 |
| 0.5 | 4 | 0.054271 | 6.277964e-03 | 0.052622 | 0.119304 | 9.159336e-03 | 0.254216 |
| 0.5 | 6 | 0.020021 | 8.863254e-04 | 0.020004 | 0.044308 | 7.743329e-03 | 0.236743 |
| 0.5 | 8 | 5.896010e-03 | 1.995179e-04 | 5.897455e-03 | 0.033831 | 7.726492e-03 | 0.229027 |
| 1 | 4 | 0.0554 | 0.010831 | 0.054553 | 0.19854 | 0.018804 | 0.508432 |
| 1 | 6 | 0.019524 | 1.834565e-03 | 0.019643 | 0.093397 | 0.016161 | 0.473486 |
| 1 | 8 | 5.698843e-03 | 4.071193e-04 | 5.705852e-03 | 0.071351 | 0.016129 | 0.458054 |

## Table 3: Dense Structured Case Decomposition

| setting | eps_cutoff | eps_shadow | eps_total | shadow_fraction | DeltaZ_F | leakage_l_rms |
| --- | --- | --- | --- | --- | --- | --- |
| J=8, alpha_total=0.5, K0=5, R_hops=1 | 0.029793 | 2.979840e-04 | 0.029789 | 0.010003 | 3.565241e-03 | 0.101099 |
| J=8, alpha_total=0.5, K0=5, R_hops=3 | 0.029793 | 2.979840e-04 | 0.029789 | 0.010003 | 4.784983e-04 | 0.077749 |

## Dense-Case Note

In the current implementation, changing `R_hops` affects the reference set used for the certificate (`DeltaZ_F` and `leakage_l_rms`), but the density reconstruction itself is determined by the retained task subspace `K`. Accordingly, the dense-case decomposition rows share the same density metrics at fixed `K0` while still showing improved certificate quantities as the reference set grows.
