# Rebuttal Initial-Condition Robustness Check

These results were generated for the response to Reviewer gBE6.

## Setup

- Numerical precision: NumPy complex128 for state/Fourier quantities and float64 for density fields and norms.
- Base simulation setting: `nx=5`, `N=32`, `alpha=0.5`, `q=(1,0)`, `t=0.5`, `R_hops=1`, and canonical simulation seed `0`.
- Tested cutoffs: `K0 in {6, 8}`.
- `rho_full_lp` is always reconstructed from exact full-state coefficients truncated to the same task cutoff `K0`; the ShadowFluid method itself is unchanged.
- Initial conditions:
  - Gaussian vortex: reused `cases.vortex_case(nx=5, seed=0)`.
  - Multi-scale packet: deterministic two-bump spatial packet with two widths and fixed phase factors.
  - Random band-limited state: deterministic random Fourier coefficients inside `||k|| <= 10` with fixed seed `20260417`.

## Auto Summary

For Reviewer gBE6, we performed a compact initial-condition robustness check under the standard single-cosine benchmark (alpha=0.5, q=(1,0), t=0.5, R_hops=1). The qualitative hierarchy is preserved across all tested initial conditions: eps_shadow remains smaller than eps_cutoff in every row. The maximum observed eps_shadow is 0.020003, the maximum shadow_fraction is 0.049337, and the maximum eps_shadow / eps_cutoff ratio is 0.049211. The largest relative shadow contribution occurs for Random band-limited state at K0=8, but the cutoff term still dominates in that case.

## Main Table

| Initial condition | K0 | eps_cutoff | eps_shadow | eps_total | shadow_fraction | DeltaZ_F | leakage_l_rms | Observation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian vortex | 6 | 0.020021 | 8.863254e-04 | 0.020004 | 0.044308 | 7.743329e-03 | 0.236743 | Cutoff-dominated; shadow contribution remains small. |
| Gaussian vortex | 8 | 5.896010e-03 | 1.995179e-04 | 5.897455e-03 | 0.033831 | 7.726492e-03 | 0.229027 | Cutoff-dominated; shadow contribution remains small. |
| Multi-scale packet | 6 | 0.026353 | 8.322385e-04 | 0.026458 | 0.031455 | 0.015687 | 0.236743 | Cutoff-dominated; shadow contribution remains small. |
| Multi-scale packet | 8 | 0.023255 | 4.684063e-04 | 0.02324 | 0.020156 | 0.015684 | 0.229027 | Cutoff-dominated; shadow contribution remains small. |
| Random band-limited state | 6 | 0.59462 | 0.020003 | 0.593415 | 0.033708 | 8.606715e-03 | 0.236743 | Cutoff-dominated; shadow contribution remains small. |
| Random band-limited state | 8 | 0.367524 | 0.018086 | 0.366591 | 0.049337 | 9.684881e-03 | 0.229027 | Cutoff-dominated; shadow contribution remains small. |

## Construction Notes

- The Gaussian vortex row reuses the paper/codebase default family rather than introducing a new state definition.
- The multi-scale packet is intentionally simple and deterministic, so that robustness is tested against a qualitatively different spatial structure without changing the simulation pipeline.
- The random band-limited state is deterministic through its fixed seed and uses only the existing FFT conventions already present in `shiftflow/core_v0.py`.
