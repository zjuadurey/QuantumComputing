# Feasibility Note

Prepared for Reviewer oB6r.

## Reused Repository Modules

- Full-state evolution: `shiftflow/core_v1.py::evolve_full`, with Hamiltonians from `build_H_dense` and `eigendecompose`.
- Shadow/reduced representation: `shiftflow/core_v1.py::evolve_galerkin`, `build_R_closure`, and the same rank-1 coherence construction `Z = b_K b_R^*` already used in `actual_Z_error`.
- Task-level low-pass inputs: `shiftflow/core_v0.py::low_freq_mask` and `shiftflow/core_v1.py::reconstruct_lowpass`, then density via `density_from_components`.
- Initial conditions: `shiftflow/cases.py::vortex_case`.

## Actual Feature / Target Definitions

- Shadow features: flattened real/imaginary parts of the two-component reduced coherence matrices `Z_1(t)` and `Z_2(t)`, where `Z_c(t) = b_{K,c}(t) b_{R,c}(t)^*` from the existing reduced ShadowFluid evolution.
- Low-pass baseline features: the task-level low-pass density field at time `t`, reconstructed from the exact Fourier coefficients restricted to the same cutoff `K0 = 4`. This is used only as an input representation, not as a separate forecasting method.
- Downstream target: next-step unresolved high-frequency energy `E_HF(t+dt) = sum_{||k|| > K0} (|b_1(k, t+dt)|^2 + |b_2(k, t+dt)|^2)`.

## Setup Actually Used

- Benchmark: single-cosine potential with `q = (1, 0)`.
- Grid: `nx = 4`, `N = 16`.
- Cutoff: `K0 = 4`.
- Time settings: `t in {0, 0.2, 0.4, 0.6}`, `dt = 0.2`.
- Coupling strengths: `alpha in {0.2, 0.6, 1}`.
- Initial-condition family: Gaussian vortex only, but with deterministic `seed in {0, 1, 2, 3, 4, 5, 6, 7}` to reach enough samples for train sizes 16/32/64 while staying within one family.
- Numerical precision: states/coherences in `complex128`, observables/ML features in `float64`.

## Expected Runtime / Cost

- Very small CPU job. The full benchmark uses `N=16`, three alphas, four time points, and eight deterministic seeds for a total of 96 supervised samples.
- The only moderately expensive step is building reduced propagators via `scipy.linalg.expm`, but they are cached per `(alpha, t)`.