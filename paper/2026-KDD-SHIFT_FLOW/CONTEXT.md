# SHIFT-FLOW Repo Context (2026-02-06)

This markdown captures the working context and key decisions from the current
development session.

## Goal (KDD-style)

Build a reproducible experiment pipeline for SHIFT-FLOW:

- **Baseline (V=0)**: full-state free evolution of a 2D two-component
  wavefunction on a periodic grid.
- **SHIFT-FLOW shadow**: truncated-mode / shadow-observable dynamics
  (low-frequency Fourier subset `S` / mask) and reconstruction of low-pass
  fields/metrics.
- **NOT classical shadow tomography**: no randomized Pauli/Clifford
  measurements, no tomography estimator.

## Source of Truth

- Reference implementation: `test/shadow_test_v4.py`.
- Refactor must preserve the math/outputs.

Note: `test/shadow_test_v4.py` was updated so Qiskit imports are **optional**
(imported inside functions). This allows importing the module even if
`qiskit-aer` is not installed.

## Core Definitions

- Grid: `nx` qubits per dimension => `N = 2**nx` points per axis.
- Two-component state: `psi1(x,y)`, `psi2(x,y)` complex arrays of shape `(N,N)`.
- Density: `rho = |psi1|^2 + |psi2|^2`.
- Momentum/current (v4 definition):
  - Periodic central differences with `dx = dy = 2*pi/N`.
  - `J = Im(psi* grad psi)` summed over both components.
  - We treat momentum fields as `(Jx, Jy)`.
- Vorticity (v4 definition):
  - `u = J/rho` (with epsilon floor)
  - `omega = d uy/dx - d ux/dy` (periodic central differences).
- Fourier convention (unitary, matching QFT normalization):
  - `b = fft2(psi) / N`
  - `psi = ifft2(b) * N`
- Free Hamiltonian spectrum (integer-k convention):
  - `kx, ky = fftfreq(N)*N`
  - `E(k) = (kx^2 + ky^2)/2`
- Low-frequency mask (circular): keep modes with `sqrt(kx^2+ky^2) <= K0`.

## Implemented Modules

### Task A (refactor + FFT baseline + sanity test)

- `shiftflow/core_v0.py`
  - Refactored math from v4 (IC, unitary FFT/IFFT, mask, energy grid,
    coherence-shadow evolution, density/current/vorticity).
  - Added FFT baseline evolution for sweeps:
    - `evolve_components_fft_v0(psi1_0, psi2_0, t, E)`.
  - Qiskit statevector baseline kept as optional spot-check:
    - `evolve_statevector_v0(nx, ny, t, initial_state)`.

- `tests/sanity_core.py`
  - Validates core vs v4 on `nx=6, K0=2.5, t=0.3, seed=0`.
  - Main validation uses FFT (no Qiskit required).
  - If `qiskit-aer` is installed, also compares FFT baseline vs Qiskit baseline.

### Task B (metrics + cases)

- `shiftflow/metrics.py`
  - `rel_l2`, `rel_l2_vec`
  - errors from components:
    - `err_rho_from_components`
    - `err_momentum_from_components` (vector L2 on `(Jx,Jy)`)
    - optional `err_omega_from_components`
  - task-only metric:
    - `E_LP_from_coeffs(b1,b2,mask) = sum_mask(|b1|^2+|b2|^2)`
  - cost proxies:
    - `q_base = 2*nx + 1`
    - `q_shift = ceil(log2(M)) + 1` (assumes compressed encoding of the M kept modes)
    - `cost_proxies(...)` returns `CostProxies`.

- `shiftflow/cases.py`
  - Deterministic IC generator based on v4 vortex IC:
    - `vortex_case(nx, seed, ...)`
  - Seeded variations: sigma jitter, periodic roll shift, tiny complex noise.
  - Global normalization across both components.

### Task C (sweep runner)

- `experiments/run_sweep.py`
  - Loops `(nx, K0, t, seed)` and writes `results/sweep.csv`.
  - Sweeps use FFT baseline (fast) and coherence-shadow for the truncated set.
  - Comparisons are recorded for:
    - **shadow vs low-pass baseline** (sanity; should be ~ machine precision for V=0)
    - **shadow vs full baseline** (main truncation-to-full story)
    - **low-pass baseline vs full baseline** (best possible under truncation)
  - Also records task-only `E_LP` error and cost/runtime proxies.

Current defaults:

- `nx_list = [5,6,7]`
- `t_list = 0.0..1.0 step 0.1`
- `seeds = [0..4]`
- **K0 list was doubled to 8 groups**:
  - `K0_list = [2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]`

CSV columns (high level):

- identifiers: `nx,N,K0,M,t,seed` + case meta
- k0 references: `k0_1_*`, `k0_2_*`
- errors:
  - `err_rho`, `err_momentum`, `err_omega` (shadow vs low-pass)
  - `err_*_vs_full` (shadow vs full)
  - `err_*_lp_vs_full` (low-pass vs full)
- task-only: `E_LP_base`, `E_LP_shadow`, `err_E_LP`
- cost: `q_base,q_shift,measurement_proxy,postprocess_*`
- runtimes: `rt_baseline_full_s, rt_baseline_lp_s, rt_shadow_s, rt_metrics_s, rt_total_s`

### Qiskit credibility spot-check (recommended for paper)

- `experiments/run_qiskit_spotcheck.py`
  - Requires `qiskit-aer`.
  - Samples a small number of points and compares FFT baseline vs Qiskit
    baseline on `rho`, `J`, and optional `omega`.
  - Writes `results/qiskit_spotcheck.csv`.

## Plotting

Plot style guidance: `experiments/plot.md`.

Two sets of plot scripts exist:

- "exp*" scripts (initial set): `experiments/plot_exp1.py`, ...
- "v2" scripts (more academic styling): `experiments/plot_v2_exp1.py`, ...

Additionally, Exp1 exploration:

- `experiments/plot_exp1_8variants.py`
  - Generates 8 design variants for Exp1 under `figs_exp1_8/`.
- `experiments/plot_exp1_var4_palettes.py`
  - Generates 8 palette options for the grouped-bar style.
- `experiments/plot_exp1_var4_palettes_nature.py`
  - Generates 8 restrained, Nature-like palettes.

Generated figure directories are outputs only and can be ignored on another
machine:

- `figs/`, `figs_v2/`, `figs_exp1_8/`, `figs_exp1_var4_palettes/`,
  `figs_exp1_var4_nature/`

## How to Reproduce (CLI)

Sanity (core vs v4):

```bash
python3 tests/sanity_core.py
```

Run the full sweep:

```bash
python3 experiments/run_sweep.py --overwrite
```

Generate Exp1 8 design variants:

```bash
python3 experiments/plot_exp1_8variants.py
```

Generate Exp1 var4 palette options:

```bash
python3 experiments/plot_exp1_var4_palettes.py
python3 experiments/plot_exp1_var4_palettes_nature.py
```

Qiskit spot-check (optional, requires qiskit-aer):

```bash
python3 experiments/run_qiskit_spotcheck.py --from-sweep results/sweep.csv --n 12 --overwrite
python3 experiments/plot_exp4.py
```

## Key Interpretation Notes (Route A)

- Under V=0 and a fixed physical cutoff `K0`, the truncation error to the full
  solution is driven mainly by `K0/M` and can be weakly dependent on `nx`.
  This is expected when the initial condition has a fixed physical bandwidth
  (sigma in physical units) and `K0 << N/2`.
- FFT baseline vs Qiskit baseline differences should be ~ machine precision.

## Open / Next Work

- Produce 8 design variants for Exp2..Exp7 (analogous to Exp1).
- Decide final paper styling (fonts/palette).
- Consider adding a **noise model** (gate/decoherence or shot noise) as an
  additional section if needed for narrative, without drifting into classical
  shadow tomography.
