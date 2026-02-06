# SHIFT-FLOW Experiments: Figure Inventory and Coverage Notes

This note describes what each current figure plots, what it supports in the
paper narrative, and what is still missing / could be strengthened.

Data sources:
- Main sweep: `results/sweep.csv`
- Full-state Aer timing: `results/full_aer_times.csv`
- FFT vs Qiskit baseline spot-check: `results/qiskit_spotcheck.csv`

Key comparisons (as recorded in `results/sweep.csv`):
- `err_*` = shadow vs baseline low-pass (sanity; should be ~ machine precision at V=0)
- `err_*_vs_full` = shadow vs baseline full (main truncation-to-full story)
- `err_*_lp_vs_full` = baseline low-pass vs baseline full (best possible under truncation)

## Current Main Figures (`figs/`)

### `figs/exp1_accuracy_vs_K0_M.pdf`
- Plots: accuracy at an evaluation time (default: max `t`)
  - x-axis: K0 (tick labels include M)
  - y-axis: log scale
  - panels: `err_rho_vs_full` and `err_momentum_vs_full` (mean +/- std across seeds)
- Purpose: show that keeping more modes (larger K0/M) improves agreement with the
  full solution.
- Expected trend: errors decrease as K0/M increases; momentum error often
  decreases more slowly than density because it involves derivatives.

### `figs/exp2_pareto_error_vs_q_shift.pdf`
- Plots: Pareto-style trade-off between error and cost.
  - x-axis: `q_shift` (or `M` via CLI option in v2 scripts)
  - y-axis: combined error `max(err_rho_vs_full, err_momentum_vs_full)` (log scale)
- Purpose: make the KDD-style “error vs resource” argument explicit.
- Expected trend: moving right (higher cost) should reduce error.

### `figs/exp3_scaling_nx_runtime_cost.pdf`
- Plots: scaling with system size `nx` at fixed K0.
  - left: runtime proxies (baseline full FFT, baseline low-pass FFT, shadow,
    total) vs nx (log y)
  - right: qubit proxies `q_base` vs `q_shift` vs nx
- Purpose: show scaling separation:
  - `q_base = 2*nx+1` grows with nx
  - at fixed physical cutoff (fixed K0), M is nearly constant, so `q_shift`
    changes weakly.
- Notes: these runtimes are wall-clock for the Python pipeline, not hardware.

### `figs/exp4_qiskit_spotcheck.pdf`
- Plots: FFT baseline vs Qiskit baseline agreement on a small sample.
  - error panels: `err_*_fft_vs_qiskit`
  - runtime: `rt_fft_s` vs `rt_qiskit_s`
- Purpose: credibility check that FFT and Qiskit implement the same V=0 physics
  (errors should be ~1e-15).

### `figs/exp5_task_only_E_LP_error.pdf`
- Plots: task-only metric accuracy.
  - y: `err_E_LP` where `E_LP = sum_mask(|b1|^2 + |b2|^2)`
- Purpose: demonstrate a task-only evaluation path that does not require full
  field reconstruction.
- Note: under V=0 and ideal simulation, `E_LP` is time-invariant, so this is
  more of a pipeline / sanity check.

### `figs/exp6_multicase_boxplot_nx6.pdf`
- Plots: seed robustness at fixed nx and evaluation time.
  - per K0: distribution across seeds for `err_rho_vs_full` and
    `err_momentum_vs_full`
- Purpose: show robustness and variance under controlled IC perturbations.
- Expected trend: larger K0/M lowers the median and usually tightens the
  distribution.

### `figs/exp7_error_vs_time_nx6.pdf`
- Plots: error vs time curves for several K0 at fixed nx.
  - panels: `err_rho_vs_full(t)` and `err_momentum_vs_full(t)`
  - aggregation: mean across seeds
- Purpose: show whether truncation error grows with time and how K0 changes the
  temporal behavior.

### `figs/exp_aer_runtime_shadow_vs_full.pdf`
- Plots: AerSimulator statevector timing proxy for *evolution*.
  - left: Aer time for shadow (varies with M) vs full (horizontal lines per nx)
  - right: speedup = full/shadow (linear y-axis with an inset zoom)
- Purpose: show that the shadow evolution (smaller qubit count `q_shift`) is
  faster to simulate/execute than full-state evolution (`q_base`) and that the
  speedup increases with nx.
- Notes:
  - This compares Aer *run time* for a diagonal time-evolution in the Fourier
    basis. It is a simulator wall-clock proxy, not a hardware wall time.
  - A more hardware-faithful supplement is to report transpiled depth and gate
    counts.

## Styling / Design Exploration Outputs

- Exp1 8 chart-type variants: `figs_exp1_8/exp1_var*.pdf`
- Exp1 var4 palette grid (8 palettes): `figs_exp1_var4_palettes/*.pdf`
- Exp1 var4 “Nature-like” palette grid (8 palettes): `figs_exp1_var4_nature/*.pdf`
- Academic-style v2 plots: `figs_v2/*.pdf`

## Is the current experimental set “complete”?

Coverage is good for:
- Accuracy vs truncation (Exp1), Pareto trade-off (Exp2), scaling (Exp3),
  robustness (Exp6), and time dependence (Exp7).
- Credibility checks: FFT vs Qiskit baseline (Exp4).
- Additional NISQ-motivated proxy: Aer evolution-time comparison.

Main gaps / recommended improvements (in priority order):

1) Plot the truncation baseline explicitly in main figures
- The CSV already contains `err_*_lp_vs_full` ("best possible under truncation").
- Recommendation: overlay `low-pass vs full` as a dashed curve in Exp1/Exp7 and
  add a ratio panel `err_shadow_vs_full / err_lp_vs_full` (should be ~1). This
  makes it obvious that shadow is truncation-optimal (no extra error).

2) Add hardware-faithful circuit metrics (beyond Aer wall-clock)
- Report transpiled `depth`, `count_ops()`, and 2Q gate count for shadow and
  full (at representative points). This is more stable across machines than
  wall-clock and will read well in KDD.

3) Add a minimal “finite-shot” or noise-model evaluation (optional but strong)
- Keep it task-focused (avoid tomography). For example:
  - evaluate `E_LP` under finite shots
  - or add a simple depolarizing noise model and compare degradation of full vs
    shadow as `q_base` grows.

4) Extend beyond V=0 (if needed for narrative strength)
- V=0 is an ideal correctness/closure regime. To demonstrate the full
  operator-closure story, consider a simple V(x,y) that couples k-modes and a
  controlled closure hierarchy.
