# ShadowFluid

Code for *"ShadowFluid: Operator-First Quantum Simulation of Fluid Dynamics via Shadow Hamiltonians"* (KDD 2026 AI4Science).

Most quantum fluid simulations evolve the full wavefunction and then measure everything at the end -- but in practice, you often only care about a few low-frequency density modes or energy statistics. ShadowFluid flips the order: pick the observables you need first, build a small "coherence dictionary" of rank-one operators in Fourier space, and evolve just that dictionary under a truncated shadow Hamiltonian. The truncation error is controlled by a commutator leakage residual that you can compute *before* running anything, and it grows linearly in time (not exponentially) because unitary conjugation preserves the Frobenius norm.

## Setup

Tested with **Python 3.9** and Qiskit 2.0.1 (see `requirements.txt` for full pinned versions). Other versions may work but are not tested.

```bash
conda create -n shadowfluid python=3.9 -y
conda activate shadowfluid
pip install -r requirements.txt
```

## Reproduce figures

Pre-computed sweep data is in `data/`, so you can plot directly:

```bash
python figures/plot_v0_sanity.py      # -> figures/sanity_density.pdf
python figures/plot_v1_all.py         # -> figures/*.pdf  (5 figures)
```

To regenerate from scratch (~30 min, single CPU):

```bash
python experiments/run_sweep.py --overwrite   # writes data/sweep_v1.csv
```

## Code layout

- `shiftflow/core_v0.py` -- V=0 (diagonal Hamiltonian): free Fourier evolution and shadow coherence method. Dictionary closes exactly here, so shadow evolution matches the full low-pass baseline to machine precision. (Sec 3)
- `shiftflow/core_v1.py` -- V!=0 (off-diagonal coupling): Galerkin-truncated shadow dynamics with multi-reference dictionary and BFS closure on the coupling graph. Includes the leakage bound computation. (Sec 4)
- `shiftflow/metrics.py` -- error metrics: density error, Frobenius-norm delta-Z, low-pass energy error, a priori leakage. (Sec 5)
- `shiftflow/cases.py` -- initial conditions (Gaussian vortex, deterministic seed)
- `shiftflow/qiskit_shadow_v0.py` -- Qiskit circuit for V=0 shadow evolution
- `shiftflow/qiskit_shadow_v1.py` -- Qiskit circuit for V!=0 shadow evolution
- `experiments/run_sweep.py` -- parameter sweep over alpha, K0, t, nx

## Figures -> paper

- `sanity_density.pdf` -> Fig 3: V=0 sanity check (shadow vs low-pass at machine precision)
- `error_vs_K0.pdf` -> Fig 4: error vs frequency cutoff K0 under varying coupling alpha
- `three_curves.pdf` -> Fig 5: error hierarchy (a priori bound >= ||delta-Z|| >= density error >= task error)
- `error_vs_time.pdf` -> Fig 6: temporal stability (linear growth, no exponential blowup)
- `error_vs_alpha.pdf` -> Fig 7: coupling strength sweep (leakage scales linearly, density error stays flat)

## Known issues

- When the operator dictionary is not closed under commutation with the Hamiltonian (i.e., the invariance property does not hold exactly), leakage is unavoidable. ShadowFluid handles this by providing an a priori computable leakage bound; you can also expand the dictionary (larger K0 or deeper BFS closure) to reduce leakage, at the cost of increased dimension.
- The sweep runner is single-threaded for reproducibility. Parallelizing across alpha values is straightforward but not implemented.

## License

This code is provided for review purposes.
