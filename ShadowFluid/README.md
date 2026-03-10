# ShadowFluid

**ShadowFluid: Operator-First Quantum Simulation of Fluid
Dynamics via Shadow Hamiltonians**

> Companion code for the KDD 2026 AI4Science submission.

---

## Quick Start

```bash
pip install -r requirements.txt
```

## Reproduce Figures

All figures can be reproduced from the pre-computed data in `data/`:

```bash
python figures/plot_v0_sanity.py      # → figures/v0_sanity_density.pdf
python figures/plot_v1_all.py         # → figures/v1_*.pdf  (4 figures)
```

## Run Experiments from Scratch

To regenerate the data (takes ~30 min on a single CPU core):

```bash
python experiments/run_sweep.py --overwrite
```

This writes `data/sweep_v1.csv`, which is also shipped pre-computed.

## Code Structure

| Module | Description | Paper Section |
|---|---|---|
| `shiftflow/core_v0.py` | V=0 free evolution, Fourier tools, shadow coherence method | Sec 3 |
| `shiftflow/core_v1.py` | V≠0 Galerkin-truncated evolution, leakage bounds | Sec 4 |
| `shiftflow/metrics.py` | Error metrics and cost proxies | Sec 5 |
| `shiftflow/cases.py` | Initial condition generator (vortex ICs) | Sec 5 |
| `shiftflow/qiskit_shadow_v0.py` | Qiskit circuit for V=0 shadow evolution | Sec 3 |
| `shiftflow/qiskit_shadow_v1.py` | Qiskit circuit for V≠0 shadow evolution | Sec 4 |
| `experiments/run_sweep.py` | Main experiment sweep runner | Sec 5 |

## Figure-to-Paper Mapping

| Output PDF | Paper Figure | Research Question |
|---|---|---|
| `figures/v0_sanity_density.pdf` | Fig 3 | RQ1: V=0 sanity check |
| `figures/v1_error_vs_K0.pdf` | Fig 4 | RQ2: Error vs truncation |
| `figures/v1_three_curves.pdf` | Fig 5 | RQ3: Bound vs actual error |
| `figures/v1_error_vs_time.pdf` | Fig 6 | RQ4: Temporal stability |
| `figures/v1_error_vs_alpha.pdf` | Fig 7 | RQ5: Coupling strength |

## License

This code is provided for review purposes.
