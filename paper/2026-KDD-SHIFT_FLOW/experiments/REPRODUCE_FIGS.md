# Reproduce Figures (CLI)

All commands below are intended to run from the repository root.

## Data Generation

Main sweep (writes `results/sweep.csv`):

```bash
python3 experiments/run_sweep.py --overwrite
```

Main sweep with Aer timing for **shadow** evolution (writes `results/sweep.csv`):

```bash
python3 experiments/run_sweep.py --overwrite --shadow-backend aer
```

Full-state Aer timing only (recommended separate run; writes `results/full_aer_times.csv`):

```bash
python3 experiments/run_sweep.py --overwrite --record-full-aer-times --shadow-backend statevector --K0-list 2.5 --out results/full_aer_times.csv
```

Qiskit baseline spot-check (writes `results/qiskit_spotcheck.csv`, requires `qiskit-aer`):

```bash
python3 experiments/run_qiskit_spotcheck.py --from-sweep results/sweep.csv --n 12 --overwrite
```

## Main Figures (`figs/`)

Exp1:

```bash
python3 experiments/plot_exp1.py
```

Exp2:

```bash
python3 experiments/plot_exp2.py
```

Exp3:

```bash
python3 experiments/plot_exp3.py
```

Exp4 (Qiskit spot-check):

```bash
python3 experiments/plot_exp4.py
```

Exp5:

```bash
python3 experiments/plot_exp5.py
```

Exp6:

```bash
python3 experiments/plot_exp6.py
```

Exp7:

```bash
python3 experiments/plot_exp7.py
```

Aer timing figure (shadow vs full; needs `results/full_aer_times.csv`):

```bash
python3 experiments/plot_aer_runtime.py --shadow results/sweep.csv --full results/full_aer_times.csv
```

## Style Variants / Design Exploration

Exp1: 8 chart-type variants:

```bash
python3 experiments/plot_exp1_8variants.py
```

Exp1 var4: 8 palette variants:

```bash
python3 experiments/plot_exp1_var4_palettes.py
```

Exp1 var4: 8 Nature-like palette variants:

```bash
python3 experiments/plot_exp1_var4_palettes_nature.py
```

## Academic-style v2 Figures (`figs_v2/`)

```bash
python3 experiments/plot_v2_exp1.py
python3 experiments/plot_v2_exp2.py
python3 experiments/plot_v2_exp3.py
python3 experiments/plot_v2_exp4.py
python3 experiments/plot_v2_exp5.py
python3 experiments/plot_v2_exp6.py
python3 experiments/plot_v2_exp7.py
python3 experiments/plot_v2_exp8.py
```
