# Shadow Simulation POC

Proof-of-concept for "shadow simulation" using low-frequency Fourier features of density, comparing against baseline full-state quantum evolution.

## Overview

This POC demonstrates that low-frequency dynamics of quantum fluid density can be approximated by a linear reduced model, potentially offering computational speedups for certain observables.

## Structure

```
poc/
├── __init__.py          # Package init
├── baseline.py          # Quantum evolution using Qiskit (from circuit_2D.py)
├── init_states.py       # Gaussian wavepacket initial state
├── features.py          # Fourier feature extraction and reconstruction
├── fit_generator.py     # Fit linear generator matrix A with ridge regression
├── rollout.py           # Rollout shadow dynamics using matrix exponential
├── run_poc.py           # Main entry point
└── README.md            # This file
```

## Configuration

- Grid: 16x16 (nx=4, ny=4 qubits)
- Statevector size: 512 (2×16×16)
- Initial state: Gaussian wavepacket with plane wave modulation
  - σ=0.8, (x₀,y₀)=(0,0), (k₀ₓ,k₀ᵧ)=(2,1)
- Time points: t ∈ {0.0, 0.1, ..., 0.9}
- Fourier features: K=2 (5×5 low-frequency block, 50 real features)

## Running

From the `test/` directory:

```bash
python -m poc.run_poc
```

## Outputs

Results are saved to `test/outputs/`:

- `errors_over_time.png` - Feature, energy, and density reconstruction errors vs time
- `energy_over_time.png` - Low-frequency energy comparison (baseline vs shadow)
- `runtime_summary.txt` - Runtime comparison and error statistics
- `features_full.npy` - Baseline Fourier features at all time points
- `features_shadow.npy` - Shadow-predicted Fourier features

## Method

1. **Baseline**: Run full quantum circuit simulation for each time point using Qiskit's statevector simulator.

2. **Feature extraction**: For each statevector, extract density ρ=|ψ₁|², compute 2D FFT, and take the centered 5×5 low-frequency block.

3. **Shadow model**: Fit a linear generator matrix A via ridge regression:
   ```
   (o_{i+1} - o_i) / dt ≈ A @ o_i
   ```

4. **Rollout**: Predict features using matrix exponential:
   ```
   o(t) = exp(A·t) @ o(0)
   ```

5. **Evaluation**: Compare shadow predictions to baseline using relative L2 errors.

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Qiskit
- Qiskit-Aer
