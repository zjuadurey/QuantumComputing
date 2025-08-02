"""
Compute chosen metrics of <method> w.r.t. baseline (classic).
"""
import csv, numpy as np
from pathlib import Path
from ..stateprep import single_vortex
from ..spec_classic import SpectralSolver
from ..solvers.rk4_realspace import RK4Solver            # 示例：换别的也 OK
from ..core.metrics import mse_density, l2_relative, fidelity

# ---- parameters ----
METHOD = RK4Solver()                # 可改成 QuantumSpectralSolver() 等
T_GRID = np.linspace(0, 1.5, 7)
SIGMA, CENTER = 2.5, (1.0, -1.0)
OUT = Path("metric_table.csv")

# ---- run ----
ψ1_0, ψ2_0 = single_vortex(SIGMA, CENTER)
baseline   = SpectralSolver().evolve(ψ1_0, ψ2_0, T_GRID)
peer       = METHOD.evolve(ψ1_0, ψ2_0, T_GRID)

rows = []
for t in T_GRID:
    ψb = baseline[t]
    ψp = peer[t]
    rows.append({
        "t": t,
        "mse_density": mse_density(ψp, ψb),
        "l2_relative": l2_relative(ψp, ψb),
        "fidelity":    fidelity(ψp, ψb)
    })

with OUT.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print("✓ metrics →", OUT)
